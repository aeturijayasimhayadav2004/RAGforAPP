import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

JOB_MAPPING_PATH = DATA_DIR / "job_metadata.json"
USER_PROFILE_PATH = DATA_DIR / "user_profiles.json"
INDEX_PATH = DATA_DIR / "jobs.index"

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

LINKEDIN_API_KEY = os.getenv("LINKEDIN_API_KEY")
NAUKRI_API_KEY = os.getenv("NAUKRI_API_KEY")
INDEED_API_KEY = os.getenv("INDEED_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Job Matching RAG Service", version="0.1.0")

external_job_search = ExternalJobSearch(
    providers=[
        ExternalJobProvider(
            name="linkedin",
            api_key=LINKEDIN_API_KEY,
            base_url=os.getenv("LINKEDIN_API_BASE", "https://api.linkedin.com/v2/jobSearch"),
            id_key="id",
            title_key="title",
            company_key="company",
            description_key="description",
            location_key="location",
        ),
        ExternalJobProvider(
            name="naukri",
            api_key=NAUKRI_API_KEY,
            base_url=os.getenv("NAUKRI_API_BASE", "https://api.naukri.com/jobs/search"),
            id_key="jobId",
            title_key="title",
            company_key="companyName",
            description_key="jobDescription",
            location_key="location",
            job_type_key="employmentType",
            salary_key="salary",
            url_key="jobUrl",
        ),
        ExternalJobProvider(
            name="indeed",
            api_key=INDEED_API_KEY,
            base_url=os.getenv("INDEED_API_BASE", "https://api.indeed.com/v2/search"),
            id_key="job_id",
            title_key="title",
            company_key="company",
            description_key="description",
            location_key="location",
            salary_key="salary",
            url_key="url",
        ),
    ]
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected_key = os.getenv("API_AUTH_KEY")
    if expected_key and x_api_key != expected_key:
        logger.warning("Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


class Job(BaseModel):
    job_id: int = Field(..., description="Unique job identifier")
    title: str
    company: str
    location: Optional[str] = None
    description: str
    job_type: Optional[str] = None
    salary_range: Optional[str] = None
    source_url: Optional[str] = None
    is_internal: bool = False
    is_active: bool = True
    created_at: Optional[str] = None

    @validator("title", "company", "description")
    def required_strings(cls, value: str) -> str:
        if not value:
            raise ValueError("Field cannot be empty")
        return value


class JobsBulkRequest(BaseModel):
    jobs: List[Job]


class UserProfile(BaseModel):
    user_id: int
    name: str
    headline: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience_summary: Optional[str] = None
    education_summary: Optional[str] = None
    preferred_roles: List[str] = Field(default_factory=list)
    preferred_locations: List[str] = Field(default_factory=list)
    preferred_salary_min: Optional[str] = None


class AutoApplyJob(BaseModel):
    job_id: int
    title: str
    company: str
    location: Optional[str] = None
    description: str
    is_internal: bool = False
    source_url: Optional[str] = None


class AutoApplyRequest(BaseModel):
    user_id: int
    job: AutoApplyJob


@dataclass
class ExternalJobProvider:
    name: str
    api_key: Optional[str]
    base_url: str
    id_key: str
    title_key: str
    company_key: str
    description_key: str
    location_key: str = "location"
    job_type_key: str = "job_type"
    salary_key: str = "salary_range"
    url_key: str = "source_url"

    def _request(self, query: str, limit: int) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            response = requests.get(
                self.base_url,
                params={"q": query, "limit": limit},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "results" in payload:
                results = payload.get("results", [])
            else:
                results = payload if isinstance(payload, list) else []
            return [item for item in results if isinstance(item, dict)]
        except Exception as exc:
            logger.warning("%s search failed: %s", self.name, exc)
            return []

    def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        items = self._request(query, limit)
        normalized: List[Dict[str, Any]] = []
        for item in items:
            normalized_item = self._normalize_item(item)
            if normalized_item:
                normalized.append(normalized_item)
        return normalized[:limit]

    def _normalize_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            job_id_raw = item.get(self.id_key)
            if job_id_raw is None:
                return None
            job_id = int(job_id_raw)
            title = str(item.get(self.title_key, "")).strip()
            company = str(item.get(self.company_key, "")).strip()
            description = str(item.get(self.description_key, "")).strip()
            if not (title and company and description):
                return None
            return {
                "job_id": job_id,
                "title": title,
                "company": company,
                "location": item.get(self.location_key),
                "description": description,
                "job_type": item.get(self.job_type_key),
                "salary_range": item.get(self.salary_key),
                "source_url": item.get(self.url_key),
                "is_internal": False,
                "is_active": True,
            }
        except (TypeError, ValueError):
            logger.debug("Skipping malformed job item from %s", self.name)
            return None


class ExternalJobSearch:
    def __init__(self, providers: Iterable[ExternalJobProvider]):
        self.providers = list(providers)

    def configured_providers(self) -> List[str]:
        return [provider.name for provider in self.providers if provider.api_key]

    def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for provider in self.providers:
            if not provider.api_key:
                logger.debug("Skipping %s search; API key not configured", provider.name)
                continue
            provider_results = provider.search(query, limit)
            results.extend(provider_results)
        deduped = self._dedupe_by_job_id(results)
        return deduped[:limit]

    @staticmethod
    def _dedupe_by_job_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for item in items:
            job_id = item.get("job_id")
            if job_id in seen:
                continue
            seen.add(job_id)
            unique.append(item)
        return unique


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_job_text(job: Job) -> str:
    parts = [
        f"Job Title: {job.title}",
        f"Company: {job.company}",
        f"Location: {job.location or 'N/A'}",
        f"Description: {job.description}",
    ]
    if job.job_type:
        parts.append(f"Job Type: {job.job_type}")
    if job.salary_range:
        parts.append(f"Salary Range: {job.salary_range}")
    return "\n".join(parts)


def build_user_profile_text(profile: UserProfile) -> str:
    return "\n".join(
        [
            f"Name: {profile.name}",
            f"Headline: {profile.headline or ''}",
            f"Skills: {', '.join(profile.skills)}",
            f"Experience: {profile.experience_summary or ''}",
            f"Education: {profile.education_summary or ''}",
            f"Preferred roles: {', '.join(profile.preferred_roles)}",
            f"Preferred locations: {', '.join(profile.preferred_locations)}",
            f"Preferred salary min: {profile.preferred_salary_min or ''}",
        ]
    )


def to_job_model(job_data: Dict[str, Any]) -> Optional[Job]:
    try:
        return Job(**job_data)
    except ValidationError as exc:
        logger.warning("Skipping invalid job payload: %s", exc)
        return None


def get_embedding(texts: List[str]) -> List[List[float]]:
    if not texts:
        raise HTTPException(status_code=400, detail="No text provided for embedding")
    response = client.embeddings.create(model=DEFAULT_EMBEDDING_MODEL, input=texts)
    return [record.embedding for record in response.data]


def load_faiss_index(dimension: int) -> faiss.IndexIDMap:
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
        if index.d != dimension:
            logger.warning(
                "FAISS index dimension %s does not match expected %s; recreating index",
                index.d,
                dimension,
            )
            inner_index = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIDMap(inner_index)
        return index
    inner_index = faiss.IndexFlatIP(dimension)
    return faiss.IndexIDMap(inner_index)


def save_faiss_index(index: faiss.IndexIDMap) -> None:
    faiss.write_index(index, str(INDEX_PATH))


def normalize_embeddings(embeddings: List[List[float]]) -> np.ndarray:
    arr = np.array(embeddings).astype("float32")
    faiss.normalize_L2(arr)
    return arr


def upsert_job_embedding(job: Job, embedding: List[float]) -> None:
    job_metadata = load_json(JOB_MAPPING_PATH)
    dimension = len(embedding)
    index = load_faiss_index(dimension)

    vector = normalize_embeddings([embedding])
    if str(job.job_id) in job_metadata:
        index.remove_ids(np.array([job.job_id], dtype="int64"))

    index.add_with_ids(vector, np.array([job.job_id], dtype="int64"))
    save_faiss_index(index)

    job_metadata[str(job.job_id)] = {"job": job.dict()}
    save_json(JOB_MAPPING_PATH, job_metadata)


def upsert_job_embeddings_bulk(jobs: List[Job], embeddings: List[List[float]]) -> None:
    if not jobs:
        return
    job_metadata = load_json(JOB_MAPPING_PATH)
    dimension = len(embeddings[0])
    index = load_faiss_index(dimension)

    ids_to_remove = [job.job_id for job in jobs if str(job.job_id) in job_metadata]
    if ids_to_remove:
        index.remove_ids(np.array(ids_to_remove, dtype="int64"))

    vectors = normalize_embeddings(embeddings)
    ids = np.array([job.job_id for job in jobs], dtype="int64")
    index.add_with_ids(vectors, ids)
    save_faiss_index(index)

    for job in jobs:
        job_metadata[str(job.job_id)] = {"job": job.dict()}
    save_json(JOB_MAPPING_PATH, job_metadata)


def load_user_profiles() -> Dict[str, Any]:
    return load_json(USER_PROFILE_PATH)


def save_user_profiles(profiles: Dict[str, Any]) -> None:
    save_json(USER_PROFILE_PATH, profiles)


def get_user_profile_embedding(user_id: int) -> Dict[str, Any]:
    profiles = load_user_profiles()
    profile = profiles.get(str(user_id))
    if not profile:
        raise HTTPException(status_code=400, detail="User profile not found")
    return profile


def get_job_metadata(job_id: int) -> Optional[Dict[str, Any]]:
    mapping = load_json(JOB_MAPPING_PATH)
    return mapping.get(str(job_id))


def search_jobs_for_user(user_profile: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
    embedding = user_profile["embedding"]
    if not INDEX_PATH.exists():
        return []

    dimension = len(embedding)
    index = load_faiss_index(dimension)
    if index.ntotal == 0:
        return []

    vector = normalize_embeddings([embedding])
    scores, ids = index.search(vector, top_k)
    job_ids = ids[0]
    similarities = scores[0]
    mapping = load_json(JOB_MAPPING_PATH)

    results = []
    for job_id, score in zip(job_ids, similarities):
        if job_id == -1:
            continue
        job_info = mapping.get(str(int(job_id)))
        if job_info:
            job_data = job_info.get("job")
            if job_data:
                job_data["similarity"] = float(score)
                results.append(job_data)
    return results


def call_job_match_llm(user_profile_text: str, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = (
        "You are a job matching assistant. Given a user profile and a job description,"
        " provide a strict JSON object with fields match_score (0-100), missing_skills"
        " (array of strings), reason (short text), and recommended (true/false)."
        " Respond with JSON only."
        "\n\nUser profile:\n" + user_profile_text +
        "\n\nJob description:\n" + job.get("description", "")
    )

    try:
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        return json.loads(content)
    except Exception as exc:
        logger.warning("LLM scoring failed for job %s: %s", job.get("job_id"), exc)
        return None


def call_auto_apply_llm(user_profile_text: str, job: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "Generate a personalized cover letter and application summary for the user"
        " applying to the following job. Return a JSON object with fields cover_letter,"
        " application_summary, and structured_fields containing applicant_name, role_title,"
        " and company. User profile and job details are provided below."
        "\n\nUser profile:\n" + user_profile_text +
        "\n\nJob details:\n" + json.dumps(job, ensure_ascii=False)
    )

    response = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content or "{}")
    except json.JSONDecodeError:
        return {
            "cover_letter": content or "",
            "application_summary": "",
            "structured_fields": {},
        }


@app.post("/rag/index_job")
def index_job(job: Job, _: bool = Depends(require_api_key)):
    text = build_job_text(job)
    embedding = get_embedding([text])[0]
    upsert_job_embedding(job, embedding)
    return {"status": "ok", "job_id": job.job_id}


@app.post("/rag/index_jobs_bulk")
def index_jobs_bulk(request: JobsBulkRequest, _: bool = Depends(require_api_key)):
    if not request.jobs:
        raise HTTPException(status_code=400, detail="No jobs provided")
    texts = [build_job_text(job) for job in request.jobs]
    embeddings = get_embedding(texts)
    upsert_job_embeddings_bulk(request.jobs, embeddings)
    return {"status": "ok", "indexed": len(request.jobs)}


@app.post("/rag/update_user_profile")
def update_user_profile(profile: UserProfile, _: bool = Depends(require_api_key)):
    profile_text = build_user_profile_text(profile)
    embedding = get_embedding([profile_text])[0]
    profiles = load_user_profiles()
    profiles[str(profile.user_id)] = {
        "profile_text": profile_text,
        "embedding": embedding,
        "profile": profile.dict(),
    }
    save_user_profiles(profiles)
    return {"status": "ok", "user_id": profile.user_id}


@app.get("/rag/search_external_jobs")
def search_external_jobs(
    query: str = Query(..., description="Free text query used for external providers"),
    limit: int = Query(20, ge=1, le=100),
    index_results: bool = Query(False, description="Upsert fetched jobs into FAISS index"),
    _: bool = Depends(require_api_key),
):
    configured = external_job_search.configured_providers()
    if not configured:
        raise HTTPException(status_code=503, detail="No external providers configured")

    raw_results = external_job_search.search(query, limit)
    job_models: List[Job] = []
    for job_data in raw_results:
        model = to_job_model(job_data)
        if model:
            job_models.append(model)

    if index_results and job_models:
        texts = [build_job_text(job) for job in job_models]
        embeddings = get_embedding(texts)
        upsert_job_embeddings_bulk(job_models, embeddings)

    return {
        "providers": configured,
        "jobs": [job.dict() for job in job_models[:limit]],
    }


@app.get("/rag/suggest_jobs")
def suggest_jobs(
    user_id: int = Query(..., description="User identifier"),
    limit: int = Query(20, ge=1, le=100),
    only_internal: bool = Query(False),
    min_match_score: int = Query(70, ge=0, le=100),
    _: bool = Depends(require_api_key),
):
    user_profile = get_user_profile_embedding(user_id)
    user_profile_text = user_profile["profile_text"]
    candidates = search_jobs_for_user(user_profile, top_k=50)

    filtered = []
    for job in candidates:
        if not job.get("is_active", True):
            continue
        if only_internal and not job.get("is_internal"):
            continue
        match = call_job_match_llm(user_profile_text, job)
        if not match:
            continue
        score = match.get("match_score")
        try:
            score_value = int(score)
        except (TypeError, ValueError):
            logger.debug("Discarding job %s due to invalid score", job.get("job_id"))
            continue
        if score_value < min_match_score:
            continue
        filtered.append({
            "job_id": job["job_id"],
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "match_score": score_value,
            "missing_skills": match.get("missing_skills", []),
            "reason": match.get("reason", ""),
            "is_internal": job.get("is_internal", False),
            "auto_apply_allowed": job.get("is_internal", False),
        })

    filtered.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return {"user_id": user_id, "jobs": filtered[:limit]}


@app.post("/rag/auto_apply")
def auto_apply(request: AutoApplyRequest, _: bool = Depends(require_api_key)):
    user_profile = get_user_profile_embedding(request.user_id)
    user_profile_text = user_profile["profile_text"]
    payload = call_auto_apply_llm(user_profile_text, request.job.dict())
    return {
        "status": "ok",
        "user_id": request.user_id,
        "job_id": request.job.job_id,
        "auto_apply_payload": payload,
    }


@app.exception_handler(HTTPException)
def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
