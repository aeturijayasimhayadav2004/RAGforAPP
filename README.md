# RAGforAPP

FastAPI microservice that powers RAG-based job matching and auto-apply workflows. It exposes JSON APIs for indexing jobs, storing user profiles, suggesting matches, and generating auto-apply payloads.

## Features
- Uses OpenAI embeddings for job and user profiles and stores vectors in a persisted FAISS index on disk.
- LLM-driven scoring for job recommendations with filtering options (internal-only, minimum match score, result limits).
- Auto-apply endpoint that produces cover letters and structured application payloads.
- Optional header-based auth via `X-API-KEY` checked against `API_AUTH_KEY`.

## Setup
1. Ensure Python 3.10+ is available.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables (defaults shown):
   ```bash
   export OPENAI_API_KEY=<your_key>
   export API_AUTH_KEY=<shared_secret>           # optional; if unset, auth is skipped
   export EMBEDDING_MODEL_NAME=text-embedding-3-small
   export LLM_MODEL_NAME=gpt-3.5-turbo
   export LINKEDIN_API_KEY=<linkedin_api_key>    # optional; enables LinkedIn search
   export NAUKRI_API_KEY=<naukri_api_key>        # optional; enables Naukri search
   export INDEED_API_KEY=<indeed_api_key>        # optional; enables Indeed search
   ```

## Running
Start the service:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

## Key Endpoints
- `POST /rag/index_job` – index a single job JSON payload.
- `POST /rag/index_jobs_bulk` – batch indexing for multiple jobs.
- `POST /rag/update_user_profile` – store a profile and embedding for a user.
- `GET /rag/search_external_jobs` – fetch jobs from configured LinkedIn/Naukri/Indeed providers and optionally index them.
- `GET /rag/suggest_jobs` – return ranked job suggestions for a user.
- `POST /rag/auto_apply` – generate cover letter and application details for a user/job pair.

Data and index files are stored under the local `data/` directory (created on startup).
