# Quivr × MemPalace RAG UI

A React and FastAPI application that combines Quivr document retrieval with
MemPalace long-term memory. The application supports Groq and NVIDIA NIM
models and provides a modern animated knowledge workspace.

## Features

- React and Vite frontend
- FastAPI backend
- Quivr semantic document retrieval
- MemPalace memory recall and persistence
- Groq LLM support
- NVIDIA NIM LLM and embedding support
- PDF, DOCX, CSV, TXT, and Markdown files
- OCR fallback for scanned PDFs
- Motion-powered glass, glow, spotlight, and bento-style UI
- Local Windows CMD setup scripts

## Architecture

```text
React + Vite frontend
        |
        v
FastAPI application API
        |
        +-- Quivr Core document RAG
        +-- MemPalace long-term memory
        +-- Groq or NVIDIA NIM
        +-- Local OCR for scanned PDFs
```

## Requirements

- Windows 10 or Windows 11
- Python 3.13
- Node.js 20 or newer
- `uv`
- PyMuPDF4LLM and RapidOCR for PDF extraction, including scanned PDFs

Python 3.14 is not currently supported because one of Quivr Core's language
detection dependencies may require a native build on Python 3.14.

## Project structure

```text
ragb-0.2.0/
├── src/ragb/core/                       Quivr Core source
├── src/ragb/examples/mempalace_rag_ui/  FastAPI and React application
│   ├── app.py                            Backend API
│   ├── frontend/src/                     React source
│   ├── setup.cmd                         Dependency setup
│   ├── run_app.cmd                       Production server
│   └── run_frontend.cmd                  Vite development server
└── examples/mempalace_quivr/             MemPalace integration package
```

## Installation from CMD

Open CMD in the repository root:

```cmd
cd /d C:\path\to\ragb-0.2.0
```

If an existing environment uses Python 3.14, recreate only the generated
virtual environment:

```cmd
deactivate
rmdir /s /q .venv
uv venv .venv --python 3.13
call .venv\Scripts\activate.bat
```

Install the bundled local packages first. This is required because the local
Quivr Core fork is version `0.0.33`, while some package indexes only expose
older public Quivr Core versions.

```cmd
uv pip install --python .venv\Scripts\python.exe --no-deps -e src\ragb\core
uv pip install --python .venv\Scripts\python.exe --no-deps -e examples\mempalace_quivr
uv pip install --python .venv\Scripts\python.exe -r src\ragb\examples\mempalace_rag_ui\requirements.txt
```

Alternatively, run the application setup script after recreating the Python
3.13 environment:

```cmd
cd /d src\ragb\examples\mempalace_rag_ui
setup.cmd
```

## Environment configuration

Copy the example environment file:

```cmd
cd /d src\ragb\examples\mempalace_rag_ui
copy .env.example .env
notepad .env
```

Configure at least one LLM provider:

```text
GROQ_API_KEY=your-groq-key
NVIDIA_API_KEY=your-nvidia-key
```

Useful model settings include:

```text
GROQ_MODEL=llama-3.3-70b-versatile
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b
NVIDIA_EMBEDDING_BATCH_SIZE=50
QUIVR_CHUNK_SIZE=1200
QUIVR_CHUNK_OVERLAP=150
QUIVR_OCR_DPI=150
```

Never commit `.env`, API keys, passwords, or PyPI tokens.

## Build the React frontend

```cmd
cd /d src\ragb\examples\mempalace_rag_ui\frontend
npm install
npm run build
```

The compiled frontend is written to:

```text
src/ragb/examples/mempalace_rag_ui/frontend/dist
```

## Run the application

```cmd
cd /d src\ragb\examples\mempalace_rag_ui
run_app.cmd
```

Open:

```text
http://127.0.0.1:7862
```

For React hot reload, run the API in one CMD window:

```cmd
cd /d src\ragb\examples\mempalace_rag_ui
run_app.cmd
```

Then run Vite in another CMD window:

```cmd
cd /d src\ragb\examples\mempalace_rag_ui
run_frontend.cmd
```

Open:

```text
http://127.0.0.1:5173
```

## Using the application

1. Select Groq or NVIDIA NIM.
2. Upload PDF, DOCX, CSV, TXT, or Markdown files.
3. Click **Index documents**.
4. Ask questions about the uploaded sources.
5. Use **Recall memory** to inspect MemPalace context.
6. Review the retrieval trace below the conversation.

## OCR support

PDFs are parsed with PyMuPDF4LLM. Regular PDFs use fast native text/layout
extraction; pages without selectable text use the bundled RapidOCR ONNX
backend. A system Tesseract installation is not required. Set
`QUIVR_PDF_USE_OCR=false` if you only want to accept PDFs with selectable text.

## Troubleshooting

### Quivr Core version cannot be resolved

Install the bundled local source before installing the requirements:

```cmd
uv pip install --python .venv\Scripts\python.exe --no-deps -e src\ragb\core
```

### `fasttext-predict` build failure

Check the Python version:

```cmd
.venv\Scripts\python.exe --version
```

Use Python 3.13, recreate `.venv`, and repeat the installation.

### React frontend is missing

You are running npm from the wrong directory. The root folder has no `frontend` folder, and `aicctv-databricks-app` is a different project.

Copy these commands exactly:

```cmd
cd /d C:\Users\GauravSarma\Downloads\ragb-0.2.0\src\ragb\examples\mempalace_rag_ui\frontend
npm install
npm run build
```

The build output should begin with:

```text
quivr-mempalace-rag-ui@0.2.0 build
```

Then start the backend:

```cmd
cd /d C:\Users\GauravSarma\Downloads\ragb-0.2.0\src\ragb\examples\mempalace_rag_ui
run_app.cmd
```
### Scanned PDF extraction is unavailable

Run `setup.cmd` again so `pymupdf4llm` and `rapidocr-onnxruntime` are installed.
For faster indexing, use a text-based or already OCR'd PDF. OCR is much slower
than native PDF text extraction.

### Check installed dependencies

```cmd
uv pip check --python .venv\Scripts\python.exe
```

## Development notes

The default local implementation keeps the active Quivr Brain in process
memory. Restarting the application loses the active document index. For a
production deployment, use persistent vector storage and durable MemPalace
storage such as Databricks Vector Search, Delta/Unity Catalog storage, or a
managed database.

## Databricks deployment

The FastAPI application is compatible with a Databricks Apps deployment model:

- Bind the server to `0.0.0.0`.
- Use the `DATABRICKS_APP_PORT` environment variable.
- Store provider keys in Databricks secret resources.
- Replace local FAISS and filesystem persistence with durable storage.
- Move OCR to a Linux-compatible service or pre-ingestion job.

## Render deployment

The root `render.yaml` is configured for a native Python web service. It pins
Python 3.13, installs the bundled Quivr Core and MemPalace packages, builds
the React frontend, pre-caches the smaller Kokoro int8 model, and exposes
`/api/health` for Render HTTP health checks.

Set `GROQ_API_KEY` and/or `NVIDIA_API_KEY` in the Render dashboard before
deploying. Add the admin and tracing variables described below if the admin
console is needed. The indexed Brain is still process-local, `/tmp` storage is
ephemeral, and the free Render plan has limited CPU/RAM and may spin down; use
durable storage, a worker queue, rate limiting, and a paid multi-instance plan
before treating this as a high-volume production service.

## Admin observability and tracing

The second area of the web app is an admin-only observability console. It records
safe metadata for `rag.index`, `memory.recall`, `rag.ask`, and `tts.synthesis`,
including nested retrieval, embedding, generation, and TTS spans. Prompt,
answer, and document content are not captured by default.

Tracing uses [Langfuse](https://langfuse.com/), an open-source/self-hostable
platform included in the [awesome-agent-observability catalog](https://github.com/anhermon/awesome-agent-observability).
Without Langfuse credentials, the console still provides a bounded local trace
buffer for the current process.

### Microsoft SSO setup

1. In Microsoft Entra ID, create an App Registration and add a **Web** redirect
   URI: `https://YOUR-RENDER-SERVICE.onrender.com/admin/callback`.
2. Create a client secret and copy its value immediately.
3. Set these Render environment variables:

   ```text
   MICROSOFT_TENANT_ID=<your tenant GUID or tenant domain>
   MICROSOFT_CLIENT_ID=<application/client ID>
   MICROSOFT_CLIENT_SECRET=<client secret value>
   SESSION_SECRET=<long random value>
   ADMIN_EMAILS=admin@yourcompany.com,another-admin@yourcompany.com
   COOKIE_SECURE=true
   ```

   `ADMIN_EMAILS` is an explicit allowlist. Authentication alone does not grant
   access to the admin console.
4. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` if
   traces should be exported to Langfuse. Keep
   `OBSERVABILITY_CAPTURE_CONTENT=false` unless trace content has been reviewed
   for privacy and compliance.
5. Deploy, then select **Admin observability** in the app header. Unauthenticated
   users are sent through Microsoft sign-in before `/api/admin/observability`
   can return any trace data.

### Temporary fallback access

If Microsoft SSO is not ready, set these Render variables temporarily:

```text
ADMIN_FALLBACK_ENABLED=true
ADMIN_FALLBACK_TOKEN=<long random token, at least 16 characters>
ADMIN_FALLBACK_EMAIL=fallback-admin@yourcompany.com
SESSION_SECRET=<long random session secret>
```

Open **Admin observability** and enter the token. The fallback uses the same
session-protected admin console, but it is disabled by default and should be
turned off as soon as Entra SSO is configured. The token is never placed in a
URL or written to the AGT audit log.

## Agent Governance Toolkit fork

The admin console also exposes the retrieval governance layer from the
[project fork](https://github.com/prj1010/agent-governance-toolkit), pinned to
commit `359a2332f57d9000924baba269ed24e4e15ad8b0`. The packages are installed
directly from their monorepo subdirectories, so a similarly named PyPI package
cannot silently replace the fork.

On Linux/Render the requirements also install the fork's `core[full]` bundle.
On Windows, `setup.cmd` installs the forked `agent-rag-governance` package and
the controls used by this app; the optional core bundle is skipped because its
Rust-backed ACS dependency currently publishes a manylinux wheel rather than a
Windows wheel.

The policy protects every Quivr retriever with collection allow/deny rules,
rate limiting, PII and prompt-injection scanning, and privacy-safe audit
hashes. The admin panel shows the active policy and audit events and includes a
dry-run evaluator at **Agent governance**. Quivr calls the governed toolkit's
synchronous `invoke` method in a threadpool when its internal async graph asks
for documents; this preserves governance while keeping the FastAPI event loop
responsive.

Configure the policy with `AGT_*` variables in `.env` or Render. In production,
keep `AGENT_RAG_AUDIT_SALT` secret, use a durable audit sink if audit history
must survive restarts, and set an explicit `AGT_ALLOWED_COLLECTIONS` list. The
native Render service can use the toolkit's policy and audit features, but
Docker/OS sandbox features need a separate container-capable worker; they are
not available inside a native Render Python service.

## Security

Do not commit:

```text
.env
API keys
passwords
PyPI tokens
private certificates
```

Use GitHub Secrets, Databricks Secret Scopes, or another secret manager for
deployment credentials.

## Roadmap

- Persistent Databricks Vector Search integration
- Unity Catalog-backed memory storage
- Multi-user isolation and durable session-backed state
- Streaming responses
- Background document indexing jobs
- Production OCR service integration

## License

Add the project license here and preserve the licenses and attribution notices
for Quivr, MemPalace, and all third-party UI components used by this project.
