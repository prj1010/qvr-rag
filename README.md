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
- Tesseract OCR for scanned PDFs

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

Text-based PDFs are extracted directly. Scanned PDFs use Tesseract OCR.

Install Tesseract on Windows with:

```cmd
cd /d src\ragb\examples\mempalace_rag_ui
install_ocr.cmd
```

If Tesseract is installed in a custom location, set `TESSERACT_CMD` in `.env`.

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

```cmd
cd /d src\ragb\examples\mempalace_rag_ui\frontend
npm install
npm run build
```

### OCR is unavailable

```cmd
cd /d src\ragb\examples\mempalace_rag_ui
install_ocr.cmd
```

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
- Authentication and multi-user isolation
- Streaming responses
- Background document indexing jobs
- Production OCR service integration

## License

Add the project license here and preserve the licenses and attribution notices
for Quivr, MemPalace, and all third-party UI components used by this project.
