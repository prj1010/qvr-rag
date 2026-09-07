# Combined Quivr + MemPalace RAG App

This app combines the Quivr Core fork with the MemPalace integration and a
React browser UI. FastAPI exposes the RAG and memory endpoints; Vite builds the
frontend. The visual system uses Motion animations plus Magic UI, Aceternity UI,
and Origin UI-inspired spotlight, glow, glass, and bento patterns.

The setup script uses `uv` and installs the local Quivr Core checkout. When the
standalone `quivr-mempalace` source package exists in the surrounding workspace,
it is installed locally; otherwise the package is installed from PyPI.

## Install as a pip package

After publishing the package, users can install and run it with:

```cmd
python -m pip install quivr-mempalace-rag-ui
quivr-mempalace-rag-ui
```

The package automatically installs its `quivr-mempalace` dependency. Both
packages must be published before the one-command install works for new users.

## Install from CMD

```cmd
cd /d qvr-rag\examples\mempalace_rag_ui
setup.cmd
notepad .env
build_frontend.cmd
```

## Start the app

```cmd
cd /d qvr-rag\examples\mempalace_rag_ui
run_app.cmd
```

Open `http://127.0.0.1:7862`.

For frontend hot reload, start the API in one CMD window and Vite in another:

```cmd
cd /d qvr-rag\examples\mempalace_rag_ui
run_app.cmd
```

```cmd
cd /d qvr-rag\examples\mempalace_rag_ui
run_frontend.cmd
```

Then open `http://127.0.0.1:5173`.

## Build and publish from CMD

```cmd
cd /d qvr-rag\examples\mempalace_rag_ui
build_all.cmd
publish_all.cmd
```

Set your PyPI token before publishing:

```cmd
set "UV_PUBLISH_TOKEN=pypi-your-new-token"
publish_package.cmd
```

1. Choose Groq or NVIDIA NIM.
2. Upload PDF, DOCX, CSV, TXT, or Markdown documents.
3. Click **Index documents**. MarkItDown parses supported files locally, so a
   Megaparse/NATS server or Tesseract installation is not required.
4. Click **Recall memories** to test MemPalace alone.
5. Click **Ask** to run Quivr document RAG plus MemPalace memory context.

The Microsoft Agent Governance Toolkit fork is installed by the app
`requirements.txt` at a pinned commit. The admin-only **Agent governance**
panel displays collection access, rate limits, content scanners, privacy-safe
audit events, and a dry-run policy evaluator. Set the `AGT_*` variables in
`.env` to change the policy; indexing fails closed if the governance package
cannot be loaded.

For temporary access before Microsoft SSO is configured, set
`ADMIN_FALLBACK_ENABLED=true` and a random `ADMIN_FALLBACK_TOKEN` of at least
16 characters. The **Admin observability** button then opens a token login page.
Disable the fallback after Entra SSO is ready.

The app uses NVIDIA's `nvidia/nemotron-3-embed-1b` embedding model for document
indexing. Hosted embeddings require
`NVIDIA_API_KEY`; a self-hosted NVIDIA embedding NIM can be selected with
`NVIDIA_EMBEDDING_BASE_URL`. Set `NVIDIA_EMBEDDING_BATCH_SIZE` to tune request
batching and `QUIVR_CHUNK_SIZE`/`QUIVR_CHUNK_OVERLAP` to tune chunking.

MarkItDown converts PDF, DOCX, CSV, TXT, and Markdown files to compact Markdown.
Text-based PDFs and PDFs with an existing OCR text layer work locally. Image-only
scanned PDFs use DocStrange cloud OCR when `DOCSTRANGE_API_KEY` is set. Add
that secret to `.env` locally or to Render. The app calls DocStrange only when
MarkItDown returns no text; text-based PDFs remain local. Without the key,
scanned PDFs fail with an actionable configuration message. This deployment
intentionally does not bundle Tesseract or RapidOCR.
Vectors are stored in a temporary SQLite database through `sqlite-vec`, avoiding
the extra in-process FAISS index. Use a persistent pgvector or managed vector
service for production durability.
Hugging Face support remains available in the standalone integration package
through its optional `huggingface` extra.
