# Quivr + MemPalace

This is the first project slice for a future pip package. It gives a Quivr
application a local, persistent memory layer backed by
[MemPalace](https://github.com/mempalace/mempalace).

The package deliberately keeps the model provider separate from memory. It can
therefore be connected to Quivr, OpenAI, Anthropic, Ollama, or any other
responder that accepts a question and a context string.

## Install locally

From this directory:

```bash
python -m pip install -e ".[dev,models,ui]"
```

MemPalace uses a local palace by default. You can select another palace with
`--palace` or `MEMPALACE_PALACE_PATH`.

The `models` extra adds LangChain integrations for Groq, NVIDIA AI Endpoints
(including compatible local NVIDIA NIM servers), and Hugging Face. These
provider versions intentionally use the LangChain 0.3 generation required by
the current Quivr Core package. The Hugging Face `full` extra also installs the
Transformers and sentence-transformers dependencies needed for local
pipelines and embeddings.

The `ui` extra adds Gradio for the browser-based RAG test interface.

## Run the Gradio RAG UI

From CMD:

```cmd
quivr-rag-ui
```

Then open `http://127.0.0.1:7860`. Choose Groq or NVIDIA NIM, enter a question,
and click **Ask**. The interface displays the retrieved MemPalace context and
stores the completed user/assistant turn. Click **Recall memories** to test only
the retrieval stage without making an LLM request.

If the `quivr-rag-ui` command is not recognized, run it as a Python module:

```cmd
python -m quivr_mempalace.gradio_app
```

To run the Quivr Core RAG test separately:

```cmd
cd /d C:\Users\GauravSarma\Downloads\quivr-main\quivr-main
python -m pip install -e core
python -m pytest core\tests\test_quivr_rag.py -v
```

The Core install is required because the RAG tests import Quivr Core's full
dependency set, including `rapidfuzz`.

## Configure Groq and NVIDIA NIM

The project includes a local `.env` file with empty credential placeholders and
an `.env.example` template. Edit the local file from CMD:

```cmd
if not exist .env copy .env.example .env
notepad .env
```

Set `GROQ_API_KEY` for Groq. For NVIDIA, set `NVIDIA_API_KEY` when using the
hosted NVIDIA API Catalog, or set `NVIDIA_BASE_URL` to your self-hosted NIM URL,
such as `http://localhost:8000/v1`. Keep `.env` private; it is excluded by
`.gitignore`.

## Try the memory CLI

```bash
quivr-memory remember \
  --wing quivr-demo \
  --room preferences \
  "I prefer concise answers with Python examples."

quivr-memory search "How should answers be written?" --wing quivr-demo
quivr-memory wake-up --wing quivr-demo
```

Memories are stored verbatim. The same wing, room, and content produce the
same ID, so retrying a write does not create a duplicate drawer.

## Use it from Python

```python
from quivr_mempalace import MemoryAwareAssistant, MempalaceMemoryStore

store = MempalaceMemoryStore(palace_path="~/.mempalace/palace", wing="my-app")

assistant = MemoryAwareAssistant(
    store=store,
    wing="my-app",
    room="conversation",
    responder=lambda question, context: my_model(question, context),
)

answer = assistant.ask("What style do I prefer?")
print(answer)
```

`responder` is intentionally a two-argument callable. A Quivr adapter can
be created with `QuivrBrainResponder(brain)`; it forwards `question` to
`Brain.ask(...)` and places the recalled `context` in the system prompt. The
memory package does not force a particular LLM or API key.

For LangChain models, use `LangChainResponder`. It works with any LangChain
model that exposes `invoke`, including `ChatGroq`, `ChatNVIDIA`,
`HuggingFacePipeline`, and `ChatHuggingFace`.

```python
from langchain_groq import ChatGroq
from quivr_mempalace import LangChainResponder

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
assistant = MemoryAwareAssistant(
    store=store,
    wing="my-app",
    responder=LangChainResponder(model),
)
```

For NVIDIA, use:

```python
from quivr_mempalace import create_nvidia_model

model = create_nvidia_model()
```

For Groq, use:

```python
from quivr_mempalace import create_groq_model

model = create_groq_model()
```

Both models can be passed to `LangChainResponder(model)`.

For Hugging Face, create a `HuggingFacePipeline` or `ChatHuggingFace` from
`langchain_huggingface` and pass it to `LangChainResponder`.

```python
from quivr_mempalace import QuivrBrainResponder

assistant = MemoryAwareAssistant(
    store=store,
    wing="my-app",
    responder=QuivrBrainResponder(brain),
)
```

## Next packaging step

Before publishing, choose your final distribution/import name, add project
metadata such as author and repository URLs, and add the LLM adapter you want
to support. The package is already structured as a standalone Hatch project so
that the later `python -m build` / `twine upload` flow is straightforward.
