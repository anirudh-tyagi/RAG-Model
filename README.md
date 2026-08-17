# Document RAG

Ask questions about your PDFs and get answers that cite the page they came from.

Hybrid retrieval (dense + BM25, fused server-side), cross-encoder reranking, figure
captioning with a vision model, streamed answers with clickable citations, voice input,
and an evaluation harness so retrieval changes can be measured instead of guessed at.

```
┌─────────────┐         ┌──────────────┐        ┌──────────────┐
│  Next.js    │  SSE    │   FastAPI    │  jobs  │  arq worker  │
│  web app    │◄───────►│     API      │───────►│  (ingestion) │
└─────────────┘         └──────┬───────┘        └───────┬──────┘
                               │                        │
                    ┌──────────┴────────┐               │
                    ▼                   ▼               ▼
              ┌──────────┐        ┌──────────┐   ┌─────────────┐
              │  Redis   │        │  Qdrant  │   │ Ollama Cloud│
              │ state +  │        │ dense +  │   │  LLM + VLM  │
              │  queue   │        │  sparse  │   └─────────────┘
              └──────────┘        └──────────┘
```

---

## How it works

**Ingestion** (background worker, progress streamed to the browser at every stage):

```
PDF → parse to per-page markdown → caption figures with a VLM
    → heading-aware chunking → dense + sparse embeddings → Qdrant
```

**Answering**:

```
question → condense follow-ups against history
         → dense + BM25 search, fused with RRF   (40 candidates)
         → cross-encoder rerank                  (top 6 survive)
         → stream the answer with [n] citations
```

Every chunk carries its document, page number, section heading, and whether it came from a
figure — which is what lets a citation resolve to a specific page rather than a vague
"somewhere in this PDF."

### Retrieval, in more detail

Dense-only similarity search misses exact terms: part numbers, unusual proper nouns,
acronyms. BM25 alone misses paraphrase. Qdrant stores **both** vectors on every point and
fuses the two result lists with reciprocal rank fusion in a single round trip, so a passage
that only one retriever found still reaches the candidate set.

Fusion decides what's *in* the shortlist; the **cross-encoder** decides the order. It scores
each `(question, passage)` pair jointly rather than comparing two independently-computed
vectors — slower, so it only runs over the ~40 fused candidates, then the best 6 go to the
model.

Sparse vectors carry Qdrant's `IDF` modifier, so term weights are computed against the whole
corpus rather than guessed per batch at index time.

### Figures

Charts and tables are where the numbers live, and standard text extraction throws them away.
Each extracted image goes to a vision model and becomes its own retrievable chunk, so "what
accuracy does Figure 3 report?" has something to match against. Captioning runs concurrently
under a semaphore, is capped per document, and one failed figure costs one caption rather
than the whole ingest.

> **Off by default.** It needs a multimodal model, and Groq — the default provider — does not
> host one. Point `VISION_BASE_URL` / `VISION_API_KEY` at a provider that does (Ollama Cloud's
> `qwen3-vl`, or OpenAI) and set `CAPTION_IMAGES=true`.

---

## Quickstart

### Docker (everything)

```bash
cp .env.example .env       # then set LLM_API_KEY
docker compose up --build
```

Then open **http://localhost:3000**.

The LLM is reached over the **OpenAI-compatible** API, so the provider is a base-URL and
model-name change rather than a code change:

| Provider | `LLM_BASE_URL` | Key from |
|---|---|---|
| **Groq** (default) | `https://api.groq.com/openai/v1` | <https://console.groq.com/keys> |
| Ollama Cloud | `https://ollama.com/v1` | <https://ollama.com/settings/keys> |
| OpenAI | `https://api.openai.com/v1` | <https://platform.openai.com/api-keys> |

> First start downloads ~1.9GB of models (BGE embeddings, the reranker, Whisper). They
> persist in the `models` volume, so subsequent starts are fast.

### Local development

Needs **Python 3.12** (via `uv`) and **Node 20+**. Backing services still come from Docker:

```bash
docker compose up qdrant redis     # just the infrastructure
make setup                         # install both apps, create .env

make dev-api                       # http://localhost:8000  (docs at /docs)
make dev-worker                    # ingestion worker
make dev-web                       # http://localhost:3000
```

`make help` lists every target.

---

## Configuration

All settings live in `.env` — see [`.env.example`](.env.example) for the annotated list.
Only `LLM_API_KEY` is strictly required.

The ones worth knowing about:

| Variable | Default | Why you'd change it |
|---|---|---|
| `LLM_BASE_URL` | Groq | Any OpenAI-compatible endpoint (see table above) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Any chat model your provider hosts |
| `VISION_MODEL` / `VISION_BASE_URL` / `VISION_API_KEY` | llama-4-scout | Figure captioning. Set separately because your chat provider may not host a multimodal model — **Groq currently does not.** Leave URL and key blank to reuse the chat provider |
| `DENSE_MODEL` | `BAAI/bge-large-en-v1.5` | `bge-base-en-v1.5` is ~3× faster on CPU. **Changing this needs a re-index** — the dimensions differ |
| `RERANK_ENABLED` | `true` | Turn off to see what reranking is buying you |
| `RETRIEVAL_CANDIDATES` | `40` | Shortlist size handed to the reranker |
| `RETRIEVAL_TOP_K` | `6` | Passages given to the model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `150` | Characters |
| `PDF_PARSER` | `pymupdf` | `docling` handles complex layouts and tables far better, but pulls in torch (`uv sync --extra docling`) |
| `CAPTION_IMAGES` | `false` | Turn on once `VISION_*` points at a multimodal model. One vision call per figure |
| `WHISPER_TASK` | `translate` | Whisper turns any supported language straight into English |
| `LANGFUSE_*` | unset | Set to trace retrieval and generation. Tracing is a no-op without keys |

---

## Evaluation

Retrieval tuning without measurement is guesswork, so there's a harness:

```bash
# Fast, free, deterministic — no LLM calls. The right loop for tuning chunk size,
# the embedding model, candidate count, or whether reranking earns its latency.
make eval-retrieval

# Full run: also generates answers and grades them.
make eval
```

Build a golden set from [`evals/golden.example.jsonl`](evals/golden.example.jsonl).
Relevance is labelled at **page** level, not chunk level — chunk ids change every time you
touch chunking, which would invalidate the whole dataset on exactly the changes you most
want to measure.

| Metric | What it catches |
|---|---|
| recall@k, MRR, nDCG@k | Did the right page surface, and how high? |
| keyword coverage | Did the answer contain the exact figures it should? |
| **citation validity** | A `[7]` when only six passages were supplied is a fabricated reference — caught with no LLM judge needed |
| abstention rate | Tracked separately: declining to answer is *correct* when retrieval misses |
| groundedness | Optional LLM-as-judge (`--judge`) |

`--fail-under-recall 0.8` exits non-zero, so a regression can gate CI.

---

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/documents` | Upload a PDF → `202` + document id |
| `GET` | `/api/documents` | List with ingestion state |
| `GET` | `/api/documents/{id}/events` | **SSE** live ingestion progress |
| `DELETE` | `/api/documents/{id}` | Remove vectors, files and registry entry |
| `POST` | `/api/chat` | **SSE** `meta` → `sources` → `token`… → `done` |
| `GET` | `/api/conversations` | History |
| `POST` | `/api/search` | Raw retrieval, no LLM — for tuning and debugging |
| `POST` | `/api/transcribe` | Audio → English text |
| `GET` | `/api/health` | Per-dependency status |

Interactive docs at `/docs`.

---

## Layout

```
apps/
├── api/                        FastAPI, async, strict mypy
│   ├── src/rag/
│   │   ├── ingest/             parse → caption → chunk → pipeline
│   │   ├── retrieval/          embeddings · store · rerank · searcher
│   │   ├── chat/               llm · memory · prompts · service
│   │   ├── jobs/               queue · worker
│   │   ├── eval/               dataset · metrics · runner · cli
│   │   └── api/routes/         documents · chat · search · audio · health
│   └── tests/                  132 tests, fully offline
└── web/                        Next.js 15 · React 19 · TypeScript · Tailwind v4
    ├── app/
    ├── components/
    └── lib/                    api client · SSE reader · hooks
```

---

## Development

```bash
make check      # lint + typecheck + tests, i.e. everything CI runs
make test
make fix        # auto-fix lint and formatting in both apps
```

The API test suite runs **fully offline** — fakeredis, an in-memory vector store, a scripted
LLM, and no model downloads. It needs no services and no API key, which is why CI can run it
on every push in about a second.

CI additionally builds both Docker images.

---

## What changed from the original version

The first version worked, but the design had some load-bearing problems. For the record:

| | Before | Now |
|---|---|---|
| **Dependencies** | No manifest at all — not reproducible | `uv` + committed lockfile |
| **Pipeline** | Three scripts shelled out via `subprocess.run`, passing data by filename convention | One in-process async pipeline with shared models |
| **Output paths** | `Base.py` wrote a directory relative to the process CWD; the next script guessed the same path | Explicit paths throughout |
| **Job durability** | `BackgroundTasks` — a restart mid-parse lost the job silently | arq on Redis; jobs survive an API restart |
| **Progress** | `capture_output=True` sent it to a log nobody watched; the UI waited 1.5s and redirected | Every stage published to Redis and streamed to the browser |
| **Startup failure** | `exit()` — a missing key killed the process before it could say why | Degraded status per dependency on `/api/health` |
| **Upload safety** | `PDF_FOLDER / file.filename`, unsanitised | Stored as `<uuid>.pdf`; magic-byte and size checks |
| **Collections** | One Chroma collection per PDF, named by a sanitiser that could collide | One Qdrant collection filtered by `doc_id` — questions can span documents |
| **Retrieval** | Dense top-5 | 40 candidates from dense + BM25 fused by RRF, cross-encoder reranked to 6 |
| **Chunking** | 512 chars, markdown flattened, headings discarded | 1200 chars, heading-aware, page-attributed |
| **Answers** | One blocking JSON response | Streamed tokens |
| **Citations** | None | Numbered, clickable, resolving to document + page + section |
| **Memory** | A JS array the backend never saw | Server-side history + follow-up condensing |
| **Captioning** | Strictly serial inside a `re.sub` callback | Concurrent, capped, failure-isolated |
| **Speech** | `speech_recognition` (undocumented Google endpoint) + `googletrans` + `pydub`/ffmpeg | faster-whisper — transcribes and translates in one step, no ffmpeg |
| **Text to speech** | Server-side gTTS writing wav files | Browser `SpeechSynthesis` |
| **Embeddings** | sentence-transformers, ~2.5GB of torch to embed text on CPU | ONNX via fastembed — no torch in the default install |
| **Frontend** | Vanilla JS, Tailwind from a CDN | Next.js 15, React 19, TypeScript, Tailwind v4 |
| **LLM provider** | `ChatOllama`, hardcoded to Ollama Cloud | OpenAI-compatible client — Groq, Ollama Cloud or OpenAI by config |
| **Tests** | None | 132, offline, in CI |

---

## Status

**Statically verified:** ruff, strict mypy and **132 passing tests** on the API; Biome and
`tsc --noEmit` on the web app.

**Verified against live services:**

- LLM client against Groq — health, non-streaming, streaming, abstention, error mapping
- `pymupdf4llm` parsing a real PDF, with correct per-page attribution
- heading-aware chunking, producing the right headings and page numbers
- fastembed dense + BM25 sparse encoding
- Qdrant collection creation and hybrid `Prefetch` + RRF `FusionQuery` (local mode)
- cross-encoder reranking, ordering correctly
- the full loop — retrieve → rerank → stream a cited answer — including follow-up
  condensing and conversation persistence

Two real bugs were caught this way and fixed: `gpt-oss` emits full-width `【1】` rather than
`[1]`, which would have left every citation unclickable and scored citation validity at zero;
and the streaming response was not closed when a client disconnected mid-answer.

**Still unverified:** the Docker images have never been built, and the stack has not run
under `docker compose` with Qdrant and Redis as real services. `next build` has not run
either — it needs Node 20+.

**Known limitations:**

- Figure captioning is off by default: Groq hosts no multimodal model (see Configuration).
- Ingestion is CPU-bound and slow on a laptop for large PDFs; `pymupdf` is the default parser
  for that reason.
- Scanned PDFs need OCR — `pymupdf` will find no text. Try `PDF_PARSER=docling`.
- When on, captioning costs one vision call per figure, capped at `MAX_CAPTIONS_PER_DOC=60`.
- Conversations are kept in Redis with a 30-day TTL; they are a convenience, not a system of
  record.
- There is no authentication. This is a single-user local application as it stands.
