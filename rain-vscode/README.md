# Rain ⛈️ — IDE Integration

Connect any editor to Rain's local AI server. No cloud. No API keys. No tracking.

Rain runs a local HTTP server at `http://localhost:7734` that speaks both its own
protocol **and** a fully OpenAI-compatible `/v1/chat/completions` endpoint. This means
any tool that supports a custom OpenAI API base URL works out of the box.

---

## Quick Start

Make sure Rain's web server is running first:

```bash
cd ~/Rain
./rain-web
```

Verify it's up:

```bash
curl http://localhost:7734/api/health
# {"status":"ok","ollama":true,"model":"llama3.1"}
```

---

## ZED

ZED supports custom AI providers via its OpenAI-compatible API interface.

### Configuration

Open your ZED settings (`Cmd+,` on macOS → **Open Settings JSON**) and add:

```json
{
  "assistant": {
    "version": "2",
    "default_model": {
      "provider": "openai",
      "model": "rain"
    },
    "openai_api_url": "http://localhost:7734/v1"
  }
}
```

Set any string as your API key when ZED asks — Rain ignores it:

```json
{
  "openai_api_key": "local"
}
```

### Usage in ZED

- **Assistant panel** (`Cmd+?`): Full chat interface. Rain's multi-agent pipeline runs on every message.
- **Inline assist** (`Cmd+Enter` on selected code): Ask Rain about highlighted code.
- **Terminal**: ZED's built-in terminal — run `python3 rain.py --interactive` directly.

### ZED + Project Awareness

To give Rain full context of your codebase, index your project once:

```bash
python3 indexer.py --index /path/to/your/project
```

Then in ZED's assistant, Rain will automatically retrieve relevant file context
for every query.

---

## VSCode Extension (this extension)

### Install

**Option A — Load from source (no marketplace needed):**

```bash
cd Rain/rain-vscode
npm install          # installs @vscode/vsce (dev dep only)
npx @vscode/vsce package
# generates rain-ai-0.1.0.vsix
```

In VSCode: `Cmd+Shift+P` → **Extensions: Install from VSIX...** → select the `.vsix` file.

**Option B — Developer mode (live reload):**

1. Open `Rain/rain-vscode/` in VSCode
2. Press `F5` to launch the Extension Development Host
3. The Rain extension is active in the new window

### Commands

All commands are available via `Cmd+Shift+P` (or `Ctrl+Shift+P`):

| Command | Shortcut | Description |
|---|---|---|
| **Rain: Open Chat Panel** | `Cmd+Shift+Alt+R` | Open the Rain chat panel beside your editor |
| **Rain: Ask About Selection** | `Cmd+Shift+R` | Ask a question about selected code |
| **Rain: Explain This Code** | — | Plain-English explanation of selection |
| **Rain: Refactor Selection** | — | Refactor selected code (with instructions) |
| **Rain: Find Bugs in Selection** | — | Audit selection for bugs and security issues |
| **Rain: Write Tests for Selection** | — | Generate unit tests for selection |
| **Rain: Ask About This File** | — | Send the entire file to Rain with a question |
| **Rain: Index This Project** | — | Semantically index the workspace for code-aware Q&A |
| **Rain: Set Server URL** | — | Change the Rain server URL (default: `http://localhost:7734`) |

### Right-click menu

Select any code → right-click → **Rain: Ask About Selection / Explain / Refactor / Find Bugs**

### Extension Settings

| Setting | Default | Description |
|---|---|---|
| `rain.serverUrl` | `http://localhost:7734` | URL of the Rain server |
| `rain.webSearch` | `false` | Auto-augment queries with live DuckDuckGo results |
| `rain.sandbox` | `false` | Enable Rain's code execution sandbox |
| `rain.autoIndexOnOpen` | `false` | Auto-index workspace when project opens |

### Status Bar

The Rain status bar item (bottom-right) shows:
- **☁️ Rain ⛈️** — server is running, click to open chat
- **☁️ Rain offline** — server not reachable, run `./rain-web`

---

## Continue.dev (VSCode + JetBrains)

[Continue](https://continue.dev) is a popular open-source AI coding assistant. It supports
custom OpenAI-compatible providers natively.

### VSCode

Install Continue from the VSCode Marketplace, then edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Rain (local)",
      "provider": "openai",
      "model": "rain",
      "apiBase": "http://localhost:7734/v1",
      "apiKey": "local"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Rain Autocomplete",
    "provider": "openai",
    "model": "rain",
    "apiBase": "http://localhost:7734/v1",
    "apiKey": "local"
  }
}
```

### JetBrains (IntelliJ, PyCharm, etc.)

Same config — Continue's JetBrains plugin reads the same `~/.continue/config.json`.

### Usage

- `Cmd+I` — inline code edit with Rain
- `Cmd+Shift+I` — open Continue chat sidebar
- Highlight code → `Cmd+L` — add to context and ask Rain

---

## Aider

[Aider](https://aider.chat) is a terminal-based AI pair programmer. It supports custom
OpenAI endpoints directly:

```bash
pip install aider-chat

# Point Aider at Rain
aider \
  --openai-api-base http://localhost:7734/v1 \
  --model rain \
  --openai-api-key local
```

Or set environment variables to avoid repeating flags:

```bash
export OPENAI_API_BASE=http://localhost:7734/v1
export OPENAI_API_KEY=local
aider --model rain
```

Add to your shell profile to make it permanent:

```bash
echo 'export OPENAI_API_BASE=http://localhost:7734/v1' >> ~/.zshrc
echo 'export OPENAI_API_KEY=local'                    >> ~/.zshrc
echo 'export AIDER_MODEL=rain'                        >> ~/.zshrc
```

---

## Cursor

Cursor supports custom models via its OpenAI-compatible override:

1. Open Cursor Settings → **Models**
2. Add a custom model:
   - **Model name:** `rain`
   - **API base:** `http://localhost:7734/v1`
   - **API key:** `local` (any string)

---

## OpenAI Python SDK / any OpenAI client

Any code that uses the OpenAI Python SDK can talk to Rain with two lines:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:7734/v1",
    api_key="local",   # ignored — Rain is free and local
)

response = client.chat.completions.create(
    model="rain",
    messages=[{"role": "user", "content": "Explain Lightning Network channels"}],
)
print(response.choices[0].message.content)
```

Streaming:

```python
stream = client.chat.completions.create(
    model="rain",
    messages=[{"role": "user", "content": "Write a Bitcoin price checker in Python"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

---

## Project-Aware Q&A (all editors)

Index your codebase once so Rain can answer questions about your actual code:

```bash
# Index a project
python3 indexer.py --index /path/to/your/project

# Verify it worked
python3 indexer.py --list

# Search manually (useful for debugging)
python3 indexer.py --search /path/to/project "how does auth work?"
```

After indexing, pass `project_path` in your API requests:

```python
# Rain's native API with project context
import requests

response = requests.post("http://localhost:7734/api/chat", json={
    "message": "How does the payment flow work in this codebase?",
    "project_path": "/path/to/your/project",
    "web_search": False,
})
```

The VSCode extension does this automatically — it passes the open workspace path
with every message when a project is indexed.

---

## Direct REST API

Rain's full API is available at `http://localhost:7734`. Useful for scripting,
custom integrations, or calling from any language.

### Chat (SSE streaming)

```bash
curl -N -X POST http://localhost:7734/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is a Lightning invoice?",
    "web_search": false,
    "sandbox": false
  }'
```

Events streamed back:
```
data: {"type":"routing","agent":"Domain Expert (Bitcoin/sovereignty topic detected)"}
data: {"type":"progress","message":"💭 Domain Expert thinking..."}
data: {"type":"progress","message":"🔍 Reflection complete (rating: GOOD)"}
data: {"type":"done","content":"A Lightning invoice is...","confidence":0.87,"duration":8.2}
```

### OpenAI-compatible (non-streaming)

```bash
curl -X POST http://localhost:7734/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "rain",
    "messages": [{"role": "user", "content": "Explain Taproot"}],
    "stream": false
  }'
```

### Index a project

```bash
curl -X POST http://localhost:7734/api/index-project \
  -H "Content-Type: application/json" \
  -d '{"project_path": "/path/to/project", "force": false}'
```

### List indexed projects

```bash
curl http://localhost:7734/api/indexed-projects
```

---

## Troubleshooting

**Rain server not reachable**
```bash
# Check if it's running
curl http://localhost:7734/api/health

# Start it
cd ~/Rain && ./rain-web

# Check the port isn't in use by something else
lsof -i :7734
```

**Ollama not running**
```bash
# Start Ollama
ollama serve

# Then start Rain
./rain-web
```

**Model not found**
```bash
# Pull the base model
ollama pull llama3.1

# Pull the embedding model (required for project indexing)
ollama pull nomic-embed-text

# Check what's installed
ollama list
```

**Indexing is slow**
Project indexing calls `nomic-embed-text` for every file chunk — it can take a few
minutes for large projects. Run it in a terminal so you can watch progress:

```bash
python3 indexer.py --index /path/to/project
```

On Apple Silicon, indexing typically runs at 20–50 files/minute.
On x86 with no GPU, expect 5–15 files/minute.
Index once; re-index only when files change.

**ZED not connecting**
Make sure `openai_api_url` ends with `/v1` (not `/v1/`) — ZED appends paths to it directly.

---

## Architecture

```
Your Editor (ZED / VSCode / Aider / Cursor)
        │
        │  HTTP  (OpenAI format at /v1/chat/completions
        │          OR Rain's native SSE at /api/chat)
        ▼
Rain Server  (server.py — FastAPI at localhost:7734)
        │
        ├── AgentRouter → Dev Agent / Logic Agent / Domain Expert / Search Agent
        ├── Reflection Agent (always on)
        ├── Synthesizer (fires when quality is low)
        ├── RainMemory (SQLite at ~/.rain/memory.db)
        │     ├── Working memory (last 20 messages)
        │     ├── Episodic summaries
        │     ├── Semantic vectors (nomic-embed-text)
        │     ├── Learned corrections (Tier 4)
        │     └── User profile + session facts (Tier 5)
        └── ProjectIndexer (project_index table in memory.db)
                │
                └── Retrieves relevant file chunks for every query
                    when project_path is provided
        │
        ▼
Ollama  (local model runtime — no cloud, no tracking)
        │
        ├── llama3.2 / llama3.1  (reasoning, logic, domain)
        ├── codestral             (Dev Agent — code generation)
        ├── llama3.2-vision       (multimodal — images)
        └── nomic-embed-text      (semantic search + project indexing)
```

Everything runs on your machine. Nothing leaves your network.

---

*"Be like rain — essential, unstoppable, and free."* ⛈️