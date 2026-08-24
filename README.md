# Turbo Fair Field Harness

A small local chat UI for an OpenAI-compatible model server. It streams generated text from `http://127.0.0.1:8080` by default.

## Structure

```text
app/
├── main.py           # Application factory and dependency wiring
├── config.py         # Environment-backed settings
├── schemas.py        # API request and response models
├── model_gateway.py  # Model discovery, requests, and stream relay
├── file_tools.py     # Workspace file discovery, reading, editing, and deletion
├── pdf_tools.py      # Bounded PDF text and metadata extraction
├── web_tools.py      # Public web search and readable page extraction
├── terminal_tools.py # Bounded shell command execution and output capture
├── state_tools.py    # Bounded persistent model-managed state
├── approvals.py      # In-memory user approval broker
├── tool_loop.py      # Model tool-call orchestration
├── routes.py         # HTTP route handlers
└── static/           # Browser UI
```

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m app
```

Open <http://127.0.0.1:8000>.

The app checks `/v1/models`, chooses the first returned model, and sends conversations to `/v1/chat/completions` with `stream: true`.

## Configuration

Copy `.env.example` if you want a reference, then export the variables you need before starting the app:

```bash
export MODEL_BASE_URL=http://127.0.0.1:8080
export MODEL_NAME=your-model-id
export APP_PORT=8001
python -m app
```

`MODEL_NAME` is optional when the server supports `/v1/models`. `MODEL_API_KEY` is also optional and is sent as a bearer token when present. The harness listens on port `8000` by default; set `APP_PORT` when that port is occupied.

## File tools

The model can list files, search text, read, create, overwrite, make exact text
replacements, and trash UTF-8 text files inside `model_workspace/`. Read-only
operations run automatically. Mutations pause for approval in the chat UI, and
deletes move files into `model_workspace/.trash/` for recovery. `create_file`
never overwrites an existing file; `write_file` explicitly supports both create
and overwrite behavior.

Set `MODEL_FILE_ROOT` to use another isolated directory.

## PDF tool

The model can use `read_pdf` to extract text and metadata from PDFs in the model
workspace. It can request specific 1-based pages, with at most 20 pages and
50,000 text characters returned per call. Without a page selection, the first
20 pages are read. PDF files are capped at 25 MB. Image-only or scanned pages
return a warning because this basic reader does not perform OCR. Encrypted PDFs
are not supported.

For a PDF on the web, the model can download it into the workspace with
`run_command` and then call `read_pdf`.

## Web tools

The model can use `search_web` to find public pages and `fetch_url` to extract
readable text and links from HTTP or HTTPS pages. Web responses are bounded by
timeouts, a 2 MB download limit, and a 20,000-character extracted-text limit.
Local and private-network URLs are rejected because these tools are intended for
information outside the machine.

## Terminal tool

The model can request `run_command` to execute a shell command starting from
`model_workspace/` (or a workspace-relative subdirectory). Commands run without
an approval prompt. Results include stdout, stderr, exit code, duration, and
timeout/truncation metadata. Commands time out after 30 seconds by default, with
a maximum requested timeout of 120 seconds, and each output stream is capped at
64 KB.

This is command execution, not an operating-system sandbox: a command can still
reference absolute paths or network resources available to the app process.

## Persistent state

The model can list, selectively read, write, replace, and delete durable state
entries across conversations. State defaults to
`model_workspace/.harness_state.json`; set `MODEL_STATE_FILE` to move it.

Stored values are never injected wholesale into prompts. The model receives a
short instruction to list available keys and read only relevant entries when
prior context may matter. State is limited to 128 entries, 8 KB per value, and
64 KB across all values. Listings return metadata rather than values, and writes
replace an entry instead of appending history. These constraints keep state and
prompt growth bounded.

## Test

```bash
pytest
```
