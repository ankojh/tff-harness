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
├── agent_runs.py     # Durable system-owned run state and lifecycle controls
├── agent_loop.py     # Budgeted, stoppable, resumable agent orchestration
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
`model_workspace/` (or a workspace-relative subdirectory). Every command pauses
for approval in the chat UI. Results include stdout, stderr, exit code, duration,
execution mode, and timeout/truncation metadata. Commands time out after 30
seconds by default, with a maximum requested timeout of 120 seconds, and each
output stream is capped at 64 KB.

Terminal execution has three modes, selected with `MODEL_TERMINAL_MODE`:

- `sandbox` (default) runs each approved command in a fresh Docker container.
- `host` is an explicit compatibility mode that runs approved commands with the
  harness process's host permissions.
- `disabled` removes `run_command` from the tools offered to the model.

Build the default sandbox image before using terminal commands:

```bash
docker build -t tff-harness-sandbox:latest -f sandbox/Dockerfile .
```

Set `MODEL_SANDBOX_IMAGE` to use a different prebuilt local image. Sandbox runs
use `--pull never`, so tool execution never downloads an image implicitly.

### Sandbox boundary

The Docker backend mounts only the model workspace at `/workspace` and runs with
the invoking user's numeric UID and GID. The container has no network, a
read-only root filesystem, a bounded temporary filesystem, all Linux
capabilities dropped, `no-new-privileges`, and CPU, memory, process-count,
output, and wall-time limits. The host home directory, environment variables,
credentials, and Docker socket are not passed into the container.

The workspace itself is deliberately writable, so an approved command can
create, replace, or delete anything inside it. Docker and the selected image are
part of the trusted computing base. Do not mount the Docker socket or sensitive
host paths into a custom sandbox image. A bind-mounted workspace does not have a
separate storage quota, so keep the model workspace on a volume with adequate
host-level free-space monitoring. Public research remains available through the
separately bounded web tools rather than container networking.

## Agent mode

Select **Agent** in the toolbar and submit one explicit goal. Agent mode owns a
single supervised run through planning, workspace execution, verification, and
completion. Its plan, current step, usage, status, verification evidence, and
final summary are visible in the UI. File mutations and terminal commands keep
their per-operation approval prompts.

The harness, rather than the model, persists authoritative run state. By default
it is stored in `.tff_agent_runs/<workspace-name>.json`, beside rather than
inside the model workspace. Set `MODEL_AGENT_STATE_FILE` to choose another
location; the app rejects paths inside `MODEL_FILE_ROOT` so workspace tools
and sandboxed commands cannot rewrite their own status or budgets. Explicit
`host` terminal mode does not provide that isolation and remains trusted access.

Only one run can be active, and only the current run is retained in v1. A run
interrupted by a client disconnect or harness restart becomes `stopped` and must
be resumed manually. Runs waiting on the user, blocked runs, stopped runs, and
transiently failed runs are resumable. Completed and budget-exhausted runs are
not. Stop is available both in the run panel and the composer.

Completion is lifecycle-gated: the agent must create a plan, complete every
step, enter verification, record concrete evidence, and then call the completion
control. The harness enforces this sequence, although the quality of semantic
verification still depends on the model and the evidence-producing tools.

Agent execution has fixed cumulative budgets:

- `AGENT_MAX_TOOL_ROUNDS` defaults to 32 model/tool rounds.
- `AGENT_MAX_TOOL_CALLS` defaults to 64 calls, including lifecycle controls.
- `AGENT_MAX_SECONDS` defaults to 900 elapsed streaming seconds, including time
  spent waiting for an approval.
- `AGENT_MAX_CONSECUTIVE_FAILURES` defaults to 3 tool failures.

The UI restores the current run after a reload and exposes manual Resume when
the stored status allows it. Ordinary Chat mode remains available and does not
create durable agent-run state.

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
