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
├── memory_tools.py   # Provenance-aware lessons, run memory, and repository index
├── context_compaction.py # Automatic bounded agent-context compaction
├── agent_runs.py     # Durable system-owned run state and lifecycle controls
├── agent_loop.py     # Budgeted, stoppable, resumable agent orchestration
├── agent_workers.py  # Scoped specialist and isolated implementation workers
├── worker_workspaces.py # Private snapshots, change sets, integration, rollback
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

Agent memory defaults to `.tff_memory/<workspace-name>.json` beside the model
workspace. Set `MODEL_MEMORY_FILE` to move it; like agent run state, the memory
file must remain outside `MODEL_FILE_ROOT` so workspace tools cannot rewrite it.

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
completion. Its plan, current step, usage, status, verification evidence,
execution receipts, event count, and final summary are visible in the UI.

File mutations and terminal commands keep their per-operation approval prompts
by default. Before starting a new run, the UI can optionally grant either
workspace mutations or terminal commands for that run only. Grants are opt-in,
recorded in the run journal, and cannot be added after the run starts. They are
bound to that run ID and never carry into a different run. A terminal
grant in `host` mode is trusted host execution, so prefer the sandbox backend.

The harness, rather than the model, persists authoritative run state. By default
it is stored in `.tff_agent_runs/<workspace-name>.json`, beside rather than
inside the model workspace. Set `MODEL_AGENT_STATE_FILE` to choose another
location; the app rejects paths inside `MODEL_FILE_ROOT` so workspace tools
and sandboxed commands cannot rewrite their own status or budgets. Explicit
`host` terminal mode does not provide that isolation and remains trusted access.

Only one run can be active. v1.1 retains the 50 most recent runs, with a bounded
append-only event journal for each run. The **History** and **Activity** controls
show recent run and event records; the same data is available through
`GET /api/agent/runs` and `GET /api/agent/runs/{id}/events`.

A run interrupted by a client disconnect or harness restart becomes `stopped`
and must be resumed manually. Runs waiting on the user, blocked runs, stopped
runs, and transiently failed runs are resumable. Completed and budget-exhausted
runs are not. Stop is available both in the run panel and the composer.

While a run is streaming, type guidance and use **Steer** to queue it for the
next model boundary. Steering is persisted before the API acknowledges it, so
unapplied guidance survives a harness restart and is consumed after resume.
Replacing an existing plan requires a reason, and both the prior and revised
plans are retained in the run record.

Completion is lifecycle-gated: the agent must create a plan, complete every
step, enter verification, record concrete evidence, and then call the completion
control. Every external tool execution gets a durable receipt. Verification
evidence must link to a successful receipt, and repeated side-effecting calls use
their prior completed receipt instead of executing again. An interrupted
side-effecting call with an uncertain outcome pauses for user input rather than
being retried automatically. The harness enforces these mechanics, although the
quality of semantic verification still depends on the model and the
evidence-producing tools.

`/api/status` reports whether Docker and the configured sandbox image are ready.
The UI marks an unavailable sandbox in red and disables the terminal grant until
the readiness check succeeds.

### Agent v2: specialist delegation

Agent v2 adds bounded multi-agent delegation without giving multiple actors
competing write access. The root agent may call `agent_delegate_tasks` with one
to three independent tasks. The harness runs those specialists concurrently and
returns their reports to the root agent as one receipt-backed tool result.

Specialists have explicit roles:

- `researcher` gathers implementation context and evidence.
- `reviewer` examines correctness, risks, and edge cases.
- `tester` inspects test coverage and recommends verification work.

Each task is confined to an existing workspace-relative directory. Inside that
scope, workers can list, search, and read files, read PDFs, use the bounded public
web tools, and read persistent-state entries. They cannot write or delete files,
run terminal commands, mutate persistent state, approve actions, delegate more
workers, change the root plan, record verification, or complete the run. Root
run capability grants are never inherited by workers.

The root agent remains accountable for evaluating worker reports, resolving
conflicts, making all changes, running verification, and deciding whether the
goal is complete. This keeps the single-writer safety boundary from v1.1 while
allowing parallel analysis.

Worker execution is bounded independently from the root run:

- At most 3 workers run in one delegation batch and 32 are retained per run.
- Each worker gets at most 6 model rounds, 12 read-only tool calls, and 120
  seconds.
- Three consecutive tool failures stop that worker.
- Reports are capped at 16,000 characters.

Worker status, role, scope, usage, result, and errors are durable run state and
appear in the **Specialists** section of the Agent panel. Worker lifecycle and
tool-use events are included in **Activity** and
`GET /api/agent/runs/{id}/events`; full task records are available from
`GET /api/agent/runs/{id}/tasks`. If a stream is cancelled or the harness
restarts, running workers become `stopped` instead of being silently retried.

### Agent v3: isolated implementation and integration

Agent v3 adds an `implementer` role without abandoning the root agent's
single-writer authority. An implementer receives `isolated_write` mode and a
private snapshot of its assigned workspace scope. File mutations happen only in
that snapshot. When the terminal backend is sandboxed, the implementer may also
run commands inside a container mounted to the private snapshot. Host terminal
mode is never exposed to workers, and root capability grants are not inherited.

Private snapshots are bounded to 2,000 regular files and 64 MB. Repository and
trash internals are excluded, symbolic links and special files are rejected,
and at most four implementation workers may be retained in a run. A completed
worker produces a hash-addressed change set of at most 500 created, updated, or
deleted files. Its report and changes appear in the **Specialists** panel, but
the root workspace remains unchanged.

The root must explicitly resolve every pending change set:

- `agent_integrate_worker` compares every affected root file with the worker's
  original snapshot before writing anything. Divergence produces a conflict
  instead of an overwrite. Integration requires the normal user approval or an
  explicit run-scoped workspace-mutation grant.
- Successful integration captures the original affected files in a checkpoint
  and applies the staged files with same-directory atomic replacements.
- `agent_rollback_worker` restores that checkpoint after another approval. It
  refuses if an integrated file changed afterward.
- `agent_discard_worker` records rejection and deletes unintegrated private
  artifacts without touching the root workspace.

The harness will not accept root completion while a completed implementer still
has pending changes. Integration, conflicts, discards, and rollback are journaled
and receipt-backed. Failed and cancelled workers discard private snapshots;
artifacts belonging to history records are removed when those records age out of
the retained 50-run history.

### Agent v4: dependency-aware orchestration

Agent v4 adds a durable task graph above the isolated v3 worker boundary. The
root agent can create up to 32 keyed nodes with explicit dependencies and a
concurrency limit of one to three. Ready nodes run automatically in parallel
waves. Reports from completed dependencies are included in downstream worker
instructions, while integrated files are available through the normal root
workspace.

Implementation nodes are scheduling barriers: successors do not start merely
because an implementer finished its private work. The root must inspect and
integrate the change set first. A successful approved integration automatically
unlocks eligible successors. Independent implementers must use non-overlapping
directory scopes; overlapping scopes are allowed only when a dependency orders
the workers sequentially.

The root can adapt an active graph with `agent_replan_task_graph`. Replans use an
expected revision number, require a reason, and may cancel unresolved nodes or
add new nodes without rewriting completed history. Integration conflicts have a
dedicated `agent_resolve_worker_conflict` workflow: retry creates a fresh private
snapshot from the current root, while discard rejects the branch and blocks its
dependents. No conflict-resolution action silently overwrites root files.

Every graph shares cumulative limits of 36 worker model rounds, 72 worker tool
calls, and 360 elapsed seconds. These are in addition to per-worker bounds and
the root run budget. Exhaustion prevents new waves from starting. The graph,
node states, revision history, usage, dependency links, and worker links are
durable and visible in the Agent panel. They are also returned with the run and
from `GET /api/agent/runs/{id}/graph`.

Final verification is root-owned and runs against the combined integrated
workspace. The harness rejects `agent_begin_verification` and run completion
until the graph is complete. Ad-hoc delegation is disabled while a graph is
active so it cannot bypass dependency ordering, scope checks, or graph budgets.

### Agent v4.1: human change review

Agent v4.1 makes isolated worker output reviewable before root integration.
`agent_review_worker` and `GET /api/agent/runs/{id}/tasks/{task_id}/review`
return bounded unified diffs, per-file conflict state, and stable hunk IDs. Text
review is limited to 1 MB per file and 64,000 diff characters per worker.
Binary and oversized files are identified explicitly and require whole-file
acceptance.

`agent_integrate_worker` remains approval-gated and now accepts whole-file paths,
individual hunk IDs, or both. Omitting both selection fields preserves the prior
accept-all behavior. The approval card shows the diff and lets the user refine
the model's proposed selection with file and hunk checkboxes before allowing the
operation. Empty selections are rejected; use discard when none of the changes
should be retained.

For partial text acceptance, the harness reconstructs a staged file from the
unchanged baseline plus only the selected diff hunks. It rechecks the root hash
and every staged hash before writing. The durable review decision records whole
files, hunks, and completely rejected files. The worker's integrated change set
is replaced with the exact applied result, so rollback and restart recovery act
only on accepted content; rejected files and hunks never touch the root.

### Agent v5: context and memory engineering

Agent v5 adds a bounded, local, provenance-aware memory layer alongside the
existing explicit key/value state. The root and read-only workers can use
`search_memory` and `read_memory` across four kinds of evidence:

- `repository` records are line-addressed chunks from a lexical workspace index.
  The index is refreshed when a run starts and before repository searches. It is
  bounded to 2,000 UTF-8 files, 4,000 chunks, 4 MB of text, and 512 KB per file.
  Repository internals, dependency trees, binary files, likely credential files,
  private keys, and likely literal secrets are excluded.
- `run` records capture the goal, final summary, plan, and receipt-linked
  verification evidence of completed runs.
- `context` records retain a safe activity summary whenever old model context is
  automatically compacted. Tool-call/result groups remain structurally intact in
  the active context, and recent work is retained verbatim within a fixed bound.
- `lesson` records are concise cross-run inferences. `save_lesson` requires one
  or more successful receipt IDs from the current run, a confidence from 0.5 to
  1.0, and an expiry from 1 to 365 days. Optional source paths are hash-linked;
  changing or deleting a source makes the lesson stale automatically.

Search results expose trust, confidence, expiry, stale reason, source hashes,
run IDs, and receipt provenance. Expired and stale records are excluded by
default. Every prompt and tool result labels retrieved content as untrusted
evidence rather than instructions, reducing the chance that indexed files or a
poisoned prior lesson can override the active goal. Only agent-authored lessons
can be deleted through `forget_lesson`; harness-owned run and context records are
immutable through model tools.

The memory status and repository index can be inspected with
`GET /api/memory/status`, searched with `GET /api/memory/search?q=...`, and
refreshed with `POST /api/memory/index/refresh`. The toolbar memory indicator
shows active record and index counts; clicking it performs a manual refresh.

### Agent v6: richer graph execution

Agent v6 extends the durable scheduler with typed node behavior while preserving
the existing DAG, sandbox, integration, approval, and root-verification
boundaries. Ordinary nodes remain backward compatible.

- Conditional nodes evaluate a bounded declarative rule against one dependency:
  dependency status, report containment, artifact truthiness, or artifact
  equality. False branches become `skipped`; conditions cannot execute code or
  treat artifact text as instructions.
- `map` nodes discover work by returning a configured JSON-array artifact. The
  scheduler expands at most six read-only child nodes from a template and can
  add an all-success reduce join. `{item}` values are JSON-encoded and explicitly
  labeled untrusted; scope substitutions are separately slugged and revalidated.
- `join` nodes support `all`, `quorum`, and `first_success`. A reached quorum can
  cancel remaining queued dependencies. Already-running workers finish their
  bounded execution, so short-circuiting never abandons an uncertain mutation.
- Nodes may select a model and set model-round, tool-call, and wall-time limits.
  These remain subordinate to the graph-wide and root-run budgets.
- Read-only `loop` nodes repeat at most five times until a declarative report or
  artifact condition matches. Every iteration is a separate durable worker
  record; exhaustion fails the node instead of creating a graph cycle.
- `agent_control_graph_node` can pause/resume queued work or rerun a terminal
  node. `agent_replace_graph_node` swaps one definition. Prior attempts are
  retained, and completed descendants are invalidated by default so stale
  downstream results cannot be reused.
- Three built-in templates cover research/implementation/review, parallel
  analysis/synthesis, and test/fix/verify. They are available through
  `agent_list_graph_templates`, `agent_create_graph_from_template`, and
  `GET /api/agent/graph/templates`.
- Workers may return up to eight durable `text`, `json`, or `file_manifest`
  artifacts, with a 16 KB per-artifact and 64 KB per-worker total. Artifacts have
  stable IDs and content hashes. Graph results expose metadata; content is read
  explicitly with `agent_read_graph_artifact` or the graph artifact API and is
  labeled untrusted data.

For stopped or otherwise resumable runs, the UI exposes pause, resume, and rerun
controls. Equivalent endpoints live under
`/api/agent/runs/{run_id}/graph/nodes/{key}`; direct controls are rejected while
an agent execution is active to avoid concurrent scheduler ownership. A paused
graph cannot pass final verification or completion.

The richer graph still retains at most 32 worker executions, four isolated
implementers, three concurrent workers, 36 graph worker model rounds, 72 graph
worker tool calls, and 360 graph seconds.

### Agent v7: operator experience

Agent v7 adds an operator workspace around the durable v6 execution model:

- Task graphs render as a layered DAG with SVG dependency edges. Selecting a
  node opens its instruction, policy, status, attempts, artifacts, and safe
  controls. Resumable runs can pause, resume, rerun, or replace a node from the
  editor; replacement still uses revision history and downstream invalidation.
- The DAG can switch to a durable worker/graph event timeline. The Activity
  view shows the complete run timeline with event data and timestamps.
- Built-in graph templates are explicitly versioned and publish bounded
  parameter definitions. Template creation accepts an exact version and a
  validated `focus` value, and the chosen version and rendered parameters are
  persisted with the graph.
- Artifact chips open an opt-in viewer with type, byte count, content hash,
  formatted content, copy, and download actions. Artifact content remains
  excluded from ordinary run and graph responses.
- Run History can reopen earlier runs and choose two for structural comparison.
  Compare reports lifecycle, usage, plan, worker, graph, and per-node changes.
- Clone creates a fresh stopped run with the source goal, model settings,
  budgets, and grants while deliberately excluding prior messages, tasks,
  approvals, receipts, and graph results. The lineage ID is durable and the
  clone is ready for an explicit resume.
- Debug returns a bounded public snapshot, budget utilization, unresolved
  receipts and integrations, failed or paused graph nodes, and the latest 100
  events. It does not expose model messages, receipt results, or artifact
  content.

Operator endpoints include `GET /api/agent/runs/{id}`,
`GET /api/agent/runs/compare`, `POST /api/agent/runs/{id}/clone`, and
`GET /api/agent/runs/{id}/debug` in addition to the v6 graph-control APIs.

### Agent v8: complete memory integration

Agent v8 connects durable graph output to cross-run retrieval and operator
feedback:

- Completing a run indexes up to 64 unique worker artifacts as `artifact`
  memories. Each record retains run, graph, node, task, artifact type, and
  content-hash provenance. Exact duplicate artifacts collapse into one memory
  with bounded occurrence history. Unfinished-run artifacts are not promoted to
  cross-run memory.
- Search uses a dependency-free hybrid ranker: the existing exact lexical score
  is combined with a deterministic local semantic feature vector using
  normalized terms, concepts, and fuzzy character features. Results expose both
  score components and do not require a separate embedding service.
- Saving a highly similar lesson automatically consolidates it into the older
  record. Confidence, expiry, receipt/source provenance, run lineage, and
  feedback are merged. `consolidate_lessons` and
  `POST /api/memory/lessons/consolidate` also audit existing active lessons.
- Operators can mark lessons helpful or unhelpful with an optional reason.
  Feedback changes retrieval rank; three strongly negative ratings quarantine a
  lesson as stale. User-rejected lessons are excluded from normal search but
  remain available for explicit historical inspection.
- Clicking the Memory indicator opens a browser for artifact, lesson, run,
  compacted-context, and repository search. It shows hybrid scores, provenance,
  freshness, consolidation counts, full opt-in reads, feedback controls, and
  repository refresh.

Memory files migrate from version 1 to 2 on read. New endpoints include
`GET /api/memory/entries/{id}` and
`POST /api/memory/lessons/{id}/feedback`. All retrieved content remains labeled
untrusted evidence rather than instructions.

### Agent v9: observability and evals

Agent v9 makes execution measurable and regression-testable across the root
agent and graph workers:

- Every run receives a durable trace ID and root span. Correlated model, tool,
  worker-model, and worker-tool spans retain status, duration, safe attributes,
  usage, cost, and classified errors. `GET /api/agent/runs/{id}/trace` combines
  those spans with the complete event journal as a relative run timeline.
- Streaming provider usage is requested and recorded when supported. Otherwise
  the harness records a deterministic estimate and labels it `estimated`.
  Aggregate input/output/total tokens, model and tool latency, wall latency,
  and model-call counts are exposed on every run.
- Cost accounting uses `MODEL_INPUT_COST_PER_MILLION` and
  `MODEL_OUTPUT_COST_PER_MILLION`. Both default to zero; zero-cost results are
  explicitly labeled `unpriced` so custom/local model cost is never guessed.
- Model, tool, worker, permission, timeout, conflict, interruption, and budget
  failures receive a stable category, severity, retriability flag, fingerprint,
  occurrence count, and related span. Credential-shaped values are redacted
  from classified errors and trace-span errors.
- Terminal runs receive deterministic quality and safety scores based only on
  observable evidence: lifecycle completion, plan completion, verification,
  tool reliability, graph resolution, receipt integrity, bounded execution,
  permission behavior, uncertain side effects, and credential exposure.
- Any terminal run can be saved as a versioned regression scenario. A scenario
  preserves the bounded request and assertions for status, minimum quality and
  safety, maximum latency, and configured cost. Replays create fresh isolated
  agent runs and persist assertion-by-assertion results.

Operator controls add an Observe view with metrics, checks, failures, and the
span timeline, plus an Evals view for saving and replaying scenarios. APIs are
available at `GET /api/agent/runs/{id}/evaluation`,
`GET /api/evals/scenarios`, `POST /api/evals/scenarios/from-run/{id}`, and
`POST /api/evals/scenarios/{id}/replay`. Eval state defaults outside the model
workspace under `.tff_evals`; use `MODEL_EVAL_FILE` to override it.

The `/api/status` response reports `agent_version: "9.0"`.

Agent execution has fixed cumulative budgets:

- `AGENT_MAX_TOOL_ROUNDS` defaults to 320 model/tool rounds (configurable from 1 to 1,000).
- `AGENT_MAX_TOOL_CALLS` defaults to 640 calls, including lifecycle controls (1 to 5,000).
- `AGENT_MAX_SECONDS` defaults to 9,000 elapsed streaming seconds (2.5 hours), including time
  spent waiting for an approval.
- `AGENT_MAX_CONSECUTIVE_FAILURES` defaults to 3 tool failures.

Budgets are captured when a run is created and accumulate across resumes.
Changing these settings applies to new runs; clones retain their source run's budgets.

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
