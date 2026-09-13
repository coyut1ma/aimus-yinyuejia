# Music Edit Demo

This is an end-to-end MUSDB18 demo for editing a selected time range from a
Chinese natural-language instruction. It supports single-stem and multistem
plans over `vocals`, `drums`, `bass`, and `other`, produces three candidates,
crossfades the edited region, remixes the stems, and exposes both a CLI and a
FastAPI service.

The default `fake` generator is an explicitly labelled DSP baseline. It proves
that the workflow and preservation constraints work, but it is not a music
generation model and must not be used to assess generation quality. No
training or fine-tuning is performed by this demo.

## Setup

From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[api]" --no-build-isolation
```

PyAV decodes MUSDB18 `.stem.mp4` containers directly, so a system FFmpeg
installation is not required.

## One-command real-data demo

With MUSDB18 at `..\musdb18`:

```powershell
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real `
  --musdb-root ..\musdb18 `
  demo-real
```

This deterministically selects 10 test songs of at least 94 seconds, writes the
frozen 20-task manifest, imports the audio needed by `pilot_v1_01_1`, and runs
the 20-28 second drum-density edit. Decoded stems are cached. Re-running the
same song does not decode it again.

Run a different frozen task with, for example:

```powershell
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real `
  --musdb-root ..\musdb18 `
  demo-real --task-id pilot_v1_01_2
```

`pilot_v1_01_2` edits drums and bass together. Results are written under
`runtime_real\artifacts\<job_id>\candidates\seed_<seed>\rendered`.

## Inspect and run the pilot set

```powershell
# Inspect dataset tracks.
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real --musdb-root ..\musdb18 `
  musdb-list --split test --limit 3

# Build the bound 20-task manifest.
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real --musdb-root ..\musdb18 `
  pilot-build --output runtime_real\pilot_tasks_v1.jsonl

# Run one task from that manifest.
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real --musdb-root ..\musdb18 `
  pilot-run --manifest runtime_real\pilot_tasks_v1.jsonl `
  --task-id pilot_v1_01_1
```

There are two tasks per song: one single-stem task and one multistem task. The
task IDs, instructions, edit regions, expected targets, preservation rules,
and seeds are fixed. These 20 items are an evaluation set, not training data.

## API

```powershell
$env:MUSIC_EDIT_RUNTIME_DIR = (Resolve-Path .\runtime_real)
$env:MUSDB_ROOT = (Resolve-Path ..\musdb18)
.\.venv\Scripts\python.exe -m music_edit_demo serve --port 8001
```

Open `http://127.0.0.1:8001/docs`. The main flow is:

1. `GET /v1/musdb/tracks?split=test`
2. `POST /v1/musdb/import`
3. `POST /v1/edit-plans`
4. `POST /v1/edit-jobs`
5. Poll `GET /v1/edit-jobs/{job_id}`
6. Read `GET /v1/edit-jobs/{job_id}/result`
7. Download candidate mixes or individual stems from the audio endpoints

## Real model adapters

Natural-language understanding can be delegated to a Qwen OpenAI-compatible
endpoint:

```powershell
$env:MUSIC_EDIT_PARSER = "qwen"
$env:QWEN_BASE_URL = "http://<qwen-host>:<port>/v1"
$env:QWEN_MODEL = "Qwen3-4B-Instruct"
```

Music generation can be delegated to an ACE-Step repaint service that follows
the contract in `AceStepHttpGenerator`:

```powershell
$env:MUSIC_EDIT_GENERATOR = "ace_step_http"
$env:ACE_STEP_BASE_URL = "http://<gpu-host>:<port>"
```

The remote service receives the four source paths, structured target edits,
time range, prompt, preservation constraints, and seeds. It must return one
full-duration WAV per requested target stem and seed on a shared filesystem.

For an ablation that keeps the ACE-Step joint frontend path but disables the
cross-stem attention exchange, set:

```powershell
$env:ACE_STEP_CROSS_STEM_ATTENTION = "0"
```

This still sends all four stems and uses the shared first-block joint path on
the ACE-Step side; only the cross-stem attention residual is skipped.

## Runtime configuration

| Variable | Default | Purpose |
|---|---|---|
| `MUSIC_EDIT_RUNTIME_DIR` | `./runtime` | Projects, plans, jobs, results, and artifacts |
| `MUSDB_ROOT` | empty | Directory containing `train` and `test` |
| `MUSIC_EDIT_PARSER` | `controlled` | `controlled` or `qwen` |
| `MUSIC_EDIT_GENERATOR` | `fake` | `fake` or `ace_step_http` |
| `QWEN_BASE_URL` | empty | OpenAI-compatible base URL |
| `QWEN_API_KEY` | empty | Optional parser API key |
| `QWEN_MODEL` | `Qwen3-4B-Instruct` | Parser model name |
| `ACE_STEP_BASE_URL` | empty | ACE-Step adapter service URL |

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use synthetic audio. They verify instruction parsing, all 20 frozen
task templates, target-only editing, non-target preservation, outside-region
sample equality, and distinct candidate outputs without requiring MUSDB18.

## Local-edit evaluation

The demo includes a deterministic evaluator for the three preservation
measurements used by local editing:

- beat timing error in the edited mix, using `librosa.beat.beat_track` on the
  unedited context and edited region, then matching their beat positions;
- onset alignment error between each edited target stem and every unedited
  reference stem, after subtracting their natural offset outside the edit;
- RMS level deviation in dB between the edited target region and the same
  stem's unedited context.

It does not calculate a composite score or rank candidates.  A level deviation
of at most 3 dB is reported as `within_good_threshold: true`.

Evaluate the selected candidate from a completed job:

```powershell
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real `
  evaluate --job-id job_<id> `
  --output runtime_real\evaluations\job_<id>.json
```

Evaluate a particular ACE candidate instead:

```powershell
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real `
  evaluate --job-id job_<id> --seed 202
```

Use `--all` to evaluate every candidate in the result manifest. When writing
multiple reports, the output file contains a JSON array.

The same operation is available as `POST /v1/edit-jobs/{job_id}/evaluation`
with an optional `seed` query parameter.  Reports use schema `evaluation.v1`.

## MuseCPEval context-preservation evaluation

The repository also carries MuseCPEval under `../third_party/MuseCPEval` as an
optional objective evaluator for context preservation.  It compares the
original full mix with an edited candidate mix and can also compare edited
target stems against their original stems.

Install the downloaded tool into the active environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ..\third_party\MuseCPEval
```

Build a MuseCPEval manifest for the selected candidate of a completed job:

```powershell
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real `
  musecpeval-manifest --job-id job_<id> `
  --output runtime_real\musecpeval\job_<id>\pairs.json
```

Run MuseCPEval directly from the demo.  By default this runs harmony, rhythm,
melody, and timbre; add `structure` only after installing MuseCPEval's optional
structure dependencies.

```powershell
.\.venv\Scripts\python.exe -m music_edit_demo `
  --runtime-dir runtime_real `
  musecpeval-run --job-id job_<id> --all --include-target-stems `
  --output-dir runtime_real\musecpeval\job_<id>\results
```

Output files follow MuseCPEval's own format: `results.jsonl`, `results.json`,
and `summary.csv`.
