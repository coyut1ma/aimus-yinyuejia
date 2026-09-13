# MuseCPEval

Measures **music context preservation** between an original audio file and an edited
version of it: how much of the original's harmony, rhythm, structure, melody, and
timbre survives the edit.

Give it a reference/estimate pair and it returns a JSON object of scores. Give it a
manifest or a pair of directories and it scores thousands of pairs in parallel.

## Metrics

| CLI name | Output key | Returns |
|---|---|---|
| `harmony` | `harmony_tonality` | `key_relatedness` (`distance_steps`, `distance_norm_0to1`), `chroma_similarity` (`chroma_dtw_cosine`, `mean_chroma_cosine`) |
| `rhythm` | `rhythm_meter` | `delta_bpm_folded`, `beat_mir_eval` (`F-measure`, `Information gain`) |
| `structure` | `structural_form` | `pairwise_f`, `ari` |
| `melody` | `melodic_content` | `contour_dtw_similarity`, `motif_3gram_recall` |
| `timbre` | `timbre_texture` | `mfcc_skl_similarity`, `mean_mfcc_cosine` |

All five run by default. Scoring an identical pair (same file as reference and
estimate) returns `1.0` on the similarity metrics and `0.0` on the distance metrics —
`mean_chroma_cosine` lands a few float ulps short of `1.0` rather than exactly on it.
That is a quick way to confirm an install is sane.

## Data

The audio used for the objective evaluation is available on
[Google Drive](https://drive.google.com/file/d/1KcMMKKes7gmxfvtBx9sEB7EWLeLfLf9M/view?usp=sharing).

## Install

The metric code needs **numpy ≤ 2.2** — librosa depends on numba, and numba refuses
newer numpy — so a dedicated environment is the least painful route:

```bash
conda create -y -n musecpeval python=3.10
conda activate musecpeval

pip install musecpeval==0.3.0
```

Pinning the version is the recommended form — the metric numbers are what you cite, so
you want them reproducible. `pip install musecpeval` takes the newest release instead.
Published at [pypi.org/project/musecpeval](https://pypi.org/project/musecpeval/).

<details>
<summary>Installing from GitHub instead</summary>

For an unreleased change, or to work on the metrics:

```bash
pip install "git+https://github.com/Yashvishe13/MuseCPEval.git"   # straight from main
pip install "git+https://github.com/Yashvishe13/MuseCPEval.git@v0.1.0"  # at a tag

git clone https://github.com/Yashvishe13/MuseCPEval.git && cd MuseCPEval
pip install .                   # from a clone
pip install -e .                # editable, so edits to the metrics take effect
```
</details>

Either way you get the `musecpeval` command and the importable package:

```python
from musecpeval import (
    harmony_score, melody_score, rhythm_score, structural_score, timbre_score,
)

harmony_score("original.wav", "edited.wav")
```

`--metrics structure` additionally needs **msaf**, which is not pulled in by default:
its PyPI release is old and pins numpy/scipy versions that fight the rest of the stack,
so install it yourself and expect to referee the pins.

```bash
pip install msaf                # or: pip install "musecpeval[structure]"
```

Verify:

```bash
musecpeval --ref path/to/file.wav --est path/to/file.wav
```

`python -m musecpeval` is equivalent, and `python runner.py` still works from a clone
without installing anything.

## Run

### Single pair

```bash
# JSON to stdout
musecpeval --ref original.wav --est edited.wav

# JSON to a file
musecpeval --ref original.wav --est edited.wav --out-json result.json

# a subset of metrics
musecpeval --ref original.wav --est edited.wav --metrics harmony rhythm
```

### Batch

Exactly one input source is required.

```bash
# JSON manifest
musecpeval --batch-json pairs.json --output-dir results/

# CSV manifest
musecpeval --batch-csv pairs.csv --output-dir results/

# two flat directories, paired by filename
musecpeval --ref-dir originals/ --est-dir edited/ --output-dir results/

# nested edits (edited/<section>/<slug>/001.wav) against flat originals
musecpeval --ref-dir originals/ --est-dir edited/ --recursive --output-dir results/
```

```json
[
  {"ref": "originals/001.wav", "est": "edited/harmony/-3semitone/001.wav",
   "id": "harmony/-3semitone/001", "section": "harmony", "n_steps": -3}
]
```

### Output

`--output-dir` receives three files:

| File | Contents |
|---|---|
| `results.jsonl` | one record per pair, appended and flushed as each finishes |
| `results.json` | the same records as a single array, written at the end |
| `summary.csv` | one row per pair, nested scores flattened to dotted columns |

`results.jsonl` is what makes a run resumable, so it is written incrementally;
`results.json` and `summary.csv` only appear once the run finishes.

### Resuming

Rerunning the same command skips pairs already present in `results.jsonl`, so an
interrupted job continues instead of restarting. Ctrl-C exits `130` with partial
results intact. Pass `--no-resume` to discard prior results and start clean.

Pairs are identified by `id` when present, otherwise by the absolute ref and est paths.

There is no lock file: **two runs against the same `--output-dir` can process the same
pairs.** Use separate output directories for concurrent jobs.

## Options

| Flag | Meaning |
|---|---|
| `--ref FILE`, `--est FILE` | single-pair input |
| `--out-json FILE` | single mode: write here instead of stdout |
| `--batch-json FILE` | batch from a JSON manifest |
| `--batch-csv FILE` | batch from a CSV manifest |
| `--ref-dir DIR`, `--est-dir DIR` | batch by pairing two directories |
| `--recursive` | walk `--est-dir` subdirectories |
| `--ext EXT` | extension for directory pairing (default `.wav`) |
| `--output-dir DIR` | batch output directory (default `./results`) |
| `--metrics ...` | any of `harmony rhythm structure melody timbre` (default: all) |
| `--n-workers N` | worker processes (default `min(8, cpus - 1)`) |
| `--max-cpu` | use every CPU as a worker |
| `--no-parallel` | run serially in one process |
| `--no-resume` | ignore and overwrite prior results |
| `--limit N` | only the first N pairs, for smoke tests |

Worker count is capped at the number of pairs.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | everything scored cleanly |
| `1` | a pair failed, a metric family failed, or a score was degraded — or bad arguments |
| `130` | interrupted; partial results kept and resumable |


## Repository layout

```
musecpeval/
  __init__.py          lazily re-exports the five scoring functions
  __main__.py          `python -m musecpeval`
  runner.py            CLI: single-pair and batch evaluation
  metrics/             the five metric families
    harmony_tonality.py  key relatedness, chroma similarity
    rhythm_meter.py      tempo delta, beat F-measure
    structural_form.py   segmentation agreement (needs msaf)
    melody_motif.py      contour DTW, motif n-gram recall
    timbre.py            MFCC symmetric-KL similarity, mean-MFCC cosine
    utils.py             shared helpers
pyproject.toml         deps, the [structure] extra, the `musecpeval` entry point
requirements.txt       mirror of the dependency list, for `pip install -r`
runner.py              shim, so `python runner.py` keeps working from a clone
```

Each metric module also has its own `__main__` block that scores a directory of pairs
for that family alone — `python -m musecpeval.metrics.rhythm_meter --orig-dir a/
--edit-dir b/`. The flag spelling is not consistent between them: `rhythm_meter.py`
and `timbre.py` take `--orig-dir` / `--edit-dir`, the other three take `--orig_dir` /
`--edit_dir`.
