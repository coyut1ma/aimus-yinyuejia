#!/usr/bin/env python3
"""MuseCPEval runner.

Computes the five context-preservation metric families for reference/estimate
audio pairs. All paths are supplied on the command line.

Single pair:
    musecpeval --ref path/to/ref.wav --est path/to/est.wav
    musecpeval --ref ref.wav --est est.wav --out-json result.json
    musecpeval --ref ref.wav --est est.wav --metrics harmony rhythm

Batch (pick exactly one source):
    musecpeval --batch-json pairs.json  --output-dir results/
    musecpeval --batch-csv  pairs.csv   --output-dir results/
    musecpeval --ref-dir orig/ --est-dir edited/ --output-dir results/

    # tuning
    musecpeval --batch-json pairs.json --output-dir results/ --n-workers 16
    musecpeval --batch-json pairs.json --output-dir results/ --no-parallel
    musecpeval --batch-json pairs.json --output-dir results/ --no-resume
    musecpeval --ref-dir orig/ --est-dir edited/ --output-dir results/ --limit 10

Batch writes into --output-dir:
    results.jsonl   one record per pair, appended as each finishes (resume source)
    results.json    the same records as one array, written at the end
    summary.csv     flattened numeric columns, one row per pair

Layout:
    MetricRegistry      the five metric families, their output keys, and the
                        12 paper-reported metrics each one is pruned down to
    JsonCodec           JSON encode/decode, including numpy coercion
    PairPreprocessor    turn manifests or directories into normalized pair dicts
    ResultStore         all filesystem writes and reads under --output-dir
    MetricEvaluator     run metric families on one pair, capturing failures
    RunOutcome          tally failures/degradations into an exit code
    SinglePairRunner    --ref/--est mode
    BatchRunner         batch modes, serial or parallel
    RunnerCLI           argument parsing and dispatch
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .metrics.harmony_tonality import harmony_score
from .metrics.melody_motif import melody_score
from .metrics.rhythm_meter import rhythm_score
from .metrics.structural_form import structural_score
from .metrics.timbre import timbre_score


class MetricRegistry:
    """The metric families, keyed by their CLI name.

    Maps CLI name -> (output key used in records, scoring function). Every
    scoring function takes (audio_ref, audio_est) and returns a dict.
    """

    FAMILIES: dict[str, tuple[str, Callable[[str, str], dict]]] = {
        "harmony": ("harmony_tonality", harmony_score),
        "rhythm": ("rhythm_meter", rhythm_score),
        "structure": ("structural_form", structural_score),
        "melody": ("melodic_content", melody_score),
        "timbre": ("timbre_texture", timbre_score),
    }

    # The 12 metrics reported in the paper (§2), as dotted paths into each
    # family's score dict. Scoring functions also return intermediates that are
    # inputs to these numbers rather than metrics in their own right (e.g.
    # harmony's unnormalized circle-of-fifths step count, which CoF is the
    # [0, 1] normalization of); those are dropped from the record so the output
    # is exactly the reported metric set.
    #
    #   harmony    CoF, ChromaSim, ChromaDTWS
    #   rhythm     dBPM, BeatF, IG
    #   structure  StructPairF, ARI
    #   melody     ContourDTWS, MotifRec
    #   timbre     TimbreSKLS, MFCCcos
    PAPER_KEYS: dict[str, tuple[str, ...]] = {
        "harmony": (
            "key_relatedness.distance_norm_0to1",   # CoF
            "chroma_similarity.mean_chroma_cosine",  # ChromaSim
            "chroma_similarity.chroma_dtw_cosine",   # ChromaDTWS
        ),
        "rhythm": (
            "delta_bpm_folded",                # dBPM
            "beat_mir_eval.F-measure",         # BeatF
            "beat_mir_eval.Information gain",  # IG
        ),
        "structure": (
            "pairwise_f",  # StructPairF
            "ari",         # ARI
        ),
        "melody": (
            "contour_dtw_similarity",  # ContourDTWS
            "motif_3gram_recall",      # MotifRec
        ),
        "timbre": (
            "mfcc_skl_similarity",  # TimbreSKLS
            "mean_mfcc_cosine",     # MFCCcos
        ),
    }

    @classmethod
    def cli_names(cls) -> list[str]:
        return list(cls.FAMILIES)

    @classmethod
    def resolve(cls, cli_name: str) -> tuple[str, Callable[[str, str], dict]]:
        return cls.FAMILIES[cli_name]

    @classmethod
    def keep_paper_keys(cls, cli_name: str, scores: dict) -> dict:
        """Prune a family's score dict down to the paper's reported metrics.

        Keeps the nesting the rest of the pipeline (and summary.csv column
        names) already expects. "error" entries are always kept — they are
        diagnostics, not metrics — and a family with no whitelist is passed
        through untouched.
        """
        allowed = cls.PAPER_KEYS.get(cli_name)
        if not allowed or not isinstance(scores, dict) or "error" in scores:
            return scores

        pruned: dict = {}
        for path in allowed:
            head, _, tail = path.partition(".")
            if head not in scores:
                continue
            if not tail:
                pruned[head] = scores[head]
            elif isinstance(scores[head], dict) and tail in scores[head]:
                pruned.setdefault(head, {})[tail] = scores[head][tail]

        # A sub-dict that failed wholesale reports {"error": ...} instead of
        # its metrics; keep that rather than silently emitting nothing.
        for head, value in scores.items():
            if isinstance(value, dict) and "error" in value and head not in pruned:
                pruned[head] = value
        return pruned


class JsonCodec:
    """JSON encode/decode. Knows how to make numpy output serializable."""

    @staticmethod
    def coerce(obj: Any) -> Any:
        """Coerce numpy scalars/arrays to JSON types; non-finite floats to null."""
        if isinstance(obj, dict):
            return {k: JsonCodec.coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [JsonCodec.coerce(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return JsonCodec.coerce(obj.tolist())
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return value if np.isfinite(value) else None
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    @staticmethod
    def encode_pretty(obj: Any) -> str:
        return json.dumps(obj, indent=2)

    @staticmethod
    def encode_line(obj: Any) -> str:
        """One record as a single JSONL line, newline included."""
        return json.dumps(obj) + "\n"

    @staticmethod
    def decode_manifest(text: str, source: str) -> list[dict]:
        """Read a pair manifest: a JSON list, or {"pairs": [...]} / {"data": [...]}."""
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("pairs", data.get("data"))
        if not isinstance(data, list):
            raise SystemExit(f"{source}: expected a JSON list of pairs, or {{'pairs': [...]}}")
        return data

    @staticmethod
    def decode_lines(text: str) -> tuple[list[dict], int]:
        """Parse JSONL, tolerating a truncated final line from a killed run.

        Returns (records, n_unparseable).
        """
        records, bad = [], 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
        return records, bad


class PairPreprocessor:
    """Builds the normalized pair dicts that everything downstream consumes.

    A normalized pair is {"ref": abs path, "est": abs path, "id"?: str,
    "extras": {passthrough columns}}.
    """

    @staticmethod
    def to_absolute_path(value: Any) -> str:
        """Absolute path string. Tolerates paths that don't exist yet."""
        return str(Path(str(value).strip()).expanduser().resolve())

    @staticmethod
    def normalize_manifest_row(raw: dict, source: str) -> dict:
        """Pull ref/est/id out of a manifest row, keeping any extra columns."""
        row = {str(k).strip(): v for k, v in raw.items() if k is not None}

        ref = row.pop("ref", None) or row.pop("reference", None)
        est = row.pop("est", None) or row.pop("estimate", None) or row.pop("edit", None)
        if not ref or not est:
            raise SystemExit(
                f"{source}: every pair needs 'ref' and 'est' keys; got {sorted(raw)}"
            )

        # Absolute, so a record and its resume key don't depend on the cwd the
        # run was launched from.
        pair = {
            "ref": PairPreprocessor.to_absolute_path(ref),
            "est": PairPreprocessor.to_absolute_path(est),
        }
        pair_id = row.pop("id", None) or row.pop("song_id", None)
        if pair_id:
            pair["id"] = str(pair_id).strip()

        # Anything else (section, slug, n_steps, ...) is carried through to output.
        pair["extras"] = {k: v for k, v in row.items() if v not in (None, "")}
        return pair

    @classmethod
    def from_json_manifest(cls, path: Path) -> list[dict]:
        rows = JsonCodec.decode_manifest(path.read_text(), str(path))
        return [cls.normalize_manifest_row(row, str(path)) for row in rows]

    @classmethod
    def from_csv_manifest(cls, path: Path) -> list[dict]:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise SystemExit(f"{path}: no rows")
        return [cls.normalize_manifest_row(row, str(path)) for row in rows]

    @classmethod
    def from_directories(
        cls,
        ref_dir: Path,
        est_dir: Path,
        ext: str = ".wav",
        recursive: bool = False,
    ) -> list[dict]:
        """Pair files that share a name in both directories.

        Flat by default. With recursive=True, est_dir is walked and each file is
        matched against ref_dir first by mirrored relative path, then by bare
        filename — which is what a nested layout like
        edits_audio/{section}/{slug}/001.wav against a flat original_audio/
        needs. The pair id becomes the relative path without extension
        ("harmony/-3semitone/001"), since every slug reuses the same basenames.
        """
        if not ref_dir.is_dir():
            raise SystemExit(f"missing ref dir: {ref_dir}")
        if not est_dir.is_dir():
            raise SystemExit(f"missing est dir: {est_dir}")

        ext = ext if ext.startswith(".") else f".{ext}"
        candidates = est_dir.rglob("*") if recursive else est_dir.iterdir()
        est_files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() == ext.lower())

        pairs, unmatched = [], 0
        for est in est_files:
            relative = est.relative_to(est_dir)
            ref = ref_dir / relative
            if not ref.exists():
                ref = ref_dir / est.name
            if not ref.exists():
                unmatched += 1
                continue
            pairs.append(
                {
                    "ref": cls.to_absolute_path(ref),
                    "est": cls.to_absolute_path(est),
                    "id": str(relative.with_suffix("")) if recursive else est.stem,
                    "extras": {},
                }
            )

        if unmatched:
            print(
                f"skipped {unmatched} est file(s) with no matching ref in {ref_dir}",
                file=sys.stderr,
            )
        if not pairs:
            hint = "" if recursive else " (try --recursive if est files are in subdirectories)"
            raise SystemExit(
                f"no matching *{ext} pairs between {ref_dir} and {est_dir}{hint}"
            )
        return pairs

    @staticmethod
    def resume_key(pair: dict) -> str:
        """Stable identity for resume. Explicit id wins; otherwise the two paths."""
        return pair.get("id") or f"{pair['ref']}|{pair['est']}"


class ResultStore:
    """Every read and write under --output-dir.

    results.jsonl is appended and flushed per record so a killed run stays
    resumable; results.json and summary.csv are rewritten at the end.
    """

    RESULTS_JSONL = "results.jsonl"
    RESULTS_JSON = "results.json"
    SUMMARY_CSV = "summary.csv"

    LEADING_COLUMNS = ["key", "id", "ref", "est"]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.jsonl_path = output_dir / self.RESULTS_JSONL
        self.json_path = output_dir / self.RESULTS_JSON
        self.csv_path = output_dir / self.SUMMARY_CSV

    # ---------------- setup / prior state ----------------
    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def discard_previous(self) -> None:
        if self.jsonl_path.exists():
            self.jsonl_path.unlink()

    def read_completed(self) -> tuple[list[dict], set[str]]:
        """Prior records and their resume keys, for --resume."""
        if not self.jsonl_path.exists():
            return [], set()

        records, bad = JsonCodec.decode_lines(self.jsonl_path.read_text())
        if bad:
            print(f"ignored {bad} unparseable line(s) in {self.jsonl_path}", file=sys.stderr)
        return records, {r.get("key") for r in records if r.get("key")}

    # ---------------- incremental writing ----------------
    @contextmanager
    def appending(self) -> Iterator[Callable[[dict], None]]:
        """Yield an append(record) that flushes immediately."""
        self.prepare()
        with self.jsonl_path.open("a") as fh:

            def append(record: dict) -> None:
                fh.write(JsonCodec.encode_line(record))
                fh.flush()

            yield append

    # ---------------- final artifacts ----------------
    def write_results_array(self, records: list[dict]) -> None:
        self.json_path.write_text(JsonCodec.encode_pretty(records))

    def write_summary_csv(self, records: list[dict]) -> None:
        rows = [self.flatten_record_to_columns(r) for r in records]
        rest = sorted({k for row in rows for k in row} - set(self.LEADING_COLUMNS))
        columns = [c for c in self.LEADING_COLUMNS if any(c in row for row in rows)] + rest

        with self.csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def flatten_record_to_columns(obj: Any, prefix: str = "") -> dict:
        """Flatten nested metric dicts into dotted column names for CSV.

        {"rhythm_meter": {"beat_mir_eval": {"F-measure": 1.0}}}
            -> {"rhythm_meter.beat_mir_eval.F-measure": 1.0}

        Lists have no sensible column form, so they are kept as JSON text.
        """
        flat: dict[str, Any] = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                flat.update(ResultStore.flatten_record_to_columns(value, child))
        elif isinstance(obj, list):
            flat[prefix] = json.dumps(obj)
        else:
            flat[prefix] = obj
        return flat

    @staticmethod
    def write_json_file(path: Path, payload: Any) -> None:
        """One-off JSON write, used by single-pair mode's --out-json."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(JsonCodec.encode_pretty(payload))


class MetricEvaluator:
    """Runs metric families on one pair and captures how each one went."""

    # structural_form warns rather than raises when msaf and then the
    # uniform-grid fallback both fail; it then scores against a dummy segment,
    # so the number it returns is meaningless. Nothing in the returned dict
    # marks this, so the warning text is the only signal — matched here and
    # surfaced as "degraded_metrics" on the record.
    DEGRADED_WARNING_RE = re.compile(
        r"any score derived from this file is meaningless"
        r"|scores are not comparable to msaf-derived ones",
        re.IGNORECASE,
    )

    def __init__(self, metrics: list[str] | None = None, quiet: bool = False):
        self.metrics = metrics or MetricRegistry.cli_names()
        self.quiet = quiet

    def evaluate(self, ref: Any, est: Any) -> dict:
        """Score one pair.

        A family that raises is recorded as {"error": ...} so one bad family
        does not lose the others. Families whose score came off a degraded code
        path are listed under "degraded_metrics".
        """
        ref_str, est_str = str(ref), str(est)
        scores: dict = {}
        degraded: list[str] = []

        for cli_name in self.metrics:
            output_key, score_fn = MetricRegistry.resolve(cli_name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    scores[output_key] = MetricRegistry.keep_paper_keys(
                        cli_name, score_fn(ref_str, est_str)
                    )
                except Exception as exc:
                    scores[output_key] = {"error": f"{type(exc).__name__}: {exc}"}
                    self._note(f"  ! {output_key}: {scores[output_key]['error']}")

            if self._warns_degraded(caught):
                degraded.append(output_key)
                self._note(f"  ~ {output_key}: degraded — score is not meaningful")

        if degraded:
            scores["degraded_metrics"] = degraded
        return scores

    @classmethod
    def _warns_degraded(cls, caught) -> bool:
        return any(cls.DEGRADED_WARNING_RE.search(str(w.message)) for w in caught)

    def _note(self, message: str) -> None:
        if not self.quiet:
            print(message, file=sys.stderr)


class RunOutcome:
    """Tallies how a batch went and turns that into a process exit code."""

    def __init__(self, records: list[dict], interrupted: bool = False):
        self.records = records
        self.interrupted = interrupted
        self.failed = [r for r in records if r.get("error")]
        self.degraded = [r for r in records if r.get("degraded_metrics")]
        self.partial = [
            r
            for r in records
            if not r.get("error")
            and any(isinstance(v, dict) and "error" in v for v in r.values())
        ]

    def describe(self, json_path: Path) -> str:
        return (
            f"\ndone — {len(self.records)} record(s) → {json_path}\n"
            f"  failed pairs:            {len(self.failed)}\n"
            f"  pairs w/ failed metric:  {len(self.partial)}\n"
            f"  degraded (meaningless):  {len(self.degraded)}"
        )

    def exit_code(self) -> int:
        if self.interrupted:
            return 130
        return 1 if (self.failed or self.partial or self.degraded) else 0


class SinglePairRunner:
    """--ref/--est mode: one pair, JSON to stdout or --out-json."""

    def __init__(self, metrics: list[str], out_json: Path | None = None):
        self.metrics = metrics
        self.out_json = out_json

    def run(self, ref: Path, est: Path) -> dict:
        for label, path in (("ref", ref), ("est", est)):
            if not path.is_file():
                raise SystemExit(f"missing {label} audio: {path}")

        print(f"ref: {ref}", file=sys.stderr)
        print(f"est: {est}", file=sys.stderr)
        print(f"metrics: {', '.join(self.metrics)}", file=sys.stderr)

        record = {"ref": str(ref), "est": str(est)}
        record.update(MetricEvaluator(self.metrics).evaluate(ref, est))
        payload = JsonCodec.coerce(record)

        if self.out_json:
            ResultStore.write_json_file(self.out_json, payload)
            print(f"saved: {self.out_json}", file=sys.stderr)
        else:
            print(JsonCodec.encode_pretty(payload))

        return payload


class BatchRunner:
    """Runs many pairs, serially or across worker processes."""

    def __init__(
        self,
        metrics: list[str],
        output_dir: Path,
        n_workers: int = 1,
        resume: bool = True,
    ):
        self.metrics = metrics
        self.n_workers = n_workers
        self.resume = resume
        self.store = ResultStore(output_dir)

    # ---------------- worker side ----------------
    @staticmethod
    def score_pair_task(task: tuple) -> dict:
        """Worker entry point.

        Must stay a staticmethod: an instance method would pickle the whole
        BatchRunner (including its open results.jsonl handle) into each child.
        """
        pair, metrics = task
        record = {
            "key": PairPreprocessor.resume_key(pair),
            "ref": pair["ref"],
            "est": pair["est"],
        }
        if "id" in pair:
            record["id"] = pair["id"]
        record.update(pair.get("extras") or {})

        ref, est = Path(pair["ref"]), Path(pair["est"])
        for label, path in (("ref", ref), ("est", est)):
            if not path.is_file():
                record["error"] = f"missing {label} audio: {path}"
                return JsonCodec.coerce(record)

        try:
            record.update(MetricEvaluator(metrics, quiet=True).evaluate(ref, est))
        except Exception as exc:  # a metric family escaping its own handler
            record["error"] = f"{type(exc).__name__}: {exc}"
        return JsonCodec.coerce(record)

    # ---------------- driver side ----------------
    def run(self, pairs: list[dict]) -> int:
        self.store.prepare()

        done_records, done_keys = [], set()
        if self.resume:
            done_records, done_keys = self.store.read_completed()
            if done_keys:
                print(f"resuming — {len(done_keys)} pair(s) already done", file=sys.stderr)
        else:
            self.store.discard_previous()

        todo = [p for p in pairs if PairPreprocessor.resume_key(p) not in done_keys]
        print(
            f"{len(pairs)} pair(s) total, {len(todo)} to run, {self.n_workers} worker(s)\n"
            f"metrics: {', '.join(self.metrics)}\noutput:  {self.store.output_dir}",
            file=sys.stderr,
        )

        new_records: list[dict] = []
        interrupted = False

        with self.store.appending() as append:

            def keep(record: dict, index: int) -> None:
                append(record)
                new_records.append(record)
                self._log_progress(record, index, len(todo))

            try:
                if self.n_workers == 1:
                    self._run_serial(todo, keep)
                else:
                    self._run_parallel(todo, keep)
            except KeyboardInterrupt:
                interrupted = True
                print("\ninterrupted — partial results kept; rerun to resume", file=sys.stderr)

        return self._finalize(done_records + new_records, interrupted)

    def _run_serial(self, todo: list[dict], keep: Callable[[dict, int], None]) -> None:
        for index, pair in enumerate(todo, 1):
            keep(self.score_pair_task((pair, self.metrics)), index)

    def _run_parallel(self, todo: list[dict], keep: Callable[[dict, int], None]) -> None:
        # Spawn, not fork: the thread limits set by RunnerCLI only take effect
        # in a child that imports numpy/librosa fresh.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(self.score_pair_task, (pair, self.metrics)): pair for pair in todo
            }
            for index, future in enumerate(as_completed(futures), 1):
                pair = futures[future]
                try:
                    keep(future.result(), index)
                except Exception as exc:
                    keep(
                        {
                            "key": PairPreprocessor.resume_key(pair),
                            "ref": pair["ref"],
                            "est": pair["est"],
                            "error": f"worker died: {type(exc).__name__}: {exc}",
                        },
                        index,
                    )

    def _finalize(self, all_records: list[dict], interrupted: bool) -> int:
        self.store.write_results_array(all_records)
        if all_records:
            self.store.write_summary_csv(all_records)

        outcome = RunOutcome(all_records, interrupted)
        print(outcome.describe(self.store.json_path), file=sys.stderr)
        return outcome.exit_code()

    @staticmethod
    def _log_progress(record: dict, index: int, total: int) -> None:
        flag = ""
        if record.get("error"):
            flag = f"  ! {record['error']}"
        elif record.get("degraded_metrics"):
            flag = f"  ~ degraded: {', '.join(record['degraded_metrics'])}"
        print(f"[{index}/{total}] {record['key']}{flag}", file=sys.stderr)


class RunnerCLI:
    """Argument parsing, input-source validation, and dispatch."""

    BLAS_THREAD_VARS = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        all_metrics = MetricRegistry.cli_names()
        parser = argparse.ArgumentParser(
            description="Run MuseCPEval context-preservation metrics on one pair or a batch.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                "Exactly one input source is required:\n"
                "  --ref + --est          single pair\n"
                "  --batch-json FILE      [{ref, est, id?, ...}, ...]\n"
                "  --batch-csv FILE       columns ref,est[,id,...]\n"
                "  --ref-dir + --est-dir  pair files sharing a name\n"
            ),
        )

        single = parser.add_argument_group("single pair")
        single.add_argument("--ref", type=Path, help="Reference audio file.")
        single.add_argument("--est", type=Path, help="Estimated/edited audio file.")
        single.add_argument("--out-json", type=Path, help="Write result here instead of stdout.")

        batch = parser.add_argument_group("batch")
        batch.add_argument("--batch-json", type=Path, help="JSON manifest of pairs.")
        batch.add_argument("--batch-csv", type=Path, help="CSV manifest of pairs.")
        batch.add_argument("--ref-dir", type=Path, help="Directory of reference audio.")
        batch.add_argument("--est-dir", type=Path, help="Directory of estimated audio.")
        batch.add_argument(
            "--ext", default=".wav", help="Extension for dir pairing (default: .wav)."
        )
        batch.add_argument(
            "--recursive",
            action="store_true",
            help="Walk --est-dir subdirectories; match refs by relative path then filename.",
        )
        batch.add_argument(
            "--output-dir", type=Path, default=Path("./results"), help="Batch output directory."
        )
        batch.add_argument(
            "--n-workers",
            type=int,
            default=None,
            help="Parallel worker processes (default: min(8, cpus - 1)).",
        )
        batch.add_argument("--max-cpu", action="store_true", help="Use every CPU as a worker.")
        batch.add_argument("--no-parallel", action="store_true", help="Run serially in-process.")
        batch.add_argument(
            "--no-resume", action="store_true", help="Ignore and overwrite prior results.jsonl."
        )
        batch.add_argument("--limit", type=int, help="Only run the first N pairs (smoke tests).")

        parser.add_argument(
            "--metrics",
            nargs="+",
            choices=all_metrics,
            default=all_metrics,
            help=f"Metric families to run (default: all — {' '.join(all_metrics)}).",
        )
        return parser

    @staticmethod
    def resolve_input_source(args: argparse.Namespace) -> tuple[str, list[dict] | None]:
        """Validate that exactly one input source was given, and load it."""
        chosen = [
            name
            for name, given in (
                ("single", bool(args.ref or args.est)),
                ("batch-json", bool(args.batch_json)),
                ("batch-csv", bool(args.batch_csv)),
                ("dirs", bool(args.ref_dir or args.est_dir)),
            )
            if given
        ]
        if not chosen:
            raise SystemExit(
                "no input given: use --ref/--est, --batch-json, --batch-csv, "
                "or --ref-dir/--est-dir"
            )
        if len(chosen) > 1:
            raise SystemExit(f"pick one input source, got: {', '.join(chosen)}")

        mode = chosen[0]
        if mode == "single":
            if not (args.ref and args.est):
                raise SystemExit("single mode needs both --ref and --est")
            return mode, None
        if mode == "batch-json":
            return mode, PairPreprocessor.from_json_manifest(args.batch_json)
        if mode == "batch-csv":
            return mode, PairPreprocessor.from_csv_manifest(args.batch_csv)

        if not (args.ref_dir and args.est_dir):
            raise SystemExit("dir mode needs both --ref-dir and --est-dir")
        return mode, PairPreprocessor.from_directories(
            args.ref_dir, args.est_dir, args.ext, recursive=args.recursive
        )

    @staticmethod
    def resolve_worker_count(args: argparse.Namespace, n_pairs: int) -> int:
        if args.no_parallel:
            return 1
        cpus = os.cpu_count() or 1
        if args.max_cpu:
            requested = cpus
        elif args.n_workers:
            requested = args.n_workers
        else:
            requested = min(8, max(1, cpus - 1))
        return max(1, min(requested, n_pairs))

    @classmethod
    def limit_blas_threads(cls) -> None:
        """One BLAS/OpenMP thread per worker.

        Without this, N workers each spin up N threads and spend the run
        fighting for cores. Must run before the pool spawns so children pick it
        up at import time.
        """
        for var in cls.BLAS_THREAD_VARS:
            os.environ.setdefault(var, "1")

    @classmethod
    def main(cls, argv: list[str] | None = None) -> int:
        args = cls.build_parser().parse_args(argv)
        mode, pairs = cls.resolve_input_source(args)

        if mode == "single":
            SinglePairRunner(
                metrics=args.metrics,
                out_json=args.out_json.resolve() if args.out_json else None,
            ).run(ref=args.ref.resolve(), est=args.est.resolve())
            return 0

        assert pairs is not None  # every batch mode returns a pair list
        if args.limit:
            pairs = pairs[: args.limit]
        n_workers = cls.resolve_worker_count(args, len(pairs))
        if n_workers > 1:
            cls.limit_blas_threads()

        return BatchRunner(
            metrics=args.metrics,
            output_dir=args.output_dir.resolve(),
            n_workers=n_workers,
            resume=not args.no_resume,
        ).run(pairs)


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (``musecpeval``)."""
    return RunnerCLI.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
