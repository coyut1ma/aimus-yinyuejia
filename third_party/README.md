# Third-party evaluation tools

## MuseCPEval

Source: <https://github.com/Yashvishe13/MuseCPEval>

Local checkout:

```text
third_party/MuseCPEval/
```

Downloaded commit:

```text
0121201504e614e1f059994427fa0dd733c2badf
```

Purpose in this workspace: an optional context-preservation evaluator for
music-editing outputs. It compares a reference audio file with an edited audio
file and reports harmony, rhythm, melody, timbre, and optional structure
metrics.

Install it in the active music-edit environment from the local checkout:

```bash
cd music_edit_demo
python -m pip install -e ../third_party/MuseCPEval
```

The `structure` metric needs the optional `msaf` dependency:

```bash
python -m pip install -e "../third_party/MuseCPEval[structure]"
```

Use the `music-edit-demo musecpeval-manifest` and `music-edit-demo
musecpeval-run` commands to evaluate completed edit jobs.
