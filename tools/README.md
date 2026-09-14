# Repository tools

| Folder | Responsibility | Entry point |
|---|---|---|
| `env/` | Install one locked environment | `bash tools/env/sync.sh PROFILE` |
| `backends/` | Fetch pinned sources without replacing edits | `uv run python tools/backends/bootstrap.py simany trellis menagerie` |
| `validation/` | Check release resources and distribution contents | `uv run python tools/validation/check_release.py` |

Runtime calls live under `physicalview plan/run`; see [pipelines](../pipelines/README.md).
Legacy cluster jobs and experiment probes remain in `run/` for historical reproducibility.
Read their site-specific paths before reusing them on a different machine.
