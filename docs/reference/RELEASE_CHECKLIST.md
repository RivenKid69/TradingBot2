# Release Checklist

Use this checklist before tagging or publishing a release to keep artifacts reproducible and auditable.

- [ ] **Version bump**: update `pyproject.toml` `[project].version` (and any mirrored metadata) using semantic versioning; ensure the date/version appear in release notes.
- [ ] **Changelog**: add a dated entry to `CHANGELOG.md` covering user-facing changes, migrations, and known issues.
- [ ] **Test matrix**: run `make lint`, `make build`, and `make test` on the supported CI baselines (Ubuntu + Windows, CPU lockfiles) and record outcomes; rerun targeted GPU/integration suites when relevant.
- [ ] **Hash verify**: after building extensions, run `make verify-hash` to confirm `build_hash_report.json` matches compiled artifacts; refresh the report only when changes are intentional.
- [ ] **Dependency audit**: run `pip-audit -r requirements-cpu.lock.txt -r requirements-gpu.lock.txt -r requirements-dev.txt` (adjust if GPU artifacts are not shipped) and address vulnerabilities; regenerate lockfiles with `make lock-cpu` / `make lock-gpu` once updates are applied.
- [ ] **Provenance/clean tree**: ensure `make clean && make check-clean` passes, commit the final state, and tag the release from a clean worktree; capture the commit hash in release notes.
