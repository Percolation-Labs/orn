# reproduce

Every key result we cite in any paper has one module here. `registry.py` maps
short codes (SPE-01, COU-01, ...) and slugs (`spectral_universality`, ...) to
those modules and tier metadata.

Tiers:
- **fast**           runs in seconds on a laptop; default `pytest` target.
- **slow_reproduce** overnight or cloud run; opt-in via `pytest -m slow_reproduce`.

Add a new result:

1. Create `results/<slug>.py` exposing `def run(device=None, **kw) -> dict`.
2. Register it in `registry.py` with `code`, `slug`, `tier`, `runtime_s`,
   and `papers=[...]` pointing at the relevant .tex.
3. Re-run `pytest tests/test_reproduce_fast.py`. The parametrised test will
   pick it up automatically.

Naming scheme: `<area>_<finding>` (snake_case), where area is one of
`spectral`, `coupling`, `compression`, `scaling`, `transfer`, `multimodal`,
`semigroup`, `operator`, `lorn`, `baseline`. Codes are the first three
letters of the area plus a two-digit index (SPE-01, COU-02, ...).
