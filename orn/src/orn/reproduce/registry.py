"""Registry of reproducible key results.

Each `Result` entry names a code (e.g. `SPE-01`), a slug (`spectral_universality`),
a tier (`fast` vs `slow_reproduce`), a human title, and a callable `run(device=None)`.

Fast-tier results run in under ~5 minutes total on a laptop CPU/MPS — they are
included in the default `pytest` run. Slow-tier results are opt-in via
`pytest -m slow_reproduce` or `orn reproduce --slow`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Result:
    code: str
    slug: str
    title: str
    tier: str                    # "fast" or "slow_reproduce"
    runtime_s: int               # rough wall-clock estimate on a laptop
    papers: list[str] = field(default_factory=list)
    _loader: Callable[[], Callable] = None  # returns the run() callable, lazy

    @property
    def run(self) -> Callable:
        if self._loader is None:
            raise RuntimeError(f"Result {self.code} has no loader")
        return self._loader()


def _lazy(module_path: str, fn_name: str = "run"):
    """Defer import of result modules — avoids paying the cost until invoked."""
    def loader():
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, fn_name)
    return loader


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

REGISTRY: dict[str, Result] = {}


def _register(r: Result):
    REGISTRY[r.code] = r


# ── Spectral structure of M ──────────────────────────────────────────────

_register(Result(
    code="SPE-01",
    slug="spectral_universality",
    title="W_Q^T W_K eigenspectra correlate 0.93+ across GPT-2 / SmolLM2 / Qwen2.5",
    tier="fast",
    runtime_s=60,
    papers=["born/paper/orn_lab_notebook.tex"],
    _loader=_lazy("orn.reproduce.results.spectral_universality"),
))

_register(Result(
    code="SPE-02",
    slug="m_crystallisation",
    title="M collapses to rank 19-20 of 512 during training (91M scale)",
    tier="slow_reproduce",
    runtime_s=2700,
    papers=["born/paper/orn_lab_notebook.tex"],
    _loader=_lazy("orn.reproduce.results.m_crystallisation"),
))


# ── Coupling structure ──────────────────────────────────────────────────

_register(Result(
    code="COU-01",
    slug="one_m_recovers_gpt2",
    title="Single shared M reconstructs 95.8% of GPT-2 attention",
    tier="fast",
    runtime_s=15,
    papers=["born/paper/orn_lab_notebook.tex"],
    _loader=_lazy("orn.reproduce.results.one_m_recovers_gpt2"),
))

_register(Result(
    code="COU-02",
    slug="coupling_manifold_8d",
    title="8-dim manifold spans GPT-2 per-layer coupling",
    tier="fast",
    runtime_s=30,
    _loader=_lazy("orn.reproduce.results.coupling_manifold_8d"),
))

_register(Result(
    code="COU-03",
    slug="asymmetric_m_required",
    title="AB^T works, symmetric LL^T fails (autoregressive)",
    tier="fast",
    runtime_s=90,
    _loader=_lazy("orn.reproduce.results.asymmetric_m_required"),
))


# ── Coupling memoisation (colour-matching control task) ────────────────

_register(Result(
    code="COM-01",
    slug="colour_matching_compression",
    title="16x coupling-param reduction, no quality loss on provably-invariant task",
    tier="fast",
    runtime_s=120,
    _loader=_lazy("orn.reproduce.results.colour_matching_compression"),
))


# ── Operator programme ─────────────────────────────────────────────────

_register(Result(
    code="OPE-01",
    slug="function_vs_content_attention",
    title="Function words stable (0.68-0.77), names context-dependent (0.33-0.44)",
    tier="fast",
    runtime_s=60,
    papers=["papers/drafts/taking_orbits_to_the_limit.tex"],
    _loader=_lazy("orn.reproduce.results.function_vs_content_attention"),
))

_register(Result(
    code="OPE-04",
    slug="lie_trotter_splitting",
    title="ORN layer structure = operator splitting for exp(t(A+B))",
    tier="fast",
    runtime_s=30,
    papers=["papers/drafts/taking_orbits_to_the_limit.tex"],
    _loader=_lazy("orn.reproduce.results.lie_trotter_splitting"),
))


# ── LORN ───────────────────────────────────────────────────────────────

_register(Result(
    code="LOR-01",
    slug="vq_codebook_health",
    title="VQREncoder codebook fills, entropy stays up, no collapse",
    tier="fast",
    runtime_s=60,
    papers=["lorn/paper/two_hemisphere_concept_circuits.tex"],
    _loader=_lazy("orn.reproduce.results.vq_codebook_health"),
))


# ═══════════════════════════════════════════════════════════════════════════════
# Lookups
# ═══════════════════════════════════════════════════════════════════════════════

def list_results(tier: str | None = None) -> list[Result]:
    out = list(REGISTRY.values())
    if tier:
        out = [r for r in out if r.tier == tier]
    return sorted(out, key=lambda r: r.code)


def get(identifier: str) -> Result:
    """Resolve by code (SPE-01) or slug (spectral_universality)."""
    u = identifier.upper().replace("_", "-")
    if u in REGISTRY:
        return REGISTRY[u]
    for r in REGISTRY.values():
        if r.slug == identifier.lower():
            return r
    raise KeyError(f"No result {identifier!r}. Codes: {list(REGISTRY)}")


def resolve(identifiers: list[str]) -> list[Result]:
    """Resolve a list of codes/slugs/area prefixes.

    A bare prefix like 'spectral' matches all results whose code starts with
    SPE- or whose slug starts with 'spectral'.
    """
    out: list[Result] = []
    seen: set[str] = set()
    for ident in identifiers:
        try:
            r = get(ident)
            if r.code not in seen:
                out.append(r); seen.add(r.code)
            continue
        except KeyError:
            pass
        prefix = ident.lower()
        matched = [r for r in REGISTRY.values()
                   if r.code.lower().startswith(prefix[:3]) or r.slug.startswith(prefix)]
        if not matched:
            raise KeyError(f"No result matching {ident!r}")
        for r in matched:
            if r.code not in seen:
                out.append(r); seen.add(r.code)
    return sorted(out, key=lambda r: r.code)
