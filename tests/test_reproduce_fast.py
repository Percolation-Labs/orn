"""Smoke-scale reproduce tests — one pytest case per fast-tier result.

These actually run the `run()` function from each fast result module with
default args. Each should finish in seconds on CPU.
"""
import pytest
import warnings

from orn.reproduce import list_results

FAST = list_results(tier="fast")


@pytest.mark.reproduce
@pytest.mark.parametrize("result", FAST, ids=[r.code for r in FAST])
def test_fast_reproduce(result):
    with warnings.catch_warnings():
        # numpy BLAS subnormal false-positive FPE — the result modules wrap
        # their matmuls in np.errstate, but some paths still bubble up.
        warnings.simplefilter("ignore", RuntimeWarning)
        out = result.run()
    assert isinstance(out, dict)
    assert len(out) > 0
