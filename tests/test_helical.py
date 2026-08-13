"""Guided modes of helically coiled fibers."""

import importlib.util
import os

import numpy as np


def _demo():
    """Load demos/bent/stepindex_demo.py as a module.

    That demo already encodes the reference example and the scale
    change this test needs, so it is imported rather than duplicated.
    """

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                        'demos', 'bent', 'stepindex_demo.py')
    spec = importlib.util.spec_from_file_location('bent_stepindex_demo', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_helical_stepindex_scale_invariance():
    """Compute a guided mode of a coiled step-index fiber, twice, in
    two different systems of units, and check both against the
    published value and against each other.
    """

    demo = _demo()
    bsqr = demo.betasqr_ref[0]

    (_, betasqr1, Z2_1, cfs1), = demo.findmodes(demo.build(), betasqrs=[bsqr])

    # The search may return more than the wanted mode; keep the one
    # that is actually core-guided and nearest the reference.
    def pick(betasqrs, Z2, cfs):
        j = int(
            np.argmin([
                abs(bs - bsqr) if cf > .5 else np.inf
                for bs, cf in zip(betasqrs, cfs)
            ]))
        assert cfs[j] > .5, 'no core-guided mode found near the reference'
        return betasqrs[j], Z2[j]

    betasqr1, z2_1 = pick(betasqr1, Z2_1, cfs1)
    assert abs(betasqr1 - bsqr) < 1e-2 * bsqr, \
        'guidedhelicalmodes did not find the published mode'

    # Same waveguide, different units (so L = 12.5, not 1).
    scale = 12.5
    (_, betasqr2, Z2_2, cfs2), = demo.findmodes(demo.build(scale),
                                                scale,
                                                betasqrs=[bsqr])
    betasqr2, z2_2 = pick(betasqr2, Z2_2, cfs2)

    assert abs(z2_2 - z2_1) < 1e-3 * abs(z2_1), \
        'Z² changed with the choice of units: guidedhelicalmodes is not ' \
        'nondimensionalizing lengths consistently'


if __name__ == "__main__":
    test_helical_stepindex_scale_invariance()
