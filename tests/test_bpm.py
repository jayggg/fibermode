from fibermode import StepIndex, BPM
from ngsolve import InnerProduct

nsteps = 10


def test_bpm_guided_propagation():
    """
    Does BPM propagation preserve guided modes?
    """

    p = 2
    fb = StepIndex(fibername="Nufern_Yb", curveorder=p)
    betas, zsqrs, Y = fb.guidedmodes(p=p,
                                     stop_tol=1e-14,
                                     niterations=200,
                                     verbose=False)
    diff = []

    bpm = BPM(fb)

    for i in range(len(betas)):
        bpm.setupCrankNicolson(0.1, p, kt=betas[i])
        u_initial = Y[i]
        u = bpm.propagateCrankNicolson(u_initial, nsteps)
        u.vec.data -= u_initial.vec
        diff.append(abs(InnerProduct(u.vec, u.vec)))
        print("Case", i, " error", diff[-1])

    print("Max difference after propagation:", max(diff))
    assert max(diff) < 1e-14, \
        "BPM propagation deviates too much from initial mode."
    print("Test passed: BPM propagation preserves guided modes.\n")
    print("#" * 70)


def test_bpm_leaky_propagation():
    """
    Does BPM propagation preserve leaky modes?
    """

    p = 2
    fb = StepIndex(fibername='Nufern_Yb', curveorder=p, R=2)
    center = 1.96 - 0.19j  # center of circle to search for Z-resonance values
    radius = 0.3  # search radius
    diff = []

    zsqrs, Y, Yl, betas, _ = fb.leakymode_auto(p,
                                               radiusZ2=radius**2,
                                               centerZ2=center**2,
                                               alpha=5,
                                               verbose=False,
                                               stop_tol=1e-14,
                                               niterations=200)

    bpm = BPM(fb)

    for i in range(len(betas)):
        bpm.setupCrankNicolson(1e-5,
                               p,
                               kt=betas[i],
                               pml={
                                   'type': 'auto',
                                   'alpha': 5
                               })
        u_initial = Y[i]
        u = bpm.propagateCrankNicolson(u_initial, nsteps)
        u.vec.data -= u_initial.vec
        diff.append(abs(InnerProduct(u.vec, u.vec)))
        print("Case", i, " error", diff[-1])

    print("Max difference after leaky propagation:", max(diff))
    assert max(diff) < 1e-14, \
        "BPM propagation deviates too much from initial mode."
    print("Test passed: BPM propagation preserves leaky modes.\n")
    print("#" * 70)


if __name__ == '__main__':

    test_bpm_guided_propagation()
    test_bpm_leaky_propagation()
