from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params

if __name__ == '__main__':

    A = PBG(params)

    center = 3.034
    radius = .1
    p = 2

    betas, zsqrs, E, phi, R = \
        A.leakyvecmodes(rad=radius, ctr=center,
                        alpha=A.alpha, p=p,
                        quadrule='ellipse_trapez_shift', rhoinv=.8,
                        niterations=12, npts=4, stop_tol=1e-11,
                        nspan=3, nrestarts=0)

    E.draw(name='E')
    phi.draw(name='phi')
