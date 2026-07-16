from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params

if __name__ == '__main__':

    # folder = 'your/output/folder'

    A = PBG(params)

    center = 2
    radius = 1
    p = 3

    z, y, yl, beta, P, _ = A.leakymode(p, rad=radius, ctr=center,
                                       alpha=A.alpha, stop_tol=1e-8,
                                       quadrule='ellipse_trapez_shift',
                                       rhoinv=.8, niterations=12, npts=6,
                                       nspan=3, nrestarts=0)

    y.draw()

    # A.savemodes('pbg', folder, y, p, beta, z)
