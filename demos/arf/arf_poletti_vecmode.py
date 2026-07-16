"""
Computing some modes of a 6-tube ARF structure in Poletti's paper
"""

from fiberamp.fiber.microstruct import ARF

outermaterials = 'air'
freecapil = False
a = ARF(name='poletti', outermaterials=outermaterials, freecapil=freecapil)

# Refine mesh if needed:
# a.refine()

center = 5       # center of circle to search for Z-resonance values
radius = .2            # search radius
alpha = 3

betas, zsqrs, E, phi, _ = a.leakyvecmodes(ctr=center, rad=radius, alpha=alpha,
                                          nspan=4, npts=4, p=1, niterations=20,
                                          nrestarts=0, stop_tol=1e-12)


print('\n'+'#'*64, '\nRESULTS:', '#'*55)
print('Computed non-dimensional Z-squared values:\n', zsqrs)
print('Computed approximation of physical propagation constants:\n', betas)
print('#'*64)

E.draw(name='E')
phi.draw(name='phi')
