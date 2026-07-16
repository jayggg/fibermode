
# Geometric Parameters.  Dimensional

p = 6                      # number of sides of polygonal lattice
layers = 6                 # number of layers of lattice
skip = 2                   # number of layers to skip before beginning lattice
pattern = []               # pattern determining microstructure

Λ = 7e-6                             # separation between layers
r_tube = .5 * .4 * Λ                 # radius of inner tubes
r_core = .7 * (Λ * skip - r_tube)    # radius of (computational) core region
r_fiber = (skip + layers + 1.5) * Λ   # radius of fiber, end of cladding region

t_poly = 0 * r_tube        # polymer jacket thickness
t_buffer = 0 * r_tube      # thickness of buffer region
t_outer = 3 * r_core       # thickness of PML region

r_poly = r_fiber + t_poly  # end of polymer region, start buffer region
r_pml = r_poly + t_buffer  # end of buffer, start of PML region
r_out = r_pml + t_outer    # end of domain

scale = r_core            # scaling factor


# Physical Parameters in Vacuum (Dimensional).

wavelength = 1.55e-6


# Refractive indices
# Here we need to introduce materials and Sellmeier

n_tube = 1.48                       # refractive index of tube material
n_clad = 1.45                       # refractive index of cladding material
n_core = n_clad                     # refractive index of core
n_poly = n_clad                        # refractive index of polymer
n_buffer = n_clad                        # refractive index of buffer region
n_outer = n_clad                         # refractive index of outer PML region
n0 = n_clad                             # base refractive index for V function


# PML Parameters
alpha = 5                       # PML factor
pml_type = 'radial'
square_buffer = .25


# Mesh Parameters. Non-Dimensional

pml_maxh = .5 * r_fiber / scale
buffer_maxh = .3 * r_fiber / scale
tube_maxh = .07 * r_fiber / scale
clad_maxh = .2 * r_fiber / scale
poly_maxh = .8 * r_fiber / scale
core_maxh = .01 * r_fiber / scale


params = {

    'p': p,
    'layers': layers,
    'skip': skip,
    'pattern': pattern,
    'Λ': Λ,
    'r_tube': r_tube,
    'r_core': r_core,
    'r_fiber': r_fiber,
    'r_poly': r_poly,
    'r_pml': r_pml,
    'r_out': r_out,
    'scale': scale,

    'n_buffer': n_buffer,
    'n_tube': n_tube,
    'n_clad': n_clad,
    'n_poly': n_poly,
    'n_core': n_core,
    'n_outer': n_outer,
    'n0': n0,
    'wavelength': wavelength,

    't_buffer': t_buffer,
    't_poly': t_poly,
    't_outer': t_outer,
    'alpha': alpha,
    'pml_type': pml_type,
    'square_buffer': square_buffer,

    'pml_maxh': pml_maxh,
    'buffer_maxh': buffer_maxh,
    'tube_maxh': tube_maxh,
    'clad_maxh': clad_maxh,
    'poly_maxh': poly_maxh,
    'core_maxh': core_maxh,
}
