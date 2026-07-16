"""
Configuration ofr a NANF fiber, from paper
[TODO: add reference]
"""
SCALE = 15e-6
T_CLADDING = 10e-6
T_BUFFER = 15e-6
T_OUTER = 40e-6

# Maximum mesh size for each layer
GLASS_H = 0.15
AIR_H = 0.25
CORE_H = 0.25
BUFFER_H = 1.0
PML_H = 3.0

# Define dictionary for the fiber
params = {
    "scale": SCALE,
    "t_cladding": T_CLADDING,
    "t_buffer": T_BUFFER,
    "t_outer": T_OUTER,
    "glass_maxh": GLASS_H,
    "air_maxh": AIR_H,
    "core_maxh": CORE_H,
    "buffer_maxh": BUFFER_H,
    "pml_maxh": PML_H
}

if __name__ == "__main__":
    print(params)
    from fibermode.nanf import NANF
    nanf = NANF.from_dict(params)
    print(f'Test passed for {nanf}')
