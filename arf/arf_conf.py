"""
Configuration file for ARF, based on the paper of
[TODO: add reference]
"""
# Radii of the elements
CAPILLARY_H = 0.1
INNER_CORE_H = 0.25
GLASS_H = 0.25
AIR_H = 1.0
OUTER_H = 3.0

# Default kwargs for the fiber
NAME = None
FREE_CAPILLARY = False
OUTER_MATERIALS = None
CURVE = 8
REFINE = 0
EMBEDDING = None  # Called e

# updatable lengths are
# ['Rc', 'Rto', 'Rti', 't', 'd', 'tclad', 'touter', 'touterair']

# Define dictionary for the fiber
params = {
        'capillary_h': CAPILLARY_H,
        'inner_core_h': INNER_CORE_H,
        'glass_h': GLASS_H,
        'air_h': AIR_H,
        'outer_h': OUTER_H,
        'name': NAME,
        'freecapil': FREE_CAPILLARY,
        'outermaterials': OUTER_MATERIALS,
        'curve': CURVE,
        'refine': REFINE,
        'e': EMBEDDING
        }

if __name__ == "__main__":
    from fibermode.arf import ARF
    print(params)
    test_fiber = ARF.from_dict(params)
    print(f'Test passed for {test_fiber}')
