"""
Configuration of an Bragg fiber, based on paper by
[TODO: add reference]
"""
N_AIR = 1.00027717
N_GLASS = 1.4388164768221814
TS = [4.0775e-05, 1e-5, 1.5e-5, 5.5e-5]
NS = [lambda x: N_AIR, lambda x: N_GLASS, lambda x: N_AIR, lambda x: N_AIR]
MAX_HS = [.1, .1, 1.0, 3.0]
SCALE = 15e-6
WAVELENGHT = 1.6999999999999998e-06
BETA_BRAGG = 55.44793711650212 + 1.5343515969487227e-05j

# Define dictionary of parameters
params = {
    'scale': SCALE,
    'ts': TS,
    'ns': NS,
    'maxhs': MAX_HS,
    'wl': WAVELENGHT,
    'beta_bragg': BETA_BRAGG/SCALE
}

if __name__ == "__main__":
    from fibermode.bragg import Bragg
    print(params)
    test_fiber = Bragg.from_dict(params)
    print(f'Test passed for {test_fiber}')
