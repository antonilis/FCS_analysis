import numpy as np
import pandas as pd


def calculate_water_viscosity(T_K):
    A, B, C = 2.414e-5, 247.8, 140
    return A * 10 ** (B / (T_K - C))  # wynik w Pa·s

# Taken from the paper: https://doi.org/10.1016/j.jct.2010.03.022
def calculate_ethanol_viscosity(T_K):
    A, B, C = 6.12 * 10 ** (-7), 2961.8, 97.3
    return A * np.exp(B / (T_K + C))  # wynik w Pa·s


def calculate_Rh(D, T_C):
    kb = 1.380649e-23
    T_K = T_C + 273.15
    eta = calculate_water_viscosity(T_K)
    return kb * T_K / (6 * np.pi * eta * D * 1e-12) * 1e9


def calculate_probe_diffusion_coefficient(D1, T_C, solvent):
    T1, T2 = 298.15, T_C + 273.15
    if solvent == 'water':
        eta1 = calculate_water_viscosity(T1)
        eta2 = calculate_water_viscosity(T2)
    elif solvent == 'ethanol':
        eta1 = calculate_ethanol_viscosity(T1)
        eta2 = calculate_ethanol_viscosity(T2)
    else:
        raise ValueError('Available solvents are: water and ethanol')

    return D1 * T2 / eta2 * eta1 / T1 * 100


workspace_file_name = 'workspace.dct'


probes_diffusion_data = pd.DataFrame({
    'Fluorophore': [
        'Atto655-maleimid',
        'Atto655-carboxylic acid',
        'Atto655-NHS esther',
        'Cy5',
        'Alexa 647',
        'Alexa 633',
        'Rhodamine 6G',
        'Rhodamine B',
        'Rhodamine 123',
        'Rhodamine 110',
        'Fluorescein',
        'Oregon Green 488',
        'Atto488-carboxylic acid'
    ],
    'Emmision maximum [nm]': [
        686, 685, 685, 670, 665, 647, 550, 560, 530, 535, 520, 550, 523
    ],
    'Diffusion coefficient in water at 25°C (298.15 K) [10-6 cm2s-1]': [
        4.07, 4.26, 4.25, 3.60, 3.30, 3.40, 4.30, 4.27, 4.60, 4.70, 4.25, 4.11, 4.00
    ],
    'uncertainty': [
        0.10, 0.08, 0.06, 0.10, 0.10, 0.10, 0.40, 0.04, 0.40, 0.40, 0.01, 0.06, 0.10
    ]
})

T_ref = 25 + 273.15

viscosity_ratio = calculate_water_viscosity(T_ref) / calculate_ethanol_viscosity(T_ref)

probes_diffusion_data["Diffusion coefficient in EtOH at 25°C (298.15 K) [10-6 cm2s-1]"] = probes_diffusion_data[
                                                                                              "Diffusion coefficient in water at 25°C (298.15 K) [10-6 cm2s-1]"] * viscosity_ratio
