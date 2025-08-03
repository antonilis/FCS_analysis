import numpy as np
import pandas as pd
def calculate_water_viscosity(T_K):
    A, B, C = 2.414e-5, 247.8, 140
    return A * 10 ** (B / (T_K - C))

def calculate_Rh(D, T_C):
    kb = 1.380649e-23
    T_K = T_C + 273.15
    eta = calculate_water_viscosity(T_K)
    return kb * T_K / (6 * np.pi * eta * D * 1e-12) * 1e9

def calculate_probe_diffusion_coefficient(D1, T_C):
    T1, T2 = 298.15, T_C + 273.15
    eta1 = calculate_water_viscosity(T1)
    eta2 = calculate_water_viscosity(T2)
    return D1 * T2 / eta2 * eta1 / T1 * 100


workspace_file_name = 'workspace.dct'
probes_diffusion_data = pd.read_csv(
            'C:/Users\Antoni Lis/Desktop/programy_python/FCS_analysis/picoQuant_data/diffusion_coefficients.csv')