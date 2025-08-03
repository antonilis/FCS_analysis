import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import dill
import glob
from FCS_fitting_models import FCS_fitting_functions
from utils import calculate_Rh, calculate_probe_diffusion_coefficient, workspace_file_name, probes_diffusion_data


class FCS_Analyzer:

    def __init__(self, calibration_probe, temperature=25, calibration_path=None, measurement_path=None):

        self.calibration_path = calibration_path
        self.measurement_path = measurement_path
        self.temperature = temperature

        self.calibration_probe = self.get_calibration_probe(calibration_probe)

        self.calibration_results = self.get_calibration_data(calibration_path)
        self.measurement_results = self.get_measurement_data(measurement_path)

    @staticmethod
    def read_correlation_curves(folder_path):

        # Znajdź wszystkie pliki .corr w folderze i podfolderach
        file_paths = glob.glob(os.path.join(folder_path, '**', '*.corr'), recursive=True)
        all_data = []

        for path in file_paths:
            try:
                data = pd.read_pickle(path)
                df = data['Correlation'].copy()
                df['file'] = os.path.basename(path)
                all_data.append(df)
            except Exception as e:
                print(f"Error with reading file: {path}:", e)

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            return df_all
        else:
            print("There is no data in those files")
            return None

    def get_calibration_probe(self, probe):
        mask = probes_diffusion_data[probes_diffusion_data['Fluorophore'] == probe]

        D1 = mask['Diffusion coefficient in water at 25°C (298.15 K) [10-6 cm2s-1]'].iloc[0]

        D2 = calculate_probe_diffusion_coefficient(D1, self.temperature)
        Rh = calculate_Rh(D2, self.temperature)

        final_probe = {'probe': probe, 'Diffusion Coefficient': D2, 'Temperature': self.temperature,
                       'hydrodynamic radius': Rh}

        return final_probe

    @staticmethod
    def read_workspace_data(path):

        if path is None:
            return None

        final_path = os.path.join(path, workspace_file_name)

        with open(final_path, 'r') as f:
            data = json.load(f)

        return data

    def calculate_geometry(self, data_df):
        w0_values = 2 * np.sqrt(
            self.calibration_probe['Diffusion Coefficient'] * data_df['tau_d1'] / 1000
        )

        confocal_volume = np.pi ** (3 / 2) * w0_values ** 3 * data_df['kappa']

        self.calibration_parameters = {
            'kappa': data_df['kappa'].mean(),
            'kappa_err': data_df['kappa_err'].mean(),
            'w0': w0_values.mean(),
            'w0_err': w0_values.std(),
            'confocal_volume': confocal_volume.mean(),
            'confocal_volume_err': confocal_volume.std()
        }

    @staticmethod
    def get_fit_arguments(df):

        excluded = ['file', '_err', 'B', 'chi_sqr', 'Red.chi_sqr', 'X', 'Y']
        chosen_columns = [col for col in df.columns if not any(ex in col for ex in excluded)]

        args = tuple(np.array(df[col], dtype=np.float64) for col in chosen_columns)
        return args

    def get_FCS_curve_with_fit(self, data_dic, data_df, path):
        model = FCS_fitting_functions[data_dic['Model']]
        calibration_curves = self.read_correlation_curves(path)
        merged = pd.merge(calibration_curves, data_df, on='file')

        arguments = self.get_fit_arguments(merged)

        merged['G fit'] = model(merged['X'], *arguments)

        return merged

    def get_calibration_data(self, path):
        data = self.read_workspace_data(path)
        if data is None:
            return None

        data_df = pd.DataFrame(data['STORED RESULTS'])

        self.calculate_geometry(data_df)
        self.calibration_FCS_curves = self.get_FCS_curve_with_fit(data, data_df, self.calibration_path)

        return data_df

    def calculate_measurement_parameters(self, data_df):
        w0 = self.calibration_parameters['w0']
        confocal_volume = self.calibration_parameters['confocal_volume']

        data_df['D1'] = w0 ** 2 / (4 * data_df['tau_d1'] / 1000)
        data_df['conc'] = data_df['N_p'] / confocal_volume * 10 / 6.022
        data_df['Rh [nm]'] = calculate_Rh(data_df['D1'], self.temperature)

        return data_df

    def get_measurement_data(self, path):

        if path is None:
            return None

        data_dic = self.read_workspace_data(path)

        data_df = pd.DataFrame(data_dic['STORED RESULTS'])

        self.measurement_FCS_curves = self.get_FCS_curve_with_fit(data_dic, data_df, path)

        data_df = self.calculate_measurement_parameters(data_df)

        return data_df

    @staticmethod
    def plot_FCS_curves(df):

        unique_files = df['file'].unique()

        for file in unique_files:
            df_file = df[df['file'] == file]

            tau = df_file['X']
            G = df_file['Y']
            G_err = df_file['Y_err']
            G_fit = df_file['G fit']

            err = plt.errorbar(tau, G, yerr=G_err, fmt='o', markersize=0.2, capsize=0.1, elinewidth=0.2,
                               markeredgewidth=0.5)
            plt.plot(tau, G_fit, '-', linewidth=0.2, color=err[0].get_color())

        plt.xscale('log')
        plt.xlabel(r'lag time $\tau$ [ms]')
        plt.ylabel(r'G($\tau$)')

    def save(self, path):
        with open(path, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load(path):

        with open(path, 'rb') as file:
            arr = dill.load(file)

        return arr


if __name__ == '__main__':
    cal_path = './calibration'

    fcs_path = './FCS_curves'

    result = FCS_Analyzer('Rhodamine 110', 35.3, cal_path, fcs_path)

    other_result = FCS_Analyzer('Alexa 647')
