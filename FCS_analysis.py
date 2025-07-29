import os
import numpy as np
import pandas as pd
import json
import dill
import glob

from FCS_fitting_models import FCS_fitting_functions


class FCS_Analyzer:

    def __init__(self, calibration_probe, temperature=25, calibration_path=None, FCS_curves_path=None):
        ### constants

        self.calibration_path = calibration_path
        self.FCS_curves_path = FCS_curves_path

        self.kb = 1.380649 * 10 ** (-23)

        self.probes_diffusion_data = pd.read_csv(
            'C:/Users\Antoni Lis/Desktop/programy_python/FCS_analysis/picoQuant_data/diffusion_coefficients.csv')

        self.temperature = temperature

        self.calibration_probe = self.get_calibration_probe(calibration_probe)

        self.workspace_file_name = 'workspace.dct'

        self.calibration_results = self.get_calibration_data(calibration_path)

        self.fit_results = self.get_measurement_data(FCS_curves_path)

    @staticmethod
    def calculate_water_viscosity(T):
        A, B, C = 2.414 * 10 ** (-5), 247.8, 140

        return A * 10 ** (B / (T - C))  # Pa s

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
                print(f"Błąd przy pliku {path}:", e)

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            return df_all
        else:
            print("Nie znaleziono prawidłowych danych.")
            return None

    def calculate_Rh(self, diff_coefficient, Temperature):
        rho = self.calculate_water_viscosity(Temperature + 273.15)

        r = self.kb * (Temperature + 273.15) / (
                6 * np.pi * rho * diff_coefficient * 10 ** (-12)) * 10 ** 9  # D in um^2/s

        return r

    def calculate_probe_diffusion_coefficient(self, D1, Temperature):
        rho1 = self.calculate_water_viscosity(25 + 273.15)

        rho2 = self.calculate_water_viscosity(Temperature + 273.15)

        D2 = D1 * (Temperature + 273.15) / rho2 * rho1 / (25 + 273.25) * 100  # diffusion coefficient in um^2/s

        return D2

    def get_calibration_probe(self, probe):
        mask = self.probes_diffusion_data[self.probes_diffusion_data['Fluorophore'] == probe]

        D1 = mask['Diffusion coefficient in water at 25°C (298.15 K) [10-6 cm2s-1]'].iloc[0]

        D2 = self.calculate_probe_diffusion_coefficient(D1, self.temperature)
        Rh = self.calculate_Rh(D2, self.temperature)

        final_probe = {'probe': probe, 'Diffusion Coefficient': D2, 'Temperature': self.temperature,
                       'hydrodynamic radius': Rh}

        return final_probe

    def read_workspace_data(self, path):

        if path in None:
            return None

        final_path = os.path.join(path, self.workspace_file_name)

        with open(final_path, 'r') as f:
            data = json.load(f)

        return data

    def calculate_geometry(self, data_df):
        self.kappa = data_df['kappa'].mean()
        self.kappa_err = data_df['kappa_err'].mean()

        w0_values = 2 * np.sqrt(self.calibration_probe['Diffusion Coefficient'] * data_df['tau_d1'] / 1000)
        self.w0 = w0_values.mean()
        self.w0_err = w0_values.std()

        confocal_volume = np.pi ** (3 / 2) * w0_values ** 3 * data_df['kappa']
        self.confocal_volume = confocal_volume.mean()
        self.confocal_volume_err = confocal_volume.std()

    @staticmethod
    def get_fit_arguments(df):

        excluded = ['file', '_err', 'B', 'chi_sqr', 'Red.chi_sqr']
        chosen_columns = [col for col in df.columns if not any(ex in col for ex in excluded)]

        # Sprawdzenie czy są tylko liczby
        for col in chosen_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Kolumna '{col}' nie jest numeryczna i nie może być użyta do fitu.")

        args = [df[col].mean() for col in chosen_columns]
        return args

    def get_FCS_curve_with_fit(self, data, data_df, path):
        model = FCS_fitting_functions[data['Model']]
        calibration_curves = self.read_correlation_curves(path)
        merged = pd.merge(calibration_curves, data_df, on='file')

        arguments = self.get_fit_arguments(data_df)

        merged['G fit'] = model(calibration_curves['X'], *arguments)

        return merged
        #self.calibration_FCS_curves = merged

    def get_calibration_data(self, path):
        data = self.read_workspace_data(path)
        if data is None:
            return None

        #self.calibration_dictionairy_data = data

        data_df = pd.DataFrame(data['STORED RESULTS'])

        self.calculate_geometry(data_df)
        self.calibration_curves = self.get_FCS_curve_with_fit(data, data_df, self.calibration_path)

        return data_df

    # def get_calibration_data(self, path):
    #
    #     if path == None:
    #         return None
    #
    #     data = self.read_workspace_data(path)
    #
    #     self.calibration_dictionairy_data = data
    #
    #     data_df = pd.DataFrame(data['STORED RESULTS'])
    #
    #     self.kappa = data_df['kappa'].mean()
    #
    #     self.kappa_err = data_df['kappa_err'].mean()
    #
    #     w0 = 2 * np.sqrt(self.calibration_probe['Diffusion Coefficient'] * data_df['tau_d1'] / 1000)  # um
    #
    #     self.w0 = w0.mean()
    #     self.w0_err = w0.std()
    #
    #     confocal_volume = np.pi ** (3 / 2) * w0 ** 3 * data_df['kappa']
    #
    #     self.confocal_volume = confocal_volume.mean()
    #     self.confocal_volume_err = confocal_volume.std()
    #
    #     FCS_calibration_fitting_model = FCS_fitting_functions[data['Model']]
    #
    #     calibration_curves = self.read_correlation_curves(self.calibration_path)
    #
    #     self.calibration_FCS_curves = pd.merge(calibration_curves, data_df, on='file')
    #
    #     arguments = (self.calibration_FCS_curves['X'] ,self.calibration_FCS_curves['N_p'].mean(), self.calibration_FCS_curves['tau_d1'].mean(),
    #                  self.calibration_FCS_curves['kappa'].mean())
    #
    #     self.calibration_FCS_curves['G fit'] = FCS_calibration_fitting_model(*arguments)
    #
    #     return data_df

    def get_measurement_data(self, path):

        if path is None:
            return None

        data = self.read_workspace_data(path)

        self.fit_dictionairy_data = data

        self.FCS_measurement_fitting_model = FCS_fitting_functions[data['Model']]

        data_df = pd.DataFrame(data['STORED RESULTS'])

        data_df['D1'] = self.w0 ** 2 / (4 * data_df['tau_d1'] / 1000)

        data_df['conc'] = data_df['N_p'] / self.confocal_volume * 10 / 6.022

        data_df['Rh [nm]'] = self.calculate_Rh(data_df['D1'], self.temperature)

        self.mol_brightness = data_df['B'].mean()

        self.mol_brightness_err = data_df['B'].std()

        return data_df

    def save(self, path):
        with open(path, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load(path):

        with open(path, 'rb') as file:
            arr = dill.load(file)

        return arr


if __name__ == '__main__':
    workspace_file_name = 'workspace.dct'

    cal_path = './calibration'

    fcs_path = './FCS_curves'

    result = FCS_Analyzer('Rhodamine 110', 35.3, cal_path, fcs_path)

    #other_result = FCS_Analyzer('Alexa 647')
