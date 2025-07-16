import pandas as pd
import os
from tkinter import filedialog
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

# TimeStamp zur eindeutigen Dateierzeugung erstellen
def get_current_datetime():

    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Alle Unterordner eines gegebenen Ordners durchlaufen und alle Dateien mit .dat Dateiformat öffnen und enthaltenes
# dataframe der data_list anhängen
def import_data(file_directory):

    data_list = []

    file_paths = [os.path.join(file_directory, f) for f in os.listdir(file_directory) if f.endswith('.dat')]

    for file_path in file_paths:

        with open(file_path, 'r') as f:

            headers = f.readline().strip().split()

        data = np.loadtxt(file_path, skiprows=1)

        df = pd.DataFrame(data, columns=headers)

        data_list.append(df)

    return data_list, headers

# Die Binwidth herausfinden
def get_discretization(data):

    disc = data["pos"].values[2] - data["pos"].values[1]

    return disc

#Den Datensatz so bearbeiten dass die vorgegebene Diskretisierung vorliegt
def round_to_discretization(value, disc):

    rounded = round((value / disc)) * disc

    return rounded

#Logistische Funktion
def logistic_function(x, C1, C2, x0, tau):

    return C1 + (C2 - C1) / (1 + np.exp(-(x - x0) / tau))

#Funktion um in einem Datensatz den Wendepunkt herauszufinden
def find_inflection_point(data):
    #herausfiltern von nans
    valid_data = data[np.isfinite(data['pos']) & np.isfinite(data['rho[0]'])]

    x = valid_data['pos'].values#

    y = valid_data['rho[0]'].values

    try:
        #Fit mit logistischer Funktion
        popt, _ = curve_fit(logistic_function, x, y, maxfev=10000)

        x0 = popt[2]
        #auf gleiche Diskretisierung bringen
        x0_rounded = round_to_discretization(x0, get_discretization(valid_data))
        #Abrunden da Profile nicht bei null starten
        data["pos"] = np.floor(data["pos"])
        # Dichtewert von Profil bei berechnetem Wendepunkt zurückgeben
        if x0_rounded in data['pos'].values:

            y0 = data.loc[data["pos"] == x0_rounded, "rho[0]"].values

        else:

            y0 = np.array([np.nan])

        return x0_rounded, y0

    except Exception:

        return np.nan, np.nan

#Relevante Berechnungen durchführen und Spalten umbenennen
def calculate_mean_values(all_data_list, headers):

    relevant_cols = [col for col in all_data_list[0].columns if col in headers]

    data_stack = np.stack([df[relevant_cols].values for df in all_data_list])

    mean_data = np.mean(data_stack, axis=0)

    mean_dataframe = pd.DataFrame(mean_data, columns=headers)

    mean_dataframe["pos"] = all_data_list[0]["pos_shifted"].values

    for i in range(3):

        mean_dataframe[f'ekin[{i}]'] = 0.5 * mean_dataframe[f'v_y[{i}]'] ** 2

        mean_dataframe[f'h[{i}]'] = (

                mean_dataframe[f'epot[{i}]'] +
                mean_dataframe[f'p[{i}]'] / mean_dataframe[f'rho[{i}]'] -
                mean_dataframe[f'T[{i}]'] +
                ((2 + 3) / 2) * mean_dataframe[f'T[{i}]']

        )

    mean_dataframe["jp_ges"] = mean_dataframe["v_y[0]"] * mean_dataframe["rho[0]"]

    mean_dataframe["jp_1"] = mean_dataframe["v_y[1]"] * mean_dataframe["rho[1]"]

    mean_dataframe["jp_2"] = mean_dataframe["v_y[2]"] * mean_dataframe["rho[2]"]

    mean_dataframe["q_full"] = mean_dataframe["jEF_y[0]"] - mean_dataframe["jp_ges"] * (
            mean_dataframe["h[0]"] + mean_dataframe["ekin[0]"]
    )

    mean_dataframe["q_1"] = mean_dataframe["jEF_y[1]"] - mean_dataframe["jp_1"] * (
            mean_dataframe["h[1]"] + mean_dataframe["ekin[1]"]
    )

    mean_dataframe["q_2"] = mean_dataframe["jEF_y[2]"] - mean_dataframe["jp_2"] * (
            mean_dataframe["h[2]"] + mean_dataframe["ekin[2]"]
    )

    return mean_dataframe
#Funktion um die Profile auf den Wendepunkt zu normieren und auf die gleiche Länge zu bringen
def normalize_profiles(data_list):

    shifted_profiles = []
    #Für jedes Profil den Wendepunkt bestimmen und um die Position verschieben, sodass pos = 0 nun der Wendepunkt ist
    for idx, data in enumerate(data_list):

        x0_rounded, _ = find_inflection_point(data)

        if not np.isnan(x0_rounded):

            data = data.copy()

            data["pos_shifted"] = data["pos"] - x0_rounded

            shifted_profiles.append(data)

        else:

            print("e")
    #Den niedrigsten Wert aller Profile links und den höchsten Wert aller Profile rechts bestimmen
    min_left = max(profile["pos_shifted"].min() for profile in shifted_profiles)

    max_right = min(profile["pos_shifted"].max() for profile in shifted_profiles)
    #Profile auf das "kürzeste" Profil kürzen, sodass alle Profile die gleiche länge haben.
    for profile in shifted_profiles:

        pos_shifted = profile["pos_shifted"].values

        left_end = pos_shifted.min()

        right_end = pos_shifted.max()

        if abs(left_end - min_left) > 10:

            min_left = left_end

        if abs(right_end - max_right) > 10:

            max_right = right_end

    trimmed_profiles = []

    for profile in shifted_profiles:

        mask = (profile["pos_shifted"] >= min_left) & (profile["pos_shifted"] <= max_right)

        trimmed_profiles.append(profile.loc[mask].reset_index(drop=True))

    return trimmed_profiles

#Profil mit Zeitstempel als Namen exportieren
def export_data(file_directory, mean_dataframe):

    timestamp = get_current_datetime()

    export_file_path = os.path.join(file_directory, f"{timestamp}_means.csv")

    mean_dataframe.to_csv(export_file_path, index=False)


def main():

    file_directory = filedialog.askdirectory()

    data_list, headers = import_data(file_directory)

    for idx, data in enumerate(data_list):

        x0_rounded, y0 = find_inflection_point(data)

    data_list_normalized = normalize_profiles(data_list)

    mean_dataframe = calculate_mean_values(data_list_normalized, headers)

    export_data(file_directory, mean_dataframe)


main()
