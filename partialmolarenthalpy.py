import pandas as pd
import numpy as np
import os
from tkinter import filedialog
#Datensatz einlesen und für die einzelnen Gruppen aus Datenpunkten rho/T jew. 10 Werte für x die Partiell molare Enthalpie berechnen 
def calc_partial_molar_enthalpy(file_path, output_dir):

    df = pd.read_csv(file_path)

    grouped = df.groupby(['T*  [- ]', ' rho*  [ - ]'])

    results = []

    for _, group in grouped:

        group = group.sort_values(" x_1  [ mol mol^-1 ]").copy()

        group['dHdx1'] = np.gradient(group[" h*  [ - ]"], group[" x_1  [ mol mol^-1 ]"])

        group['h_1'] = group[" h*  [ - ]"] + group[" x_2  [ mol mol^-1 ]"] * group['dHdx1']

        group['h_2'] = group[" h*  [ - ]"] - group[" x_1  [ mol mol^-1 ]"] * group['dHdx1']

        results.append(group)

    df_result = pd.concat(results)

    filename = os.path.basename(file_path)

    name1, name2 = os.path.splitext(filename)

    output_path = os.path.join(output_dir, f"{name1}_extended{name2}")

    df_result.to_csv(output_path, index=False, float_format="%.6f")

def main():

    input_dir = filedialog.askdirectory()

    output_dir = input_dir

    for file in os.listdir(input_dir):

        if file.endswith(".csv") and not file.endswith("_extended.csv"):

            full_path = os.path.join(input_dir, file)

            calc_partial_molar_enthalpy(full_path, output_dir)

main()




