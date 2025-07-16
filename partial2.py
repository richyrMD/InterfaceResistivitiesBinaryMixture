import pandas as pd
import numpy as np
from scipy.interpolate import griddata

#Datensatz einlesen
def load_data(filename):

    df = pd.read_csv(filename)

    df = df.groupby(['T*  [- ]', ' rho*  [ - ]'], as_index=False)[['h_1', 'h_2']].mean()

    df.columns = ['T', 'rho', 'h1', 'h2']

    points = np.column_stack((df['T'].values, df['rho'].values))

    return df, points
#Interpolieren, falls einzelnen Wert einzelnen Wert ausgeben, falls Wertebereich gegeben auch Wertebereich zurückgeben
def h(T, rho, method='nearest',
      filename="/home/richy/Schreibtisch/Single_State_Point_LJTS_LJTS_2_reduced_20250623_182813/Single_State_Point_LJTS_LJTS_2_reduced_results_extended.csv"):

    df, points = load_data(filename)

    T_arr = np.asarray(T).ravel()

    rho_arr = np.asarray(rho).ravel()

    query_points = np.column_stack((T_arr, rho_arr))

    h1_vals = griddata(points, df['h1'], query_points, method=method)

    h2_vals = griddata(points, df['h2'], query_points, method=method)

    if np.isscalar(T) or (np.ndim(T) == 0):

        return float(h1_vals), float(h2_vals)

    return h1_vals, h2_vals




