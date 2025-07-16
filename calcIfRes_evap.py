import pandas as pd
from scipy.signal import savgol_filter
import os
import sys
pd.set_option('display.max_rows', 500)
sys.path.insert(0, '/home/richy/mydrive/ls1-mardyn/tools/ppls1/ppls1')
import fluids.ljts.ljts as ljts

def process_file(file_path, output_dir):

    df = pd.read_csv(file_path)

    df.set_index("pos",inplace=True)

    import_name = os.path.splitext(os.path.basename(file_path))[0]
    #T und mu Profile mit Savatzky Golay Filter glätten
    df["T_smooth"] = savgol_filter(df["T[0]"], window_length=8, polyorder=3)

    df["T_y_smooth"] = savgol_filter(df["T_y[0]"], window_length=8, polyorder=3)

    df["T_xz_smooth"] = savgol_filter((df["T_x[0]"] + df["T_z[0]"]) / 2, window_length=8, polyorder=3)

    df["mu_smooth"] = savgol_filter(df['chemPot_res[0]'], window_length=8, polyorder=3)

    df["mu_1_smooth"] = savgol_filter(df['chemPot_res[1]'], window_length=8, polyorder=3)

    df["mu_2_smooth"] = savgol_filter(df['chemPot_res[2]'], window_length=8, polyorder=3)

    df["T_xz"] = (df["T_x[0]"] + df["T_z[0]"]) / 2
    #Idealteil chem. Potential hinzufügen
    df['mu'] = ljts.g_ms22PeTS(df["mu_smooth"], df['T_smooth'], df['rho[0]'])

    df['mu_1'] = ljts.g_ms22PeTS(df["mu_1_smooth"], df['T_smooth'], df['rho[0]'])

    df['mu_2'] = ljts.g_ms22PeTS(df["mu_2_smooth"], df['T_smooth'], df['rho[0]'])
    #Ströme aus Werten im Dampf beziehen
    jp_ges = df["jp_ges"][150:250].mean()

    jp_1 = df["jp_1"][150:250].mean()

    jp_2 = df["jp_2"][150:250].mean()

    q = df["q_full"][150:250].mean()
    #Kriterien für Druck
    df['p_xz'] = 0.5 * (df['p_z[0]'] + df['p_x[0]'])

    df['dp_xzy'] = (df['p_xz'] - df['p_y[0]']) / df['p_y[0]']

    minIdx = df.loc[-15:20]['dp_xzy'].idxmin()

    maxIdx = 0

    dfLiq = df.loc[:minIdx]['dp_xzy']

    dfVap = df.loc[maxIdx:]['dp_xzy']

    x_p_liq = dfLiq[dfLiq > -0.1].index[-1]

    x_p_vap = dfVap[dfVap > 0.01].index[-1]
    #Kriterien für Temperatur
    df_all = pd.DataFrame({
        'Ty': df['T_y_smooth'],
        'Txz': df['T_xz_smooth'],
    })

    df_all['Spread'] = df_all.max(axis=1) - df_all.min(axis=1)

    x_T_liq = df_all["Spread"][-6:0].idxmax()

    x_T_vap = df_all["Spread"][5:20].idxmin()
    #Kriterien für Dichte
    rho_liq = df['rho[0]'][-15:20].max()

    rho_liq_idx = df['rho[0]'][-15:20].idxmax()

    rho_vap = df.loc[x_T_vap:x_T_vap + 10]['rho[0]'].mean()

    dfLiq = abs((df['rho[0]'][rho_liq_idx:] - rho_liq) / rho_liq)

    dfVap = abs((df['rho[0]'][rho_liq_idx:] - rho_vap) / rho_vap)

    x_rho_liq = dfLiq[dfLiq > 0.01].index[0]
    #Dichtekriterium Fehler abfangen
    try:

        x_rho_vap = dfVap[dfVap < 0.01].index[0]

    except Exception:

        x_rho_vap = x_p_vap
    #Benutzt die korrekten Kriterien, falls die Bestimmung der Temperaturkriterien nicht Fehlerhaft ist
    if x_T_vap < x_rho_vap or x_T_vap < x_p_vap:

            x_liq = min(x_rho_liq, x_p_liq)

            x_vap = max(x_rho_vap, x_p_vap)

    else:

            x_liq = min(x_rho_liq, x_p_liq)

            x_vap = x_T_vap
    #Resistivitäten berechnen
    mu_if_l = df.loc[x_liq, "mu"]

    mu_if_v = df.loc[x_vap, "mu"]

    T_if_l = df.loc[x_liq, "T[0]"]

    T_if_v = df.loc[x_vap, "T[0]"]

    T_if = df.loc[0,"T[0]"]

    h_if_v = df.loc[x_vap, "h[0]"]

    p_if_v = df.loc[x_vap, "p[0]"]

    X = ((mu_if_l / T_if_l) - (mu_if_v / T_if_v)) + h_if_v * ((1 / T_if_v) - (1 / T_if_l))

    Rges = X / jp_ges

    mu_1_if_l = df.loc[x_liq, "mu_1"]

    mu_1_if_v = df.loc[x_vap, "mu_1"]

    X = ((mu_1_if_l / T_if_l) - (mu_1_if_v / T_if_v)) + h_if_v * ((1 / T_if_v) - (1 / T_if_l))

    R1 = X / jp_ges

    mu_2_if_l = df.loc[x_liq, "mu_2"]

    mu_2_if_v = df.loc[x_vap, "mu_2"]

    X = ((mu_2_if_l / T_if_l) - (mu_2_if_v / T_if_v)) + h_if_v * ((1 / T_if_v) - (1 / T_if_l))

    R2 = X / jp_ges

    Rges_calc = R1 * (df.loc[x_vap, "rho[1]"] / df.loc[x_vap, "rho[0]"]) + \
                R2 * (df.loc[x_vap, "rho[2]"] / df.loc[x_vap, "rho[0]"])

    RQ = ((1 / T_if_v) - (1 / T_if_l)) / jp_ges

    dx = x_vap - x_liq
    #exportieren
    data = {
        "T_if": T_if,
        "Rges_calc": Rges_calc,
        "RQ": RQ,
        "Rges": Rges,
        "R1": R1,
        "R2": R2,
        "dx": dx,
        "jp": jp_ges,
        "jp1":jp_1,
        "jp2":jp_2,
        "q":q,
        "x_1_if":df.loc[x_vap, "rho[1]"] / df.loc[x_vap, "rho[0]"],
        "h_if_v": h_if_v,
        "p_if_v":df.loc[x_vap,"p[0]"],
        "rho_if_v":df.loc[x_vap, "rho[0]"],
        "rho_if":df.loc[0, "rho[0]"],
        "rho_if_l":df.loc[x_liq, "rho[0]"]
    }

    dfIF = pd.DataFrame([data])

    output_filename = f"{import_name}_output.csv"

    output_path = os.path.join(output_dir, output_filename)

    dfIF.to_csv(output_path, index=False)

def main():

    input_dir = "/home/richy/Schreibtisch/self_data/means_evap"

    output_dir = "/home/richy/Schreibtisch/self_data/ifres_evap"

    os.makedirs(output_dir, exist_ok=True)

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file in all_files:

        file_path = os.path.join(input_dir, file)

        try:

            process_file(file_path, output_dir)

        except Exception as e:

            print(e)

if __name__ == "__main__":

    main()
