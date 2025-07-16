import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import glob
from partial2 import h
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy import integrate
import warnings
warnings.filterwarnings("ignore")

import ppls1.pp.pp_extprfsmpl as pp_extprfsmpl
import ppls1.pp.pp_readGamma as pp_readGamma

import ppls1.pp.pp_calcIFRes as pp_calcIFRes

import ppls1.fluids.ljts.ljts as ljts
import ppls1.fluids.ljts.petspy.petspy as petspy
#from ppls1.fluids.ljts.petspy import petseos
#from ppls1.fluids.therm_cond import lambda_lauten, lambda_lemmon
import inspect
import math  # Beispiel-Modul

funktionen = [name for name, obj in inspect.getmembers(petspy, inspect.isfunction)]

### Calculate average MD profiles, plot and save

def mirrorShiftProfiles(dfOne):
    times = dfOne['timestep'].unique()
    newDFList = list()
    for t in times:
        
        ## Mirror half of data
        
        dfT = dfOne[dfOne['timestep'] == t]
        lenDF = int(len(dfT)/2)
        binwidth = dfT['pos'].diff().iloc[2]  # first one could be NaN, therefore take second or third
        posCenter = lenDF*binwidth
        dfLeft = dfT[:lenDF]
        dfRight = dfT[lenDF:]
        dfRight.loc[:,'pos'] -= posCenter
        dfLeft = dfLeft.iloc[::-1]
        if abs(len(dfLeft)-len(dfRight)) > 10: print('WARNING: Lengths do not match!')
        # Truncate to match size of dfs
        if len(dfLeft) > len(dfRight):
            dfLeft = dfLeft.iloc[:len(dfRight)]
        elif len(dfLeft) < len(dfRight):
            dfRight = dfRight.iloc[:len(dfLeft)]
        dfLeft.loc[:,'pos'] = dfRight['pos'].to_list()
        dfLeft.reset_index(drop=True, inplace=True)
        dfRight.reset_index(drop=True, inplace=True)
        dfLeft.loc[:,'pos'] += abs(dfLeft['pos'].min())
        dfRight.loc[:,'pos'] += abs(dfRight['pos'].min())
        
        # Mirror process quantities
        dfLeft.loc[:,'v_x'] = (-dfLeft['v_x']).to_list()
        dfLeft.loc[:,'v_y'] = (-dfLeft['v_y']).to_list()
        dfLeft.loc[:,'v_z'] = (-dfLeft['v_z']).to_list()
        dfLeft.loc[:,'F_x'] = (-dfLeft['F_x']).to_list()
        dfLeft.loc[:,'F_y'] = (-dfLeft['F_y']).to_list()
        dfLeft.loc[:,'F_z'] = (-dfLeft['F_z']).to_list()
        dfLeft.loc[:,'je']  = (-dfLeft['je']).to_list()
        dfLeft.loc[:,'jp']  = (-dfLeft['jp']).to_list()
        dfLeft.loc[:,'q']  = (-dfLeft['q']).to_list()
        
        ## Shift
        inflectionIdxLeft = dfLeft[dfLeft['pos'].between(10, 100)]['rho'].diff().idxmin()
        inflectionIdxRight = dfRight[dfRight['pos'].between(10, 100)]['rho'].diff().idxmin()
        # print(t,dfLeft.iloc[inflectionIdxLeft]['pos'],dfRight.iloc[inflectionIdxRight]['pos'])
        
        dfLeft.loc[:,'pos'] -= dfLeft.iloc[inflectionIdxLeft]['pos']
        dfRight.loc[:,'pos'] -= dfRight.iloc[inflectionIdxRight]['pos']
        
        dfT2 = pd.concat([dfLeft,dfRight],axis=0,ignore_index=True).groupby('pos').mean()
        
        dfT2.loc[:,'pos'] = dfT2.index.values
        
        newDFList.append(dfT2)
        
    dfOne = pd.concat(newDFList,axis=0,ignore_index=True)
    
    return dfOne


plt.close('all')

# Import profile data
print("Hallo")
folderPathdT = '/home/richy/mydrive/data/data_09'  # Import profile data
sim_dirs = sorted(glob.glob(folderPathdT+'/T*_dT*_x1_*'))
print(sim_dirs)
path2IFData = "/home/richy/Schreibtisch/export_0009"
# Export data
path2IFData_dT_data = path2IFData+'/tempGrad'  # Export path for profile data
path2IFData_dT_figures = path2IFData+'/tempGrad_figures'  # Export path for figures

for filePath in sim_dirs:
    simName = filePath.split('/')[-1]
    print(simName)
    print(filePath)
    try:
    
        dfAll = pp_extprfsmpl.eps2df(filePath, quietMode=True, flgInterpolate=True)
        
        #dfGamma = pp_readGamma.gamma2df(filePath+'/run0*', quietMode=True)
        cid = 0
        dfOne = dfAll[dfAll['cid']==cid]
        dfOne.rename(columns={'jEF_y': 'je'}, inplace=True)
        print(simName)
        print(filePath)
        with pd.option_context('mode.chained_assignment', None):
            dfOne['jp'] = dfOne['v_y']*dfOne['rho']
            dfOne['ekin'] = 0.5*dfOne['v_y']*dfOne['v_y']  # Overrides the EPS ekin (total ekin) with the transl. ekin
            dfOne['h'] = dfOne['epot'] + dfOne['p']/dfOne['rho'] - dfOne['T'] + ((2+3)/2)*dfOne['T']
            dfOne['mu'] = dfOne['chemPot_res']
            if cid in [1, 2]:
                df_T_cid0 = dfAll[dfAll['cid'] == 0][['pos', 'T']]
                df_T_cid0 = df_T_cid0.groupby('pos').mean().reset_index()
                dfOne = dfOne.merge(df_T_cid0, on='pos', suffixes=('', '_cid0'))
                df_p_cid0 = dfAll[dfAll['cid'] == 0][['pos', 'p']]
                df_p_cid0 = df_p_cid0.groupby('pos').mean().reset_index()
                dfOne = dfOne.merge(df_p_cid0, on='pos', suffixes=('', '_cid0'))
                df_T_cid0_y = dfAll[dfAll['cid'] == 0][['pos', 'T_y']]
                df_T_cid0_y = df_T_cid0_y.groupby('pos').mean().reset_index()
                dfOne = dfOne.merge(df_T_cid0_y, on='pos', suffixes=('', '_cid0'))
                df_rho_cid0 = dfAll[dfAll['cid'] == 0][['pos', 'rho']]
                df_rho_cid0 = df_rho_cid0.groupby('pos').mean().reset_index()
                dfOne = dfOne.merge(df_rho_cid0, on='pos', suffixes=('', '_cid0'))
                dfOne['mu'] = ljts.g_ms22PeTS(dfOne['chemPot_res'], dfOne['T_cid0'], dfOne['rho'])
            else:
                dfOne['mu'] = ljts.g_ms22PeTS(dfOne['chemPot_res'], dfOne['T'], dfOne['rho'])
                dfOne['T_cid0'] =dfOne['T']
                dfOne['T_y_cid0'] =dfOne['T_y']
                dfOne["rho_cid0"] = dfOne["rho"]
                dfOne["p_cid0"] = dfOne["p"]
                print("nope")
            dfOne['q'] = dfOne['je'] - dfOne['jp']*(dfOne['h']+dfOne['ekin'])
            dfOne['T_xz'] = 0.5*(dfOne['T_x']+dfOne['T_z'])
        
        dfOne = mirrorShiftProfiles(dfOne)
        print(f'Number of samples: {len(dfOne["timestep"].unique())}')
        if dfOne['timestep'].max() >= 5000000:
            timestepMean = 5000000
        else:
            timestepMean = dfOne['timestep'].unique()[-20]
            print(f'Warning! Setting timestepMean to {timestepMean}')
        df = dfOne[dfOne['timestep'] >= timestepMean].groupby('pos').mean()
        df.drop(columns=['cid','timestep'],inplace=True)
        
        jp_const = round(df['jp'][150:200].mean(), 8)
        je_const = round(df['je'][150:200].mean(), 8)
        p_const  = round(df['p'][150:200].mean(), 8)
        
        
        # Smooth profiles
        df_raw = df.copy()
        dfTemp = pd.DataFrame()
        
        
        binwidth_fit = 0.1
        Ln=10
        Lv=50
        # Position of bin which density is closest to 99% of liquid density
        pos_x_if_l = df.iloc[(df['rho']-0.994*df['rho'][-12:-10].mean()).abs().argsort()[:1]].index[0]
        # Position of bin which density is closest to 101% of vapor density
        pos_x_if_v = df.iloc[(df['rho']-1.014*df['rho'][10:15].mean()).abs().argsort()[:1]].index[0]
        x_if_l = df.index[(np.abs(df.index - pos_x_if_l)).argmin()]
        x_if_v = df.index[(np.abs(df.index - pos_x_if_v)).argmin()]
        # Position of right boundary of interface
        x_bulk_l = x_if_l-Ln
        x_bulk_v = x_if_v+Lv
        
        x_fit_all = np.round(np.arange(x_bulk_l, x_bulk_v, binwidth_fit),3)
        dfTemp['pos'] = x_fit_all

        for col in ['T', 'T_xz', 'T_y','T_cid0','T_y_cid0', 'rho','rho_cid0', 'epot', 'mu', 'v_y', 'p','p_cid0', 'p_x', 'p_y', 'p_z']:
            # print('Smoothing:',col)
            
            # For Temperature use tanh fit
            if col in ['T', 'T_xz', 'T_y','T_cid0','T_y_cid0']:
                # print(f'Temp: {col}')
                def fit_func(x, a, b, c, d):
                    return a*np.tanh(b*x+c)+d

                """def fit_func(x, a, b, c, d):
                    return a * erf(b*x + c) + d"""
                x_fit_min = -3
                x_fit_max = 8
                # Fits
                # Interface
                params = curve_fit(fit_func, df_raw[x_fit_min:x_fit_max].index, df_raw[col][x_fit_min:x_fit_max])
                x_fit_if = np.round(np.arange(x_fit_min, x_fit_max, binwidth_fit),3)
                a = params[0][0]
                b = params[0][1]
                c = params[0][2]
                d = params[0][3]
                T_fit_if = fit_func(x_fit_if, a, b, c, d)
                # e = params[0][4]
                ### Customize ###
                # a = 0.15
                # b = 0.0
                # c = 0.15
                # d = 0.0
                #################
                # print(f'Fit parameters: \n   a = {a}\n   b = {b}\n   c = {c}\n   d = {d}')
                
                grad_liq = np.diff(np.poly1d(np.polyfit(df_raw[x_bulk_l:x_if_l].index, df_raw[x_bulk_l:x_if_l][col], 1))([0,1]))[0]
                grad_vap = np.diff(np.poly1d(np.polyfit(df_raw[x_if_v:x_bulk_v].index, df_raw[x_if_v:x_bulk_v][col], 1))([0,1]))[0]
                
                # Liquid
                try:
                    x_fit_min = round(x_fit_if[:-1][np.diff(T_fit_if)/binwidth_fit >= grad_liq][1],3)
                except:
                    x_fit_min = min(x_fit_if)
                x_fit_liq = np.round(np.arange(x_bulk_l, x_fit_min, binwidth_fit),3)
                T_fit_liq = np.poly1d(np.polyfit(df_raw[x_bulk_l:x_if_l].index, df_raw[x_bulk_l:x_if_l][col], 1))(x_fit_liq)
                # Vapor
                try:
                    x_fit_max = round(x_fit_if[:-1][np.diff(T_fit_if)/binwidth_fit >= grad_vap][-1],3)
                except:
                    x_fit_max = max(x_fit_if)
                x_fit_vap = np.round(np.arange(x_fit_max, x_bulk_v, binwidth_fit),3)
                T_fit_vap = np.poly1d(np.polyfit(df_raw[x_if_v:x_bulk_v].index, df_raw[x_if_v:x_bulk_v][col], 1))(x_fit_vap)
                
                T_fit_if = T_fit_if[(x_fit_if>max(x_fit_liq)) & (x_fit_if<min(x_fit_vap))]
                x_fit_if = x_fit_if[(x_fit_if>max(x_fit_liq)) & (x_fit_if<min(x_fit_vap))]
                
                x_fit = np.concatenate((x_fit_liq, x_fit_if, x_fit_vap), axis=None)
                T_fit = np.concatenate((T_fit_liq+(T_fit_if[0]-T_fit_liq[-1]-np.diff(T_fit_liq)[-1]),
                                        T_fit_if,
                                        T_fit_vap+(T_fit_if[-1]-T_fit_vap[0]+np.diff(T_fit_vap)[0])), axis=None)
                #plt.plot(x_fit,T_fit)
                #plt.show()
                # Use cubic spline for further smoothing
                #cs = CubicSpline(x_fit[::5], T_fit[::5])
                #T_fit = cs(x_fit_all)
                
                # Check if length is ok
                if (((max(x_fit_all)-min(x_fit_all))/binwidth_fit) - (len(x_fit_all)-1)) > 1e-6:
                    print(x_fit_all,((max(x_fit_all)-min(x_fit_all))/binwidth_fit),(len(x_fit_all)-1))
                    print('Warning! Something is wrong with x_fit')
                    break
                
                # plt.figure()
                # plt.xlim([x_bulk_l,x_bulk_v])
                # plt.title(col)
                # plt.plot(df_raw.index, df_raw['T'], '-', color='red', label='Raw')
                # plt.plot(x_fit, T_fit, '-', color='orange')
                # # plt.plot(x_fit_if, T_fit_if, '-', color='blue', label='Fit')
                # # plt.plot(x_fit_liq, T_fit_liq, '-', color='green', label='Liq')
                # # plt.plot(x_fit_vap, T_fit_vap, '-', color='green', label='Vap')
                # plt.legend()
                
                dfTemp[col] = T_fit
                if 'pos' not in dfTemp.columns:
                    dfTemp['pos'] = x_fit_all
                
            # For other quantities use cubic spline interpolation
            else:
                # print(f'Other: {col}')
                cs = CubicSpline(df_raw.index, df_raw[col])
                dfTemp[col] = cs(x_fit_all)
                
                """plt.figure()
                plt.xlim([x_bulk_l,x_bulk_v])
                plt.title(col)
                plt.plot(df_raw.index, df_raw[col], '-', color='red', label='Raw')
                plt.plot(dfTemp['pos'], dfTemp[col], '-', color='orange', label='Fit')
                plt.legend()
                plt.show()
                input()"""
        
        df = dfTemp.copy()
        df.loc[:,'pos'] = df['pos'].round(3)
        df.set_index('pos', inplace=True, drop=True)
        #df.index -= df.iloc[df.index.get_loc(0.0, method='nearest')].name
        target = 0.0
        nearest_index = df.index[np.abs(df.index - target).argmin()]

        # Index verschieben
        df.index = df.index - nearest_index
        df.index = df.index.to_series().round(5)
        
        df['p'] = p_const
        
        df['ekin'] = 0.5*df['v_y']*df['v_y']
        df['h'] = df['epot'] + df['p']/df['rho'] - df['T'] + ((2+3)/2)*df['T']
        
        # Keep T_raw but fit index to df.index
        cs = CubicSpline(df_raw.index, df_raw['T'])
        df['T_raw'] = cs(df.index.to_numpy())
            
        df['jp'] = jp_const
        df['je'] = je_const
        df['q'] = (je_const - jp_const*(df['h']+df['ekin'])).mean()

        #h1_vals, h2_vals = h(df["T_cid0"], df["p_cid0"])

        #df["h1"] = h1_vals 
        #df["h2"] = h2_vals
        #df["h1"] = df["h1"] + (5/2)*df["T_cid0"]
        #df["h2"] = df["h2"] + (5/2)*df["T_cid0"]
        # Chemical potential with PeTS
        df['mu_pets'] = np.nan
        for i in df.index:
            df.loc[i,'mu_pets'] = petspy.petseos(12,df.loc[i,'rho'],19,df.loc[i,'T'],51)
        # Chemical potential with thermodynamic relation/equation, calculated backwards
        #component1
        if cid == 0: 
            df['mu_eq'] = np.nan
            df.loc[df.index[-1],'mu_eq'] = df.loc[df.index[-1],'mu']
            diffIdx = round(df.index[1]-df.index[0],8)
            for i in df.index[-1:0:-1]:
                idxLast = round(i-diffIdx,8)
                # -2 due to reference point of PeTS
                df.loc[idxLast,'mu_eq'] = df.loc[idxLast,'T']*((df.loc[idxLast,'h'])*((1/df.loc[idxLast,'T'])-(1/df.loc[i,'T']))+(df.loc[i,'mu_eq']/df.loc[i,'T']))
                # df.loc[i,'mu_eq'] = df.loc[i,'T']*(df.loc[i,'h']*((1/df.loc[i,'T'])-(1/df.loc[idxLast,'T']))+\
                                                # (df.loc[idxLast,'mu_eq']/df.loc[idxLast,'T'])+\
                                                # ((1/(df.loc[i,'rho']*df.loc[i,'T']))*((df.loc[i,'p'])-(df.loc[idxLast,'p']))))
        if cid == 1: 
            df['mu_eq'] = np.nan
            df.loc[df.index[-1],'mu_eq'] = df.loc[df.index[-1],'mu']
            diffIdx = round(df.index[1]-df.index[0],8)
            for i in df.index[-1:0:-1]:
                idxLast = round(i-diffIdx,8)
                # -2 due to reference point of PeTS
                df.loc[idxLast,'mu_eq'] = df.loc[idxLast,'T_cid0']*((df.loc[idxLast,'h'])*((1/df.loc[idxLast,'T_cid0'])-(1/df.loc[i,'T_cid0']))+(df.loc[i,'mu_eq']/df.loc[i,'T_cid0']))
                # df.loc[i,'mu_eq'] = df.loc[i,'T']*(df.loc[i,'h']*((1/df.loc[i,'T'])-(1/df.loc[idxLast,'T']))+\
                                                # (df.loc[idxLast,'mu_eq']/df.loc[idxLast,'T'])+\
                                                # ((1/(df.loc[i,'rho']*df.loc[i,'T']))*((df.loc[i,'p'])-(df.loc[idxLast,'p']))))                                

        if cid == 2: 
            df['mu_eq'] = np.nan
            df.loc[df.index[-1],'mu_eq'] = df.loc[df.index[-1],'mu']
            diffIdx = round(df.index[1]-df.index[0],8)
            for i in df.index[-1:0:-1]:
                idxLast = round(i-diffIdx,8)
                # -2 due to reference point of PeTS
                df.loc[idxLast,'mu_eq'] = df.loc[idxLast,'T_cid0']*((df.loc[idxLast,'h'])*((1/df.loc[idxLast,'T_cid0'])-(1/df.loc[i,'T_cid0']))+(df.loc[i,'mu_eq']/df.loc[i,'T_cid0']))
                # df.loc[i,'mu_eq'] = df.loc[i,'T']*(df.loc[i,'h']*((1/df.loc[i,'T'])-(1/df.loc[idxLast,'T']))+\
                                                # (df.loc[idxLast,'mu_eq']/df.loc[idxLast,'T'])+\
                                                # ((1/(df.loc[i,'rho']*df.loc[i,'T']))*((df.loc[i,'p'])-(df.loc[idxLast,'p']))))
        
        # df['mu_eqT'] = np.nan
        # df.loc[df.index[300],'mu_eqT'] = df.iloc[300]['mu_pets']/df.iloc[300]['T']
        # diffIdx = round(df.index[1]-df.index[0],8)
        # for i in df.index[301:]:
        #     idxLast = round(i-diffIdx,8)
        #     df.loc[i,'mu_eqT'] = -(1/6)*df.loc[i,'h']*((1/df.loc[i,'T'])-(1/df.loc[idxLast,'T']))+df.loc[idxLast,'mu_eqT']
        
        
        #df['gamma'] = dfGamma[dfGamma['timestep'] >= timestepMean]['gamma[0]'].mean()
        df['gamma'] = None
        
        
        dfIFPos = pp_calcIFRes.if_positions_tempGrad(df)
        
        """df_mueq.to_csv(f'{path2IFData_dT_data}/{simName}_mueq_data.csv', float_format='{:.10e}'.format, na_rep='NaN')
        print(df_mueq)
        #%% Export dataframes
        input()"""
        df.to_csv(f'{path2IFData_dT_data}/{simName}_MD_profile_data.csv', float_format='{:.10e}'.format, na_rep='NaN')
        df_raw.to_csv(f'{path2IFData_dT_data}/{simName}_MD_profile_data_raw.csv', float_format='{:.10e}'.format, na_rep='NaN')
        
        
        #%% Plotting
        
        leftZ = -15
        rightZ = 25
        
        fig = plt.figure(figsize=(12, 8))
        plt.suptitle(simName)
        
        ax1 = plt.subplot(331)
        plt.axvline(x=dfIFPos['x_rho_liq'], color='grey', linestyle='--')
        plt.axvline(x=dfIFPos['x_rho_vap'], color='grey', linestyle='--')
        plt.axvline(x=dfIFPos['x_liq'], color='grey', linestyle='-')
        plt.axvline(x=dfIFPos['x_vap'], color='grey', linestyle='-')
        df['rho'][leftZ:rightZ].plot()
        ax1.set_title('Density')
        # ax1.set_yscale('log')
        plt.xlabel('z')
        plt.ylabel('rho')
        plt.xlim([leftZ,rightZ])
        plt.grid()
        
        ax2 = plt.subplot(332)
        (1000*df['jp'][leftZ:rightZ]).plot()
        (1000*df_raw['jp'][leftZ:rightZ]).plot()
        ax2.set_title('Massflux')
        plt.xlabel('z')
        plt.ylabel('jp *1e3')
        plt.xlim([leftZ,rightZ])
        # plt.ylim([1.5,3.5])
        plt.grid()
        
        ax3 = plt.subplot(333)
        plt.axvline(x=dfIFPos['x_T_liq'], color='grey', linestyle='--', label='_nolegend_')
        plt.axvline(x=dfIFPos['x_T_vap'], color='grey', linestyle='--', label='_nolegend_')
        plt.axvline(x=dfIFPos['x_liq'], color='grey', linestyle='-', label='_nolegend_')
        plt.axvline(x=dfIFPos['x_vap'], color='grey', linestyle='-', label='_nolegend_')
        df['T_xz'][leftZ:rightZ].plot()
        df['T_y'][leftZ:rightZ].plot()
        df['T'][leftZ:rightZ].plot()
        df_raw['T'][leftZ:rightZ].plot()
        ax3.set_title('Temperature')
        plt.xlabel('z')
        plt.ylabel('T')
        plt.xlim([leftZ,rightZ])
        plt.grid()
        plt.legend(['T_xz','T_y','T','T_raw'], ncol=3)
        
        ax4 = plt.subplot(334)
        (df['v_y'][leftZ:rightZ]).plot()
        ax4.set_title('Velocity')
        plt.xlabel('z')
        plt.ylabel('v_y')
        plt.xlim([leftZ,rightZ])
        # plt.ylim([0.0,30.0])
        plt.grid()
        
        ax5 = plt.subplot(335)
        (1000*df['je'][leftZ:rightZ]).plot()
        (1000*df_raw['je'][leftZ:rightZ]).plot()
        ax5.set_title('Energy flux')
        plt.xlabel('z')
        plt.ylabel('je *1e3')
        plt.xlim([leftZ,rightZ])
        plt.ylim([0.1*ax5.get_ylim()[0], 0.0])
        plt.grid()
        
        ax6 = plt.subplot(336)
        (1000*df['q'][leftZ:rightZ]).plot()
        (1000*df_raw['q'][leftZ:rightZ]).plot()
        ax6.set_title('Heat flux')
        plt.xlabel('z')
        plt.ylabel('q *1e3')
        plt.xlim([leftZ,rightZ])
        plt.ylim([0.1*ax6.get_ylim()[0], 0.0])
        plt.grid()
        
        ax7 = plt.subplot(337)
        plt.axvline(x=dfIFPos['x_p_liq'], color='grey', linestyle='--')
        plt.axvline(x=dfIFPos['x_p_vap'], color='grey', linestyle='--')
        plt.axvline(x=dfIFPos['x_liq'], color='grey', linestyle='-')
        plt.axvline(x=dfIFPos['x_vap'], color='grey', linestyle='-')
        (1000*df['p'][leftZ:rightZ]).plot()
        (1000*df_raw['p'][leftZ:rightZ]).plot()
        ax7.set_title('Pressure')
        plt.xlabel('z')
        plt.ylabel('p *1e3')
        plt.xlim([leftZ,rightZ])
        # plt.ylim([6.5,8.5])
        ax7.set_ylim(bottom=0.0)
        plt.grid()
        
        ax8 = plt.subplot(338)
        df['h'][leftZ:rightZ].plot()
        df_raw['h'][leftZ:rightZ].plot()
        ax8.set_title('Enthalpy')
        plt.xlabel('z')
        plt.ylabel('h')
        plt.xlim([leftZ,rightZ])
        plt.grid()
        
        ax9 = plt.subplot(339)
        print(df["mu_eq"])
        print(df["mu_pets"])
        df['mu_eq'][leftZ:rightZ].plot(color='tab:green')
        df['mu'][leftZ:rightZ].plot(color='tab:blue')
        ylimits = ax9.get_ylim()
        #df['mu_pets'][leftZ:rightZ].plot(color='tab:orange')
        #df['mu'][leftZ:rightZ].plot(color='tab:blue')
        plt.axvline(x=dfIFPos['x_T_liq'], color='grey', linestyle='--')
        plt.axvline(x=dfIFPos['x_T_vap'], color='grey', linestyle='--')
        plt.axvline(x=dfIFPos['x_liq'], color='grey', linestyle='-')
        plt.axvline(x=dfIFPos['x_vap'], color='grey', linestyle='-')
        ax9.set_title('Chemical potential')
        plt.xlabel('z')
        plt.ylabel('mu')
        plt.xlim([leftZ,rightZ])
        plt.ylim([0.97*ylimits[0],1.03*ylimits[1]])
        plt.grid()
        
        plt.tight_layout()
        
        fig.savefig(f'{path2IFData_dT_figures}/{simName}_MD_profile_plot.pdf', format='pdf')
        
        plt.close()
        
        
        # Plot time series
        
        fig = plt.figure()
        plt.suptitle(simName)
        
        quant = 'je'
        
        colors = plt.cm.jet(np.linspace(0,1,len(dfOne['timestep'].unique())))
        idx=0
        for i in dfOne['timestep'].unique():
            plt.plot(dfOne[dfOne['timestep']==i]['pos'],dfOne[dfOne['timestep']==i][quant],label=str(i), color=colors[idx])
            idx += 1
        
        plt.plot(df.index, df[quant], color='black')
        # plt.ylim((0.2*df[quant].mean(),1.5*df[quant].mean()))
        plt.xlabel('z')
        plt.ylabel(quant)
        plt.grid()
        # plt.legend(ncol=4)
        plt.tight_layout()
        
        fig.savefig(f'{path2IFData_dT_figures}/{simName}_MD_timeseries_plot.pdf', format='pdf')
        
        plt.close()
        
        
        #########################################
        
        # plt.figure()
        # ### Check validity Eq. (6)
        # df['dgT'] = (df['mu']/df['T']).diff()/df.index.to_series().diff()
        # df['hdT_inv'] = df['dT_inv']*df['h']
        # df['dgT'].plot()
        # df['hdT_inv'].plot()
        # plt.grid()
        # plt.legend(['dgT','hdT_inv'])
        
        # ### Check validity Eq. (6) RAW
        # df['dgT_raw'] = (df['g_raw']/df['T_raw']).diff()/df.index.to_series().diff()
        # df['hdT_inv_raw'] = df['dT_inv_raw']*df['h_raw']
        # df['dgT_raw'].plot()
        # df['hdT_inv_raw'].plot()
        # plt.grid()
        # plt.legend(['dgT_raw','hdT_inv_raw'])
        
        # plt.figure()
        # plt.grid()
        # n = len(dfOne['timestep'].unique())
        # colors = plt.cm.jet(np.linspace(0,1,n))
        # i = 0
        # for t in dfOne['timestep'].unique():
        #     plt.plot(dfOne[dfOne['timestep'] == t]['pos'],dfOne[dfOne['timestep'] == t]['T'],color=colors[i],alpha=0.2)
        #     i+=1
        # plt.plot(df.index,df['T_raw'],color='black')
    except Exception as e:
        print(f"Fehler bei der Verarbeitung von {simName}: {e}")
        continue
