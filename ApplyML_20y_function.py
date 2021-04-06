import numpy as np
import pandas as pd
import glob
from operational_analysis.toolkits import met_data_processing
from rex import WindResource
from time import perf_counter
import pickle as pk

# Define function to apply trained ML model to 20-year run
def ML_loop(site):
    # Time the process
    start = perf_counter()
    root_dir_single = '/datasets/WIND/Offshore_CA/v1.0.0/'
    out_dir = '/projects/oswwra/ML_AnEn/ML_std_pred/'
    ml_dir = '/projects/oswwra/ML_AnEn/ML_model_trained/'
    # _______________________________
    # Read data from single run
    # _______________________________
    # Process all variables except for wind direction
    variables = ['inversemoninobukhovlength_2m', 'windspeed_10m',
                    'windspeed_200m', 'windspeed_100m', 'temperature_40m']
    main_df = pd.DataFrame()
    # Get list of ensemble files
    mod_list = glob.glob(root_dir_single + '/*.h5')
    for filename in sorted(mod_list):
        with WindResource(filename) as f:
            print(filename)
            df = {variable: f[variable, :, site] for variable in variables}
            df = pd.DataFrame(df, index=f.time_index)
            main_df_temp = df.resample("H").mean()
            # Process wind direction
            wspd = f['windspeed_100m', :, site]
            wdir = f['winddirection_100m', :, site]
            u, v = met_data_processing.compute_u_v_components(wspd, wdir)
            df_temp = pd.DataFrame(index=f.time_index)
            df_temp['u_wnd'] = u
            df_temp['v_wnd'] = v
            df_hourly_wd = df_temp.resample("H").mean()
            wdir_final = \
                met_data_processing.compute_wind_direction(df_hourly_wd['u_wnd'],
                                                           df_hourly_wd['v_wnd'])
            main_df_temp['winddirection_100m'] = wdir_final
            # Concatenate into a single dataframe
            main_df = pd.concat([main_df, main_df_temp], axis=0)
    # _______________________________
    # Pre-processing to be able to apply trained ML model
    # _______________________________
    # Consider hour and month as part of the ML model
    main_df.loc[:, 'hour'] = main_df.index.hour
    main_df.loc[:, 'month'] = main_df.index.month
    # Consider the change in wind speed as a method for prediction
    main_df['6_hour_ws_change'] = main_df['windspeed_100m'].rolling(6).std()
    main_df['1_hour_ws_diff'] = \
        main_df['windspeed_100m'].rolling(window=2).std()
    # Shear exponent
    main_df['shear_exp'] = (np.log(main_df['windspeed_200m']
                                   / main_df['windspeed_10m'])
                            / np.log(200 / 10))
    # Treat NaNs
    main_df.dropna(inplace=True)
    # main_df.to_csv(out_dir + '20year_'+str(site)+'.csv')
    # _______________________________
    # Apply trained ML model
    # _______________________________
    # Load ML model
    filename = ml_dir + 'gbm_model_' + str(site) + '.sav'
    random_search = pk.load(open(filename, "rb"))
    # Use ML model on the 20-year dataset
    cols = ['inversemoninobukhovlength_2m', 'shear_exp', 'windspeed_100m',
            'temperature_40m', 'winddirection_100m', 'hour', 'month',
            '6_hour_ws_change', '1_hour_ws_diff']
    X_20y = main_df[cols]
    stdev_20y = random_search.best_estimator_.predict(X_20y)
    stdev_20y_pandas = pd.DataFrame(data = stdev_20y, index = X_20y.index, columns = [str(site)])
    # Save stdev time series
    stdev_20y_pandas.to_csv(out_dir + 'ML_stdev_20yrs_site_'+ str(site)+'.csv')
    stop = perf_counter()
    print(stop-start, 'seconds to read data from 20 years and apply ML model') 




