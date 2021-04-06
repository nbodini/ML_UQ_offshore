import numpy as np
import pandas as pd
import glob
from operational_analysis.toolkits import met_data_processing
from rex import WindResource
from sklearn import ensemble
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from time import perf_counter
import pickle as pk

#  Function to reate a custom train-test split function that takes 20% of data
#  per year-month
def custom_monthly_test_set(df):
    np.random.seed(42)
    # First get list of unique year-months
    year_month = df.index.strftime("%Y-%m")
    unique_ym = year_month.unique()
    #  Define data frame to fill
    test_df = pd.DataFrame()
    #  Loop through each year-month and take 20% of the data as a test set
    for ym in unique_ym:
        #  Get subset of data frame matching year-month
        df_sub = df.loc[year_month == ym]
        #  Number of data points in selected year-month
        num_points = df_sub.shape[0]
        #  Number of data points corresponding to 20% of data
        test_length = int(0.1 * num_points)
        #  Generate random number that defines the start of the test index
        ind_start = np.random.randint(0, int(0.9 * num_points))
        #  Add data starting at that index to test data frame
        df_slice = slice(ind_start, (ind_start + test_length))
        test_df = test_df.append(df_sub.iloc[df_slice, :])
    return(test_df)
    
# Define function to read data and train ML model at each site
def ML_loop(wind_h5, site):
    # Time the process
    start = perf_counter()
    # _______________________________
    # Define some variables first
    # _______________________________
    root_dir_ensemble = '/datasets/WIND/Offshore_CA/v1.0.0/ensembles/'
    root_dir_single = '/datasets/WIND/Offshore_CA/v1.0.0/'
    out_dir = '/projects/oswwra/ML_AnEn/ML_model_trained/'
    # Gradient boosting model
    params = {"learning_rate": [0.05, 0.1, 1],
              "max_depth": [4, 6, 8, 10],
              "max_features": sp_randint(1, 7),
              "min_samples_split": sp_randint(2, 20),
              "min_samples_leaf": sp_randint(1, 20),
              "n_estimators": np.arange(100, 301, 20)}
    mod = ensemble.GradientBoostingRegressor(warm_start=True)
    random_search = RandomizedSearchCV(mod,
                                       cv=ShuffleSplit(n_splits=5,
                                                       test_size=.2),
                                       param_distributions=params,
                                       n_iter=20,
                                       scoring='r2')
    # _______________________________
    # Read data from 2017 single run
    # _______________________________
    # Process all variables except for wind direction
    variables = ['inversemoninobukhovlength_2m', 'windspeed_10m',
                    'windspeed_200m', 'windspeed_100m', 'temperature_40m']
    with WindResource(wind_h5) as f:
        df = {variable: f[variable, :, site] for variable in variables}
        df = pd.DataFrame(df, index=f.time_index)
        main_df = df.resample("H").mean()
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
        main_df['winddirection_100m'] = wdir_final
    # _______________________________
    # Read data from 2017 ensemble runs
    # _______________________________
        df_en = pd.DataFrame(index=f.time_index)
    # Get list of ensemble files
    mod_list = glob.glob(root_dir_ensemble + '/*.h5')
    for file_ensemble in mod_list:
        with WindResource(file_ensemble) as f_e:
            df_en_temp = f_e['windspeed_100m', :, site]
            df_en_temp = pd.DataFrame(df_en_temp, index=f_e.time_index)
            df_en = pd.concat([df_en, df_en_temp], axis=1)
    df_en = df_en.resample("H").mean()
    # _______________________________
    # Pre-processing for ML
    # _______________________________
    # Calculate std of 100m wind speeds from model
    main_df.loc[:, 'std100'] = df_en.std(axis=1)
    # Normalize by wind speed
    main_df.loc[:, 'std100'] = (main_df.loc[:, 'std100']
                                / main_df.loc[:, 'windspeed_100m'])
    # Consider hour and month as part of the ML model
    main_df.loc[:, 'hour'] = main_df.index.hour
    main_df.loc[:, 'month'] = main_df.index.month
    # Consider the change in wind speed as a method for prediction
    main_df['6_hour_ws_change'] = \
        main_df['windspeed_100m'].rolling(6).std()
    main_df['1_hour_ws_diff'] = \
        main_df['windspeed_100m'].rolling(window=2).std()
    # Shear exponent
    main_df['shear_exp'] = (np.log(main_df['windspeed_200m']
                                   / main_df['windspeed_10m'])
                            / np.log(200 / 10))
    # Treat NaNs
    main_df.dropna(inplace=True)
    # Create train and test sets
    test_df = custom_monthly_test_set(main_df)
    training_df = main_df.drop(test_df.index)
    cols = ['inversemoninobukhovlength_2m', 'shear_exp', 'windspeed_100m',
            'temperature_40m', 'winddirection_100m', 'hour', 'month',
            '6_hour_ws_change', '1_hour_ws_diff']
    X_train = training_df[cols]
    y_train = training_df['std100']
    # _______________________________
    # Train ML model
    # _______________________________
    random_search.fit(X_train, y_train)
    # Save trained model to disk
    filename = out_dir + 'gbm_model_' + str(site) + '.sav'
    pk.dump(random_search, open(filename, 'wb'))
    
    stop = perf_counter()
    print(stop-start, 'seconds to train ML model on 2017')   
        




