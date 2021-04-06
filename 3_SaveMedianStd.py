import numpy as np
from rex import WindResource
from rex.utilities.execution import SpawnProcessPool
import pandas as pd
import os

if __name__=='__main__':

    # Single run 2017 file
    root_dir_single = '/datasets/WIND/Offshore_CA/v1.0.0/'
    single_run_file = root_dir_single + 'Offshore_CA_2017.h5'
    ml_dir = '/projects/oswwra/ML_AnEn/ML_std_pred/'
    out_dir = '/projects/oswwra/ML_AnEn/'
    
    # Get list of sites
    with WindResource(single_run_file) as f:
        site_id_list = f.meta.index
        lat_list = f.meta.latitude
        lon_list = f.meta.longitude
    
    avg_stdev = pd.DataFrame(index = site_id_list, columns = ['lat', 'lon','median', 'mean'])
    
    for site in np.arange(np.shape(site_id_list)[0]): # site_id_list
        avg_stdev.loc[site,'lat'] = lat_list[site]
        avg_stdev.loc[site,'lon'] = lon_list[site]
        if os.path.isfile(ml_dir + 'ML_stdev_20yrs_site_'+ str(site)+'.csv'):
            df = pd.read_csv(ml_dir + 'ML_stdev_20yrs_site_'+ str(site)+'.csv')
            avg_stdev.loc[site,'median'] =  float(df.median().values)
            #avg_stdev.loc[site, 'mean'] =  float(df.mean().values)
    
    avg_stdev.to_csv(out_dir + 'median_stdev_all_sites_1.csv')



