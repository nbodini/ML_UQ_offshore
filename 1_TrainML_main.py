import numpy as np
from rex import WindResource
from sklearn import ensemble
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from rex.utilities.execution import SpawnProcessPool
from TrainML_function import ML_loop
import pandas as pd

if __name__=='__main__':

    # Single run 2017 file
    root_dir_single = '/datasets/WIND/Offshore_CA/v1.0.0/'
    single_run_file = root_dir_single + 'Offshore_CA_2017.h5'

    # Get list of sites
    with WindResource(single_run_file) as f:
        site_id_list = f.meta.index
    
    # Spawn parallel processes to produce csvs for each site
    with SpawnProcessPool() as ex:
        for site in np.arange(np.shape(site_id_list)[0]): # site_id_list
            print(site)
            ex.submit(ML_loop, single_run_file, site)



