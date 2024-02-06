import os
import numpy as np
from datetime import datetime as dt
import logging
from climateclassy import DataLoader, ClimateClassifier, GridSearcher


# we create a directory to check info from each execution
app_name = './tfm/first_dataset_tests' # with monthly frequency
date = dt.today().strftime('%y%m%d')
logs_path = 'logs'

filename = dt.today().strftime('%y%m%d')  + '.log'

if not os.path.isdir(app_name):
    os.mkdir(app_name)

if not os.path.isdir(os.path.join(app_name, date)):
    os.mkdir(os.path.join(app_name, date))

if not os.path.isdir(os.path.join(app_name, date, logs_path)):
    os.mkdir(os.path.join(app_name, date, logs_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',

    handlers=[
        logging.FileHandler(os.path.join(app_name, date, logs_path, filename)),
        logging.StreamHandler()
    ]
)

if not os.path.isdir(os.path.join(app_name, date, 'results')):
    os.mkdir(os.path.join(app_name, date, 'results'))

results_path = os.path.join(app_name, date, 'results')


df_path = './tfm/data/test_data/new_data'
variables = ['Lag_RN_ERA_Ix5day_2', 'Lag_SM_GLEAM_ResPlusTrend_Sx5day_2','maxTXx_cum_T_ERA_2', 'monthly_Rx1day_P_CPCU']
# path_to_model = './first_dataset_tests/230414/results/best_model_step1.h5'

data = DataLoader(df_path, variables, data_format = 'csv')

# GS = GridSearcher(data, results_path= results_path, n_fold = 5, epochs=50000)
# best_opt, best_actf = GS.search()

# logging.info('Best optimizer {}'.format(best_opt))

classy = ClimateClassifier(data = data, results_path = results_path, n_clusters = 5, sample_size = 400, path_to_model = None, epochs_step1= 80000, epochs_final= 200)
classy.classify()
