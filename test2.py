import os
import numpy as np
from datetime import datetime as dt
import logging
from climateclassifier.climateclassy import DataLoader, ClimateClassifier, GridSearcher


# we create a directory to check info from each execution
app_name = './tfm/first_dataset_tests' # with monthly frequency
date = dt.today().strftime('%y%m%d')
logs_path = 'logs'

filename = dt.today().strftime('%y%m%d')  + '.log'

if not os.path.isdir(app_name):
    os.makedirs(app_name)

if not os.path.isdir(os.path.join(app_name, date)):
    os.makedirs(os.path.join(app_name, date))

if not os.path.isdir(os.path.join(app_name, date, logs_path)):
    os.makedirs(os.path.join(app_name, date, logs_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',

    handlers=[
        logging.FileHandler(os.path.join(app_name, date, logs_path, filename)),
        logging.StreamHandler()
    ]
)

if not os.path.isdir(os.path.join(app_name, date, 'results')):
    os.makedirs(os.path.join(app_name, date, 'results'))

results_path = os.path.join(app_name, date, 'results')


#df_path = './tfm/data/test_data/new_data'
df_path = './tfm/climate_classifier_causal_discovery/data/ugent_data2'
# variables = ['Lag_RN_ERA_Ix5day_2', 'Lag_SM_GLEAM_ResPlusTrend_Sx5day_2','maxTXx_cum_T_ERA_2', 'monthly_Rx1day_P_CPCU']  bon clustering mal data pq lag
# variables = ['MonthlyRes_Mean_RN_ERA_Residuals','MonthlyRes_Mean_SM_GLEAM_Residuals','T_CRU_Residuals','P_CRU_Residuals'] ugent_data clustering regular
variables = ['max_5consec_Ix5dayRN_ERA','max_5consec_Sx5daySM_GLEAM','T_CRU_Residuals','monthly_Rx1day_P_CPCU']
#path_to_model_step1 = './tfm/first_dataset_tests/240527/results/best_model_step1.h5'
#best_model_path = './tfm/first_dataset_tests/240212/results/best_model_final.h5'

data = DataLoader(df_path, variables, data_format = 'csv', substract_seasonality=[True, True, True, True])

# GS = GridSearcher(data, results_path= results_path, n_fold = 5, epochs=50000)
# best_opt, best_actf = GS.search()

# logging.info('Best optimizer {}'.format(best_opt))

classy = ClimateClassifier(data = data, results_path = results_path, n_clusters = 5, sample_size = 3000, dim_red_cluster = True, epochs_step1= 100000, epochs_final= 20000)
classy.classify()
