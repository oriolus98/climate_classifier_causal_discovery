import os
from datetime import datetime as dt
import logging
from climateclassy import DataLoader, ClimateClassifier


# we create a directory to check info from each execution
app_name = './second_dataset_tests' # with weekly frequency
date = dt.today().strftime('%y%m%d')
logs_path = 'logs'

filename = dt.today().strftime('%y%m%d')  + '.log'

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


df_path = './ccm_dataset/'
path_to_model = './second_dataset_tests/230411/results/best_model.h5'
variables = ['air_temperature_2m', 'precipitation', 'soil_moisture']

data = DataLoader(df_path, variables, data_format = 'nc')

classy = ClimateClassifier(data, results_path, n_clusters = 5, sample_size = 400, path_to_model= path_to_model, epochs_step1= 100, epochs_final=3)
classy.classify()