import numpy as np
import pandas as pd
from tigramite.causal_effects import CausalEffects
import sklearn
from sklearn.linear_model import LinearRegression
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from climateclassifier.climateclassy import DataLoader
from datetime import datetime as dt
import logging
import os

app_name = './first_dataset_tests' # with monthly frequency
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
partial_results_file = os.path.join(results_path,'cefs.csv')

df_path = './data/ugent_data'
variables = ['max_5consec_Ix5dayRN_ERA','max_5consec_Sx5daySM_GLEAM','T_CRU_Residuals','monthly_Rx1day_P_CPCU']
data = DataLoader(df_path, variables, data_format = 'csv', substract_seasonality=[True, True, True, True])


def caus_ef_comp(graph, X, Y, df0):
    try:
        causal_effects = CausalEffects(graph, graph_type='stationary_dag', X=X, Y=Y, S=None, 
                                hidden_variables=None, 
                                verbosity=0)
        causal_effects.fit_total_effect(
                dataframe=df0, 
                estimator=LinearRegression(),
                adjustment_set='optimal',
                conditional_estimator=None,  
                data_transform=None,
                mask_type=None,
                )
        intervention_data = 1.*np.ones((1, 1))
        y1 = causal_effects.predict_total_effect( 
                intervention_data=intervention_data
                )
        intervention_data = 0.*np.ones((1, 1))
        y2 = causal_effects.predict_total_effect( 
                intervention_data=intervention_data
                )

        cef = (y1 - y2)
        cef = cef.item()
    except:
        cef = 0
    return cef


def save_partial_results(results_df, filename, mode='a'):
    if not os.path.exists(filename) or mode == 'w':
        results_df.to_csv(filename, index=False, mode='w')
    else:
        results_df.to_csv(filename, index=False, mode='a', header=False)


variables = ['Radiation','SM','Temp','Prec']
parcorr = ParCorr(significance='analytic')
cefs = [[] for _ in range(35)]
lat = []
lon = []


link_assumptions = {}
N = data.df.shape[2]
tau_max = 2

link_assumptions[0] = {
    (i, -tau): '<?-' if (i, -tau) != (0, 0) and i != 0 else 'o?o'
    for i in range(N)
    for tau in range(tau_max + 1)
    if (i, -tau) != (0, 0)  # Filtering condition to exclude (1, 0)
}

link_assumptions[1] = {
    (i, -tau): '-?>' if (i, -tau) != (1, 0) and i == 0 else 'o?o'
    for i in range(N)
    for tau in range(tau_max + 1)
    if (i, -tau) != (1, 0)  # Filtering condition to exclude (1, 0)
}

link_assumptions[2] = {
    (i, -tau): '-?>' if (i, -tau) != (2, 0) and i == 0 else 'o?o'
    for i in range(N)
    for tau in range(tau_max + 1)
    if (i, -tau) != (2, 0)  # Filtering condition to exclude (1, 0)
}

link_assumptions[3] = {
    (i, -tau): '-?>' if (i, -tau) != (3, 0) and i == 0 else 'o?o'
    for i in range(N)
    for tau in range(tau_max + 1)
    if (i, -tau) != (3, 0)  # Filtering condition to exclude (1, 0)
}


var_names = []
for i in range(N):
    for j in range(N):
        for t in range(tau_max +1):
            if (i,t) != (j,0):
                if i != 0 and j == 0:
                    pass
                else:    
                    var_names.append('{} -> {} at lag {}'.format(i,j,-t))


batch_size = 1000 
current_batch = []

for k in range(data.df.shape[0]):

    logging.info(k)
    df0 = pp.DataFrame(data.df[k,:,:], var_names= variables)
    pcmci = PCMCI(
        dataframe=df0, 
        cond_ind_test=parcorr,
        verbosity=0)

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None, link_assumptions=link_assumptions)

    pixel_results = []
    for i in range(N):
        for j in range(N):
            for t in range(tau_max +1):
                if (i,t) != (j,0):
                    if i != 0 and j == 0:
                        pass
                    else:    
                        pixel_results.append(caus_ef_comp(results['graph'], [(i,-t)], [(j,0)], df0))

    current_batch.append(pixel_results + [data.latitud[k], data.longitud[k]])

    if len(current_batch) >= batch_size or k == data.df.shape[0] - 1:
        batch_df = pd.DataFrame(current_batch, columns=var_names + ['lat', 'long'])
        save_partial_results(batch_df, partial_results_file)
        current_batch = []  # Clear the batch after saving

print(f"All results saved to {partial_results_file}")





