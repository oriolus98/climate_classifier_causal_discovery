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

df_path = 'data/ugent_data2'
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


for k in range(data.df.shape[0]):
    df0 = pp.DataFrame(data.df[k,:,:], var_names= variables)
    pcmci = PCMCI(
        dataframe=df0, 
        cond_ind_test=parcorr,
        verbosity=0)

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None, link_assumptions=link_assumptions)

    count = 0
    for i in range(N):
        for j in range(N):
            for t in range(tau_max +1):
                if (i,t) != (j,0):
                    if i != 0 and j == 0:
                        pass
                    else:    
                        cefs[count].append(caus_ef_comp(results['graph'], [(i,-t)], [(j,0)], df0))
                        count += 1

    lat.append(data.latitud[k])
    lon.append(data.longitud[k])


var_names = []
for i in range(N):
    for j in range(N):
        for t in range(tau_max +1):
            if (i,t) != (j,0):
                if i != 0 and j == 0:
                    pass
                else:    
                    var_names.append('{} -> {} at lag {}'.format(i,j,-t))

data_dict = {}
for num, name in enumerate(var_names):
    data_dict[name] = cefs[num]

results = pd.DataFrame(data_dict)
results['lat'] = lat
results['long'] = lon
results.to_csv('../first_dataset_tests/240701/results/cefs.csv', index= False)