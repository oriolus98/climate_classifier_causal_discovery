import pytest
import numpy as np
from climateclassifier.climateclassy import DataLoader


@pytest.fixture
def data():
    df_path = 'data/ugent_data'
    variables = ['MonthlyRes_Mean_RN_ERA_Residuals','MonthlyRes_Mean_SM_GLEAM_Residuals','T_CRU_Residuals','P_CRU_Residuals']
    data = DataLoader(df_path, variables, data_format = 'csv', substract_seasonality=[False, False,False,False])
    return data

def test_data_loading(data):
    assert data.df is not None
    assert isinstance(data.df, np.ndarray)
    assert data.df.ndim == 3

def test_missing_values(data):
    assert np.sum(np.isnan(data.df)) == 0


