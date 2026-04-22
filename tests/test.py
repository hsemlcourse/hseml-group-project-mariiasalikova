import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing import clean_column_names, feature_engineering

def test_clean_column_names():
    df = pd.DataFrame({'Test Column Name': [1, 2], 'Another.Col': [3, 4]})
    cleaned_df = clean_column_names(df)
    assert list(cleaned_df.columns) == ['test_column_name', 'another_col']

def test_feature_engineering():
    df = pd.DataFrame({
        'sent_tnx': [10, 0],
        'received_tnx': [5, 0],
        'total_ether_sent': [100.0, 0.0]
    })
    processed = feature_engineering(df)
    
    # Создание ratio
    assert 'ratio_sent_received' in processed.columns
    assert np.isclose(processed['ratio_sent_received'].iloc[0], 10 / (5 + 1e-5))
    
    # Логарифмирование
    assert 'log_total_ether_sent' in processed.columns
    assert processed['log_total_ether_sent'].iloc[1] == 0.0 # log(0 + 1) = 0