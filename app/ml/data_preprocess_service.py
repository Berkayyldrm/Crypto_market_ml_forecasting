import pandas as pd
from app.general.ConfigManager import ConfigManager
from default_lists import selected_columns, all_columns
config = ConfigManager()
lag_features = config.lag_features
def data_preprocess_service(data, dev=False):
    if dev:
        data = data[all_columns]
    if not dev:
        data = data[selected_columns]
               
    # "Overbought", "Oversold", "Neutral"#########################################################################################
    momentum_over_columns = ["RSI_I", "WILLR_I", "CCI_I", "UO_I"]

    momentum_over_mapping = {
        "Overbought": 1,
        "Neutral": 0,
        "Oversold": -1,
    }
    for column in momentum_over_columns:
        data.loc[:, column] = data[column].map(momentum_over_mapping)
        data[column] = data[column].astype(int)


    # Buy, Sell, Neutral#########################################################################################
    momentumish_columns = ["STOCH_I", "STOCHRSI_I", "MACD_I", "KDJ_I"]
    
    momentumish_mapping = {
        "Bullish-Momentum": 2,
        "Buy": 1,
        "Sell": -1,
        "Bearish-Momentum": -2
    }
    for column in momentumish_columns:
        data.loc[:, column] = data[column].map(momentumish_mapping)
        data[column] = data[column].astype(int)


    #########################################################################################
    momentumish_columns_2 = ["BBP_I", "CMF_I", "AO_I"]
    momentum_over_mapping_2 = {
        "Bullish-Momentum": 1,
        "Neutral": 0,
        "Bearish-Momentum": -1
    }

    for column in momentumish_columns_2:
        data.loc[:, column] = data[column].map(momentum_over_mapping_2)
        data[column] = data[column].astype(int)


    # Band Trend#########################################################################################
    band_columns = ["BBANDS_I", "KC_I"]
    band_mapping = {
        "Bullish-Trend": 2,
        "Buy": 1,
        "Neutral": 0,
        "Stabilized": -42,
        "Sell": -1,
        "Bearish-Trend": -2
    }
    for column in band_columns:
        data.loc[:, column] = data[column].map(band_mapping)
        data[column] = data[column].astype(int)


    # Trend#########################################################################################
    trend_columns = ["MA_sm_I", "MA_sl_I", "MA_ml_I", "MA_mix_I", "PSAR_I", "OBV_I", "SuperTrend_I"]
    trend_mapping = {
        "Buy": 1,
        "Sell": -1,
    }
    for column in trend_columns:
        data.loc[:, column] = data[column].map(trend_mapping)
        data[column] = data[column].astype(int)

    #########################################################################################
    
    trend_columns_2 = ["ADX_I"]
    trend_mapping_2 = {
        "Buy": 2,
        "Neutral-Buy": 1,
        "Neutral-Sell": -1,
        "Sell": -2,
    }
    for column in trend_columns_2:
        data.loc[:, column] = data[column].map(trend_mapping_2)
        data[column] = data[column].astype(int)

    # Pivot Mapping #########################################################################################

    categoric_pivot_interpretation_columns = ["Classic_Pivot_1D", "Fibonacci_Pivot_1D", "Classic_Pivot_4H", "Fibonacci_Pivot_4H"]
    categoric_pivot_mapping = {
        "Strong-Sell": -3,
        "Sell": -2,
        "Neutral-Sell": -1,
        "Neutral-Buy": 1,
        "Buy": 2,
        "Strong-Buy": 3
    }
    for column in categoric_pivot_interpretation_columns:
        data.loc[:, column] = data[column].map(categoric_pivot_mapping)
        data[column] = data[column].astype(int)

    
    if lag_features:
        df = data.copy()
        temp_df = pd.DataFrame()
        lag_feature_names = ["RSI_I", "STOCH_I", "STOCHRSI_I", "MACD_I", "BBANDS_I", "ADX_I", "WILLR_I", "CCI_I",
                             "UO_I", "BBP_I", "MA_sm_I", "MA_sl_I", "MA_ml_I", "MA_mix_I", "KDJ_I",
                             "PSAR_I", "OBV_I", "CMF_I", "AO_I", "KC_I", "SuperTrend_I"]
        missing_columns = [col for col in data.columns if col not in lag_feature_names]
        for feature in lag_feature_names:
            for i in range(1, 2):
                df[f'{feature}_lag{i}'] = df[feature].shift(i)
            temp_df[feature] = df[[feature] + [f'{feature}_lag{i}' for i in range(1, 2)]].astype(str).agg('*'.join, axis=1)

        temp_df = temp_df.iloc[1:, :]

        temp_df = pd.get_dummies(temp_df[lag_feature_names], dtype=int)
        data = pd.concat([data[missing_columns], temp_df], axis=1)
    return data