from binance.client import Client

top_n_coin = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
    "XRPUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "ARBUSDT",
    "LINKUSDT", "LTCUSDT", "BCHUSDT", "ALGOUSDT", "MATICUSDT",
    "XLMUSDT", "VETUSDT", "ICPUSDT", "TONUSDT",
    "FILUSDT", "TRXUSDT", "FETUSDT", "ARKMUSDT", "WLDUSDT",
]

selected_coins = {
    "BTCUSDT": 0,
    }

all_columns = ["date", "RSI", "RSI_I", "STOCH_I", "STOCHRSI_I", "MACD_I", "BBANDS_I", "ADX", "ADX_I",
                    "WILLR", "WILLR_I", "CCI", "CCI_I", "ATR_Percentage", "UO", "UO_I", "BBP", "BBP_I", "SMA_10", "SMA_20",
                    "SMA_50", "SMA_100", "EMA_12", "EMA_26", "EMA_50", "EMA_100", "SMA_I", "EMA_I", "Classic_Pivot", "Demark_Pivot", 
                    "KDJ", "KDJ_I", "ICHIMOKU_I", "PSAR_I", "OBV", "OBV_I", "CMF", "CMF_I", "AO", "AO_I", "KC", "KC_I", "TD_SEQ_I", "percentage"]

selected_columns = ["date", "RSI_I", "STOCH_I", "STOCHRSI_I", "MACD_I", "BBANDS_I", "ADX_I",
                    "WILLR_I", "CCI_I", "ATR_Percentage", "UO_I", "BBP_I", "MA_sm_I", "MA_sl_I",
                    "MA_ml_I", "MA_mix_I", "Classic_Pivot_1D", "Classic_Pivot_4H", "Fibonacci_Pivot_1D", "Fibonacci_Pivot_4H",
                    "KDJ_I", "PSAR_I", "OBV_I", "CMF_I", "AO_I", "KC_I", "SuperTrend_I", "percentage"]


interval_type_dict = {
        "minutely": {
            "start_interval":Client.KLINE_INTERVAL_1MINUTE,
            "start_str_prediction": "2 days ago UTC",
            "start_str_train": "300 hours ago UTC",
            "start_str_simulation": "6 months ago UTC",
            "end_str": "1 minutes ago UTC"},
        "5minutes": {
            "start_interval":Client.KLINE_INTERVAL_5MINUTE,
            "start_str_prediction": "2 days ago UTC",
            "start_str_train": "1 years ago UTC",
            "start_str_simulation": "4 months ago UTC",
            "end_str": "5 minutes ago UTC"},
        "15minutes": {
            "start_interval":Client.KLINE_INTERVAL_15MINUTE,
            "start_str_prediction": "2 days ago UTC",
            "start_str_train": "1 years ago UTC",
            "start_str_simulation": "2 months ago UTC",
            "end_str": "15 minutes ago UTC"},
        "30minutes": {
            "start_interval":Client.KLINE_INTERVAL_30MINUTE,
            "start_str_prediction": "2 days ago UTC",
            "start_str_train": "1 years ago UTC",
            "start_str_simulation": "40 months ago UTC",
            "end_str": "24 months ago UTC"},
        "hourly": {
            "start_interval":Client.KLINE_INTERVAL_1HOUR,
            "start_str_prediction": "2 days ago ago UTC",
            "start_str_train": "24 month ago UTC",
            "start_str_simulation": "4 months ago UTC",
            "end_str": "1 hours ago UTC"},
        "4hours": {
            "start_interval":Client.KLINE_INTERVAL_4HOUR,
            "start_str_prediction": "2 week ago UTC",
            "start_str_train": "48 month ago UTC",
            "start_str_simulation": "4 months ago UTC",
            "end_str": "4 hours ago UTC"},
        "8hours": {
            "start_interval":Client.KLINE_INTERVAL_8HOUR,
            "start_str_prediction": "2 week ago UTC",
            "start_str_train": "48 month ago UTC",
            "start_str_simulation": "4 months ago UTC",
            "end_str": "8 hours ago UTC"},
        "daily": {
            "start_interval":Client.KLINE_INTERVAL_1DAY,
            "start_str_prediction": "2 week ago UTC",
            "start_str_train": "48 month ago UTC",
            "start_str_simulation": "40 months ago UTC",
            "end_str": "20 months ago UTC"}
    }

parameters = {
        "rsi_period": 14,

        "stoch_k_period": 14,
        "stoch_d": 3,
        "stoch_smooth_k": 3,

        "stochrsi_period": 14,
        "stochrsi_rsi_period": 14,
        "stochrsi_k": 3,
        "stochrsi_d": 3,

        "macd_fast_period": 12,
        "macd_slow_period": 26,
        "macd_signal_period": 9,

        "bbands_period": 20,
        "bbands_std": 2.0,
        "bbands_threshold": 0.5,

        "adx_period": 14,

        "willr_period": 21,

        "cci_period": 20,
        "cci_c": 0.015,

        "atr_period": 14,

        "hl_period": 14,

        "uo_short_period": 7,
        "uo_medium_period": 14,
        "uo_long_period": 28,
        
        "bullbear_period": 13,

        "pivot_period": 6,

        "ma_short_period": 9,
        "ma_mid_period": 26,
        "ma_long_period": 52,

        "kdj_period": 9,
        "kdj_signal_period": 3,

        "psar_af": 0.02,
        "psar_max_af": 0.2,

        "cmf_period": 20,

        "ao_fast_period": 5,
        "ao_slow_period": 34,

        "kc_period": 20,
        "kc_scalar": 2.0,
        "kc_threshold": 0.5,

        "super_trend_period": 7,
        "super_trend_multiplier": 3.0
    }

ts_params = {
    'XGBoost': {
        'n_estimators': 261,
        'max_depth': 13,
        'learning_rate': 0.0449236189287773,
        'subsample': 0.8,
        'colsample_bytree': 1.0
        },
    'Bi_LSTM': {
        'learning_rate': 0.0002130874114,
        'batch_size': 16,
        'units': 150,
        'activation': 'tanh',
        'optimizer': 'adam'
        },
    'Transformer': {
        'head_size': 128,
        'num_heads': 7,
        'ff_dim': 8,
        'nums_trans_blocks': 3,
        'mlp_units': [128],
        'dropout': 0.1,
        'mlp_dropout': 0,
        'learning_rate': 0.0007043995273,
        'batch_size': 16
        }
        }

ts_params_simulation = {
    'XGBoost': {
        'n_estimators': 261,
        'max_depth': 13,
        'learning_rate': 0.0449236189287773,
        'subsample': 0.8,
        'colsample_bytree': 1.0
        },
    'Bi_LSTM': {
        'learning_rate': 0.0002130874114,
        'batch_size': 16,
        'units': 150,
        'activation': 'tanh',
        'optimizer': 'adam'
        },
    'Transformer': {
        'head_size': 128,
        'num_heads': 7,
        'ff_dim': 8,
        'nums_trans_blocks': 3,
        'mlp_units': [128],
        'dropout': 0.1,
        'mlp_dropout': 0,
        'learning_rate': 0.0007043995273,
        'batch_size': 16
        }
        }