from app.general.ConfigManager import ConfigManager
from app.time_series.TimeSeriesFinance import LSTMFinance, TransformerFinance, XGBoostFinance
from app.data.dataframe_service import get_data_as_dataframe
from default_lists import selected_coins, ts_params
import warnings
warnings.filterwarnings('ignore')

def time_series_service():
    config = ConfigManager()
    engine = config.get_db_engine()
    time_step = config.time_step
    ts_model_type = config.ts_model_type

    for symbol in selected_coins.keys():
        data_basic_info = get_data_as_dataframe(schema_name="train_data", table_name=symbol)
        data = data_basic_info.dropna()
        data = data[["date", "close"]]
        data.to_sql(symbol, engine, schema="train_ts_data", if_exists='replace', index=False)
        data = data[["close"]]
        params = ts_params
        if ts_model_type == 'XGBoost':
            ts_finance = XGBoostFinance()
        elif ts_model_type == 'Bi_LSTM':
            ts_finance = LSTMFinance()
        elif ts_model_type == 'Transformer':
            ts_finance = TransformerFinance()
        else:
            raise ValueError("Unsupported model type")
        X_data, y_data, scaler = ts_finance.data_creator(data_set=data, time_step=time_step, scale=True)
        model = ts_finance.create_model(X_data=X_data, y_data=y_data, time_step=time_step, **params[ts_model_type])
        ts_finance.save_model(symbol=symbol, model=model)
        ts_finance.save_scaler(symbol=symbol, scaler=scaler)
