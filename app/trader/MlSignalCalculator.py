import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.time_series.TimeSeriesFinance import LSTMFinance, TransformerFinance, XGBoostFinance
from app.ml.data_preprocess_service import data_preprocess_service
from app.data.dataframe_service import get_data_as_dataframe, get_dataframe, get_technical_data_as_dataframe
from default_lists import selected_coins
from app.general.ConfigManager import ConfigManager

class MlSignalCalculator:
    def __init__(self):
        self.config = ConfigManager()
        self.engine = self.config.get_db_engine()
        self.ts_mode = self.config.prediction_ts_mode
        self.coin_based_mode = self.config.prediction_coin_based_mode
        self.general_model_mode = self.config.prediction_general_model_mode
        if self.general_model_mode:
            self.general_model = pickle.load(open("./app/ml_models/coin/general_ml.pkl", 'rb'))
        if self.ts_mode:
            self.ts_model_type = self.config.ts_model_type
            self.time_step = self.config.time_step
    
    def calculate_signal_for_symbol(self, symbol):
        data_technical_info = get_technical_data_as_dataframe(schema_name="pred_technical", table_name=symbol)
        data_technical_info = data_technical_info.iloc[-1:,:] # Get last value for prediction
        data_basic_info = get_data_as_dataframe(schema_name="pred_data", table_name=symbol)
        data_basic_info_pred_ts = data_basic_info.copy()
        data_basic_info = data_basic_info.iloc[-1:,:] # Get last value for prediction
        data_df = data_technical_info.merge(data_basic_info, on="date", how="left")
        data_df = data_df.reset_index()

        data = data_preprocess_service(data_df)
        X = data.drop(["date", "percentage"], axis=1)

        if self.ts_mode:
            data_basic_info_pred_ts = data_basic_info_pred_ts[["date", "close"]]
            data_basic_info_train_ts = get_dataframe(schema_name="train_ts_data", table_name=symbol)

            if self.ts_model_type == 'XGBoost':
                ts_finance = XGBoostFinance()
            elif self.ts_model_type == 'Bi_LSTM':
                ts_finance = LSTMFinance()
            elif self.ts_model_type == 'Transformer':
                ts_finance = TransformerFinance()
            else:
                raise ValueError("Unsupported model type")
            ts_model = ts_finance.load_model(symbol=symbol)
            ts_model_scaler = ts_finance.load_scaler(symbol=symbol)
            additional_data = ts_finance.get_additional_data_for_pred(train_data=data_basic_info_train_ts, pred_data=data_basic_info_pred_ts)
            additional_data.to_sql(symbol, self.engine, schema="train_ts_data", if_exists='append', index=False)
            data_ts = get_dataframe(schema_name="train_ts_data", table_name=symbol)
            data_ts = data_ts.iloc[(-1*self.time_step):, :]
            data_ts = data_ts[["close"]]
            data_pred_ts = ts_finance.data_creator_prediction(data_set=data_ts, time_step=self.time_step, scaler=ts_model_scaler, scale=True)
            forecast, _ = ts_finance.forecast_data(data=data_pred_ts, model=ts_model, scaler=ts_model_scaler)
            perc_forecast, _ = ts_finance.convert_2_percentage(forecast=forecast, close_reel=data_ts.iloc[-1:, :], time_step=self.time_step, prediction_mode=True)
            ts_model_result = pd.Series(perc_forecast[0], index=np.arange(1))
        else:
            ts_model_result = pd.Series(0, index=np.arange(X.shape[0]))
        if self.general_model_mode:
            try:
                general_model_result = self.general_model.predict(X)
            except ValueError as e:
                general_model_result = pd.Series(0, index=np.arange(X.shape[0]))
                print("general model error")
        else:
            general_model_result = pd.Series(0, index=np.arange(X.shape[0]))
        if self.coin_based_mode:
            try:
                filename = f'app/ml_models/coin/coin_based_{symbol}.pkl'
                coin_based_model = pickle.load(open(filename, 'rb'))
                coin_based_model_result = coin_based_model.predict(X)
            except ValueError as e:
                coin_based_model_result = pd.Series(0, index=np.arange(X.shape[0]))
                print("coin based model error")
        else:
            coin_based_model_result = pd.Series(0, index=np.arange(X.shape[0]))

        return {
            'coin': symbol,
            'general_ml_result': general_model_result[0],
            'coin_based_ml_result': coin_based_model_result[0],
            'ts_result': ts_model_result.iloc[0]
        }

    def calculate_signals(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.calculate_signal_for_symbol, symbol) for symbol in selected_coins.keys()]
            results = [future.result() for future in as_completed(futures)]
        df = pd.DataFrame(results)
        df['date'] = datetime.now()
        df = df[['date', 'coin', 'general_ml_result', 'coin_based_ml_result', 'ts_result']]
        df["result"] = df["general_ml_result"] + df["coin_based_ml_result"] + df["ts_result"]
        df.to_sql("ml_signal", self.engine, schema="pred_ml_signal", if_exists='append', index=False)
        return df