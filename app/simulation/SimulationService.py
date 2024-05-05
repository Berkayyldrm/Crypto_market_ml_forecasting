import numpy as np
import pandas as pd
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.general.ConfigManager import ConfigManager
from app.general.CustomLogger import CustomLogger
from app.time_series.TimeSeriesFinance import  LSTMFinance, TransformerFinance, XGBoostFinance
from app.simulation.SimulationStrategy import SimulationStrategy
from app.ml.data_preprocess_service import data_preprocess_service
from app.data.dataframe_service import get_technical_data_as_dataframe, get_data_as_dataframe
from default_lists import selected_coins, ts_params_simulation
import warnings
warnings.filterwarnings('ignore')

logger = CustomLogger('simulation', 'log/simulation.log')
logger.clear_log_file()
logger = logger.get_logger()

class SimulationService:
    def __init__(self, **kwargs) -> None:
        self.simulation_start_date = kwargs.get("simulation_start_date", None)
        self.simulation_end_date = kwargs.get("simulation_end_date", None)
        logger.info("Simulation Mode Starting")
        logger.info(f"Simulation Mode Prediction Time Start Date: {self.simulation_start_date}")
        logger.info(f"Simulation Mode Prediction Time End Date {self.simulation_end_date}")
        config = ConfigManager()
        self.simulation_strategy_selection = config.simulation_strategy_selection
        self.ts_mode = config.simulation_ts_mode
        self.coin_based_mode = config.simulation_coin_based_mode
        self.general_model_mode = config.simulation_general_model_mode
        self.time_step = config.simulation_time_step
        self.ts_model_type = config.simulation_ts_model_type
        self.engine = config.get_db_engine()
        self.simulation_strategy = SimulationStrategy()
    
    def get_general_model(self):
        data_df = pd.DataFrame()

        for symbol in selected_coins.keys():       
            data_technical_info = get_technical_data_as_dataframe(schema_name="simulation_technical", table_name=symbol)
            data_basic_info = get_data_as_dataframe(schema_name="simulation_data", table_name=symbol)
            data_temp_df = data_technical_info.merge(data_basic_info, on="date", how="left")
            data_temp_df["percentage"] = data_temp_df["percentage"].shift(-1)
            data_df = pd.concat([data_df, data_temp_df]).reset_index(drop=True)
            data_df = data_df.dropna()

        data = data_preprocess_service(data_df)
        test_data = data[(data['date'].dt.strftime('%Y-%m-%d %H:%M') >= self.simulation_start_date.strftime('%Y-%m-%d %H:%M')) & (data['date'].dt.strftime('%Y-%m-%d %H:%M') < self.simulation_end_date.strftime('%Y-%m-%d %H:%M'))]
        train_data = data[(data['date'].dt.strftime('%Y-%m-%d %H:%M') < self.simulation_start_date.strftime('%Y-%m-%d %H:%M'))]
        
        if train_data.empty or test_data.empty:
            raise ValueError("Check Simulation start and end date. Or check train date range")
        
        test_data.to_sql("general_test", self.engine, schema="simulation_ml_data", if_exists='replace', index=False)
        train_data.to_sql("general_train", self.engine, schema="simulation_ml_data", if_exists='replace', index=False)

        y_train = train_data["percentage"]
        X_train = train_data.drop(["date", "percentage"], axis=1)
        params = {}
        xgb_clf = xgb.XGBRegressor(**params)
        xgb_clf.fit(X_train, y_train)

        self.general_model = xgb_clf

    def get_coin_based_model_result(self, X_train, y_train, X_test):
        if self.coin_based_mode:
            params = {}
            xgb_clf = xgb.XGBRegressor(**params)
            xgb_clf.fit(X_train, y_train)
            try:
                coin_based_model_predictions = pd.Series(xgb_clf.predict(X_test)).reset_index(drop=True)
            except ValueError as e:
                coin_based_model_predictions = pd.Series(0, index=np.arange(X_test.shape[0]))
                print("coin based model error")
        else:
            coin_based_model_predictions = pd.Series(0, index=np.arange(X_test.shape[0]))
        return coin_based_model_predictions

    def get_ts_model_result(self, symbol, data_basic_info):
        test_data_ts = data_basic_info[(data_basic_info['date'].dt.strftime('%Y-%m-%d %H:%M') >= self.simulation_start_date.strftime('%Y-%m-%d %H:%M')) & (data_basic_info['date'].dt.strftime('%Y-%m-%d %H:%M') < self.simulation_end_date.strftime('%Y-%m-%d %H:%M'))].loc[:, ["date", "close"]]
        train_data_ts = data_basic_info[(data_basic_info['date'].dt.strftime('%Y-%m-%d %H:%M') < self.simulation_start_date.strftime('%Y-%m-%d %H:%M'))].loc[:, ["date", "close"]]
        test_data_ts.to_sql(f"{symbol}_ts_test", self.engine, schema="simulation_ml_data", if_exists='replace', index=False)
        train_data_ts.to_sql(f"{symbol}_ts_train", self.engine, schema="simulation_ml_data", if_exists='replace', index=False)
        X_train_ts = train_data_ts[["close"]].reset_index(drop=True)
        X_test_ts = test_data_ts[["close"]].reset_index(drop=True)
        if self.ts_mode:
            params = ts_params_simulation

            if self.ts_model_type == 'XGBoost':
                ts_finance = XGBoostFinance()
            elif self.ts_model_type == 'Bi_LSTM':
                ts_finance = LSTMFinance()
            elif self.ts_model_type == 'Transformer':
                ts_finance = TransformerFinance()
            else:
                raise ValueError("Unsupported model type")
            
            X_train, y_train, X_test, y_test, scaler = ts_finance.data_creator_simulation(train_set=X_train_ts, test_set=X_test_ts, time_step=self.time_step, scale=True)
            model = ts_finance.create_model(X_data=X_train, y_data=y_train, time_step=self.time_step, **params[self.ts_model_type])
            forecast, forecast_scaled = ts_finance.forecast_data(data=X_test, model=model, scaler=scaler)
 
            ts_finance.visualize_result(forecast, train_data_ts, test_data_ts, self.ts_model_type, symbol=symbol, time_step=self.time_step)
            ts_finance.evaluate_result(y_test, forecast_scaled)
            perc_forecast, forecast = ts_finance.convert_2_percentage(forecast=forecast, close_reel=X_test_ts, time_step=self.time_step)
        else:
            perc_forecast = pd.Series(0, index=np.arange(test_data_ts.shape[0]))
        return perc_forecast
    
    def process_symbol(self, symbol):
        logger.info(f"For {symbol}, calculating started.")
        data_technical_info = get_technical_data_as_dataframe(schema_name="simulation_technical", table_name=symbol)
        data_basic_info = get_data_as_dataframe(schema_name="simulation_data", table_name=symbol)
        data_df = data_technical_info.merge(data_basic_info, on="date", how="left")
        data_df["percentage"] = data_df["percentage"].shift(-1)
        data_df = data_df.dropna().reset_index(drop=True)
        data = data_preprocess_service(data_df)

        test_data = data[(data['date'].dt.strftime('%Y-%m-%d %H:%M') >= self.simulation_start_date.strftime('%Y-%m-%d %H:%M')) & (data['date'].dt.strftime('%Y-%m-%d %H:%M') < self.simulation_end_date.strftime('%Y-%m-%d %H:%M'))]
        train_data = data[(data['date'].dt.strftime('%Y-%m-%d %H:%M') < self.simulation_start_date.strftime('%Y-%m-%d %H:%M'))]
        price_data = data_df[(data_df['date'].dt.strftime('%Y-%m-%d %H:%M') >= self.simulation_start_date.strftime('%Y-%m-%d %H:%M')) & (data_df['date'].dt.strftime('%Y-%m-%d %H:%M') < self.simulation_end_date.strftime('%Y-%m-%d %H:%M'))].loc[:,"close"]
        
        test_data.to_sql(f"{symbol}_test", self.engine, schema="simulation_ml_data", if_exists='replace', index=False)
        train_data.to_sql(f"{symbol}_train", self.engine, schema="simulation_ml_data", if_exists='replace', index=False)
        
        y_test = test_data["percentage"]
        X_test = test_data.drop(["date", "percentage"], axis=1)
        y_train = train_data["percentage"]
        X_train = train_data.drop(["date", "percentage"], axis=1)

        coin_based_model_predictions = self.get_coin_based_model_result(X_train, y_train, X_test)
        perc_forecast = self.get_ts_model_result(symbol, data_basic_info)

        if self.general_model_mode:
            try:
                general_model_predictions = pd.Series(self.general_model.predict(X_test)).reset_index(drop=True)
            except ValueError as e:
                general_model_predictions = pd.Series(0, index=np.arange(X_test.shape[0]))
                print("general model error")
        else:
            general_model_predictions = pd.Series(0, index=np.arange(X_test.shape[0]))
        date = test_data["date"].reset_index(drop=True)
        coin_based_model_predictions = coin_based_model_predictions.reset_index(drop=True)
        price_data = price_data.reset_index(drop=True)

        simulation_aux_data = pd.DataFrame({"date": date,
                                            "general_model": general_model_predictions,
                                            "coin_based_model": coin_based_model_predictions,
                                            "ts_model": perc_forecast,
                                            "close_price": price_data})
        simulation_aux_data["result"] = simulation_aux_data["general_model"] + simulation_aux_data["coin_based_model"] + simulation_aux_data["ts_model"]
        simulation_aux_data.to_sql(symbol, self.engine, schema="simulation_ml_signal", if_exists='replace', index=False)
        return {
            "date": date,
            "symbol": symbol,
            "general_model": general_model_predictions,
            "coin_based_model": coin_based_model_predictions,
            "ts_model": perc_forecast,
            "close_price": price_data
        }
    
    def run_simulation(self):
        if self.general_model_mode:
            self.get_general_model()
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self.process_symbol, symbol) for symbol in selected_coins.keys()]
            results = [future.result() for future in as_completed(futures)]

        iterations = zip(
            [result["general_model"] for result in results],
            [result["coin_based_model"] for result in results],
            [result["ts_model"] for result in results],
            [result["close_price"] for result in results],
            [result["symbol"] for result in results],
            [result["date"] for result in results]
        )
        match self.simulation_strategy_selection:
            case "strategy1":
                self.simulation_strategy.strategy1(iterations)
            case "strategy2":
                self.simulation_strategy.strategy2(iterations)
            case "strategy3":
                self.simulation_strategy.strategy3(iterations)
            case "strategy4":
                self.simulation_strategy.strategy4(iterations)