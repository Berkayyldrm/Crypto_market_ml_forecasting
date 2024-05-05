import pandas as pd
from app.general.ConfigManager import ConfigManager
from app.general.CustomLogger import CustomLogger
from default_lists import selected_coins, interval_type_dict
import pytz
import concurrent.futures

logger = CustomLogger('general', 'log/general.log').get_logger()

class DataExtractionService:
    def __init__(self, **kwargs):
        self.prediction_mode = kwargs.get("prediction_mode", None)
        self.interval_type = kwargs.get("interval_type", None)
        self.simulation_mode = kwargs.get("simulation_mode", None)
        self.simulation_mode_update = kwargs.get("simulation_mode_update", None)
        self.config = ConfigManager()
        self.engine = self.config.get_db_engine()
        self.client = self.config.get_future_client()

    @staticmethod
    def utc_to_local(utc_dt, local_tz):
        local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
        return local_tz.normalize(local_dt)

    def get_coin_data(self, symbol):
        if self.simulation_mode and self.simulation_mode_update:
            coin = self.client.futures_historical_klines(symbol, interval_type_dict[self.interval_type]["start_interval"], interval_type_dict[self.interval_type]["start_str_simulation"], end_str=interval_type_dict[self.interval_type]["end_str"], limit=1500)
        else:
            if self.prediction_mode:
                coin = self.client.futures_historical_klines(symbol, interval_type_dict[self.interval_type]["start_interval"], interval_type_dict[self.interval_type]["start_str_prediction"], end_str=interval_type_dict[self.interval_type]["end_str"], limit=1500)
            else:
                coin = self.client.futures_historical_klines(symbol, interval_type_dict[self.interval_type]["start_interval"], interval_type_dict[self.interval_type]["start_str_train"], end_str=interval_type_dict[self.interval_type]["end_str"], limit=1500) 
        self.process_data(coin, symbol)

    def process_data(self, coin, symbol):
        coin = pd.DataFrame(coin, columns=["Date", "Open", "High", "Low", "Close", "Volume", "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"])
        coin = coin.iloc[:, :6]
        coin.Date = pd.to_datetime(coin.Date, unit='ms')
        coin['Date'] = coin['Date'].apply(lambda x: self.utc_to_local(x, local_tz=pytz.timezone("Europe/Istanbul")))
        coin['Date'] = coin['Date'].dt.tz_localize(None)
        coin = coin.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float'})
        coin["Volume"] = coin["Volume"].apply(lambda x: int(x))

        if self.prediction_mode:
            coin['percentage'] = 0
        else:
            coin['percentage'] = coin['Close'].pct_change() * 100
            coin = coin.dropna()
        
        coin.columns = ["date", "open", "high", "low", "close", "volume", "percentage"]
        if self.simulation_mode and self.simulation_mode_update:
            coin.to_sql(symbol, self.engine, schema="simulation_data", if_exists='replace', index=False)
        else:
            if self.prediction_mode:
                coin.to_sql(symbol, self.engine, schema="pred_data", if_exists='replace', index=False)
            else:
                coin.to_sql(symbol, self.engine, schema="train_data", if_exists='replace', index=False)
        return coin
    
    def extract_data(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_symbol = {executor.submit(self.get_coin_data, symbol): symbol for symbol in selected_coins.keys()}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                except Exception as e:
                    logger.info(f'An error occured for {symbol} in data extraction service: {e}')
