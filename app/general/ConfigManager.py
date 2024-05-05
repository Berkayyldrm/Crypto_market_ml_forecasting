import configparser
from sqlalchemy import create_engine
from binance.client import Client
from pandas import to_datetime
from google.cloud import secretmanager

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")

            self.db_config = self.config["postgresql"]
            self.database = self.db_config["database"]
            self.user = self.db_config["user"]
            self.password = self.db_config["password"]
            self.host = self.db_config["host"]
            self.port = self.db_config["port"]

            self.binance_config = self.config["binance"]
            #self.bnb_future_api_key, self.bnb_future_secret_key = self.gcloud_secret(self.binance_config["gcloud_project_id"], self.binance_config["gcloud_secret_id"], self.binance_config["gcloud_version_id"])
            self.bnb_future_api_key, self.bnb_future_secret_key = self.binance_config["future_api_key"], self.binance_config["future_secret_key"]

            self.parameters = self.config["parameters"]
            self.interval_type = self.parameters["interval_type"]
            self.lvrg = int(self.parameters["lvrg"])
            self.lag_features = self.parameters["lag_features"].lower() == "true"

            self.prediction = self.config["prediction"]
            self.prediction_mode = self.prediction["prediction_mode"].lower() == "true"
            self.prediction_ts_mode = self.prediction["ts_mode"].lower() == "true"
            self.prediction_coin_based_mode = self.prediction["coin_based_mode"].lower() == "true"
            self.prediction_general_model_mode = self.prediction["general_model_mode"].lower() == "true"
            self.prediction_strategy_selection = self.prediction["prediction_strategy_selection"]
            self.prediction_ml_signal_threshold = float(self.prediction["prediction_ml_signal_threshold"])
            self.prediction_ml_signal_threshold2 = float(self.prediction["prediction_ml_signal_threshold2"])

            self.train = self.config["train"]
            self.train_mode = self.train["train_mode"].lower() == "true"
            self.ts_model_type = self.train["ts_model_type"]
            self.time_step = int(self.train["time_step"])

            self.simulation = self.config["simulation"]
            self.simulation_mode = self.simulation["simulation_mode"].lower() == "true"
            self.simulation_mode_update = self.simulation["simulation_mode_update"].lower() == "true"
            self.start_date = to_datetime(self.simulation["start_date"])
            self.end_date = to_datetime(self.simulation["end_date"])
            self.simulation_budget = int(self.simulation["simulation_budget"])
            self.simulation_ts_mode = self.simulation["ts_mode"].lower() == "true"
            self.simulation_coin_based_mode = self.simulation["coin_based_mode"].lower() == "true"
            self.simulation_general_model_mode = self.simulation["general_model_mode"].lower() == "true"
            self.simulation_ts_model_type = self.simulation["simulation_ts_model_type"]
            self.simulation_time_step = int(self.simulation["simulation_time_step"])
            self.simulation_strategy_selection = self.simulation["simulation_strategy_selection"]
            self.simulation_ml_signal_threshold = float(self.simulation["simulation_ml_signal_threshold"])
            self.simulation_ml_signal_threshold2 = float(self.simulation["simulation_ml_signal_threshold2"])
            
            self._initialized = True
            
    def get_db_engine(self):
        self.engine = create_engine(f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}')
        return self.engine
    
    def get_future_client(self):
        self.client = Client(self.bnb_future_api_key, self.bnb_future_secret_key)
        return self.client
    
    def gcloud_secret(self, project_id, secret_id, version_id):
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(name=name)
        future_api_key, future_secret_key = response.payload.data.decode('UTF-8').split()
        return future_api_key.split("=")[1], future_secret_key.split("=")[1]