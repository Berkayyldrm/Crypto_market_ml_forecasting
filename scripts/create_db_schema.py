import sys
sys.path.append('./')
from app.general.ConfigManager import ConfigManager
from app.data.database import Postgres

config = ConfigManager()

database = config.db_config["database"]
user = config.db_config["user"]
password = config.db_config["password"]
host = config.db_config["host"]
port = config.db_config["port"]

schema_list = ["pred_data", "pred_ml_signal", "pred_technical",
               "simulation_data", "simulation_ml_data", "simulation_ml_signal",
               "simulation_strategy_data", "simulation_technical",
               "train_data", "train_ml_data", "train_technical",
               "wallet", "quantity", "train_ts_data"]

postgres = Postgres(database=database, user=user, password=password, host=host, port=port)
postgres.create_schemas(schema_list)
postgres.close()