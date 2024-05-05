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

postgres = Postgres(database=database, user=user, password=password, host=host, port=port)
postgres.delete_all_tables()
postgres.close()