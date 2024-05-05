from app.general.ConfigManager import ConfigManager
from app.data.database import Postgres
from app.data.data import BorsaData
from app.data.parse_service import parse_data

config = ConfigManager()

database = config.db_config["database"]
user = config.db_config["user"]
password = config.db_config["password"]
host = config.db_config["host"]
port = config.db_config["port"]


def get_all_data(schema_name: str, table_name: str):
    postgres = Postgres(database=database, user=user, password=password, host=host, port=port)
    data = postgres.fetch_all(schema_name, table_name)
    postgres.close()

    borsa_data_list = []
    for row in data:
        parsed_data = parse_data(row)
        borsa_data = BorsaData(
            date=parsed_data.date,
            open=parsed_data.open,
            high=parsed_data.high,
            low=parsed_data.low,
            close=parsed_data.close,
            volume=parsed_data.volume,
            percentage=parsed_data.percentage
        )
        borsa_data_list.append(borsa_data)
    return borsa_data_list

def get_all_technical_data(schema_name: str, table_name: str):
    postgres = Postgres(database=database, user=user, password=password, host=host, port=port)
    data, column_names = postgres.fetch_all_with_column_names(schema_name, table_name)
    postgres.close()
    return data, column_names

def get_data(schema_name: str, table_name: str):
    postgres = Postgres(database=database, user=user, password=password, host=host, port=port)
    data, col_names = postgres.fetch_general(schema_name, table_name)
    postgres.close()
    return data, col_names