import sys
sys.path.append('./')
from datetime import datetime
import numpy as np
import pandas as pd
from app.general.ConfigManager import ConfigManager
from default_lists import top_n_coin
config = ConfigManager()
client = config.get_future_client()
engine = config.get_db_engine()
exchange_info = client.futures_exchange_info()

def get_info(symbol, exchange_info, client):
    info = {
        'date': None,
        'symbol': None,
        'quantityPrecision': None,
        'stepSize': None,
        'minQty': None,
        'minNotional': None,
        'currentPrice': None
    }
    for symbol_info in exchange_info['symbols']:
        if symbol_info['symbol'] == symbol:
            info['quantityPrecision'] = symbol_info['quantityPrecision']
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'MARKET_LOT_SIZE':
                    info['minQty'] = filter['minQty']
                    info['stepSize'] = filter['stepSize']
                elif filter['filterType'] == 'MIN_NOTIONAL':
                    info['minNotional'] = filter['notional']
            break  
    ticker = client.futures_symbol_ticker(symbol=symbol)
    if ticker:
        info['currentPrice'] = ticker['price']
    info["symbol"] = symbol
    info["date"] = datetime.now()
    return info

def add_minimum_dollar_value(info):
    minQty = float(info['minQty'])
    currentPrice = float(info['currentPrice'])
    minNotional = float(info['minNotional'])

    calculatedValue = minQty * currentPrice

    minimum_dollar_value = max(calculatedValue, minNotional)

    info['minimumDollarValue'] = minimum_dollar_value
    info['minimumQuantity'] = np.round(info["minimumDollarValue"]/float(info["currentPrice"]), info["quantityPrecision"])
    return info

def process_df(df):
    df[["quantityPrecision", "minQty", "minNotional", "currentPrice", "stepSize",  "minimumDollarValue",  "minimumQuantity"]] = df[["quantityPrecision", "minQty", "minNotional", "currentPrice", "stepSize",  "minimumDollarValue",  "minimumQuantity"]].astype("float")
    for index, row in df.iterrows():
        min_qty = float(row['minimumQuantity'])
        current_price = float(row['currentPrice'])
        min_dollar_value = float(row['minimumDollarValue'])
        step_size = float(row['stepSize'])

        if min_qty * current_price < min_dollar_value:
            min_qty += step_size
            new_min_dollar_value = min_qty * current_price
            df.at[index, 'minimumQuantity'] = min_qty
            df.at[index, 'newMinimumDollarValue'] = new_min_dollar_value
        else:
            new_min_dollar_value = min_qty * current_price
            df.at[index, 'newMinimumDollarValue'] = new_min_dollar_value
    return df

def process_symbols(symbols, exchange_info, client):
    all_data = []

    for symbol in symbols:
        info = get_info(symbol, exchange_info, client)
        info = add_minimum_dollar_value(info)
        all_data.append(info)
    df = pd.DataFrame(all_data)
    df = process_df(df)
    return pd.DataFrame(df)


df = process_symbols(top_n_coin, exchange_info, client)

df.to_sql("quantities", engine, schema="quantity", if_exists='replace', index=False)