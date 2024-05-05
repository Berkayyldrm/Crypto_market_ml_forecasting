import sys
sys.path.append('./')
from app.general.ConfigManager import ConfigManager
from app.trader.binance_future_trader_service import BinanceFutureTrader

config = ConfigManager()
client = config.get_future_client()
binance_future_trader = BinanceFutureTrader()

symbol_long_amount, symbol_short_amount = binance_future_trader.get_positions()

for row in symbol_long_amount.keys():
    quantity = symbol_long_amount[row]
    binance_future_trader.short_order(symbol=row, quantity=quantity)

for row in symbol_short_amount:
    quantity = symbol_short_amount[row]
    binance_future_trader.long_order(symbol=row, quantity=quantity)