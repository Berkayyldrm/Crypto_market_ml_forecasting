import numpy as np
from binance.enums import *
from app.general.ConfigManager import ConfigManager
from datetime import datetime
import pandas as pd
from app.general.CustomLogger import CustomLogger
from default_lists import selected_coins
import concurrent.futures

logger = CustomLogger('general', 'log/general.log').get_logger()

class BinanceFutureTrader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BinanceFutureTrader, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance

    def __init__(self):
        # Initialize the client with API keys from config file
        # Initialization code will only run once
        if not hasattr(self, '_initialized'): # Ensures __init__ only runs once
            config = ConfigManager()
            self.engine = config.get_db_engine()
            self.leverage = config.lvrg
            self.client = config.get_future_client()
            self._initialized = True

    def long_order(self, symbol, quantity=None, pos="none"):
        """Create a market long order."""
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
            try:
                self.client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
            except: # NO_NEED_TO_CHANGE_MARGIN_TYPE
                pass
            if quantity < 0:
                quantity *= -1
            order = self.client.futures_create_order(
                            symbol=symbol,
                            side=SIDE_BUY,
                            type=ORDER_TYPE_MARKET,
                            quantity=quantity
                            )
            if pos == "close_open":# Position Close
                transaction = pd.DataFrame([[datetime.now(), symbol, "Short Closed - Long Opened", self.calculate_price(symbol), quantity]], columns=["date", "symbol", "type", "price", "quantity"])
            elif pos == "close":
                transaction = pd.DataFrame([[datetime.now(), symbol, "Short Closed", self.calculate_price(symbol), quantity]], columns=["date", "symbol", "type", "price", "quantity"])
            else:
                transaction = pd.DataFrame([[datetime.now(), symbol, "Long", self.calculate_price(symbol), quantity]], columns=["date", "symbol", "type", "price", "quantity"])
            transaction.to_sql("transactions", self.engine, schema="wallet", if_exists='append', index=False)
            return order
        
        except Exception as e:
            logger.info(f"An order error occurred for {symbol} type: Long, quantity: {quantity} -> {e}")
            return None

    def short_order(self, symbol, quantity=None, pos="none"):
        """Create a market short order."""
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
            try:
                self.client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
            except: # NO_NEED_TO_CHANGE_MARGIN_TYPE
                pass
            if quantity < 0:
                quantity *= -1
            order = self.client.futures_create_order(
                            symbol=symbol,
                            side=SIDE_SELL,
                            type=ORDER_TYPE_MARKET,
                            quantity=quantity
                            )
            if pos == "close_open": # Position Close
                transaction = pd.DataFrame([[datetime.now(), symbol, "Long Closed - Short Opened", self.calculate_price(symbol), quantity]], columns=["date", "symbol", "type", "price", "quantity"])
            elif pos == "close":
                transaction = pd.DataFrame([[datetime.now(), symbol, "Long Closed", self.calculate_price(symbol), quantity]], columns=["date", "symbol", "type", "price", "quantity"])
            else:
                transaction = pd.DataFrame([[datetime.now(), symbol, "Short", self.calculate_price(symbol), quantity]], columns=["date", "symbol", "type", "price", "quantity"])
            transaction.to_sql("transactions", self.engine, schema="wallet", if_exists='append', index=False)
            return order
        
        except Exception as e:
            logger.info(f"An order error occurred for {symbol} type: Short, quantity: {quantity} -> {symbol}: {e}")
            return None
        
    def calculate_quantity(self, symbol, amount):
        price = self.client.futures_symbol_ticker(symbol=symbol)["price"]
        price = float(price)
        quantity = amount/price
        quantity = np.round(quantity, selected_coins[symbol])
        return quantity
    
    def calculate_price(self, symbol):
        price = self.client.futures_symbol_ticker(symbol=symbol)["price"]
        return float(price)
    
    def get_positions_helper(self, symbol):
        positions = self.client.futures_position_information(symbol=symbol)
        long_amount = 0
        short_amount = 0
        if positions:
            for pos in positions:
                position_amount = float(pos["positionAmt"])
                if position_amount > 0:
                    long_amount = position_amount
                elif position_amount < 0:
                    short_amount = position_amount
        return symbol, long_amount, short_amount
    
    def get_positions(self):
        symbol_long_amount = dict()
        symbol_short_amount = dict()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self.get_positions_helper, symbol) for symbol in selected_coins.keys()]
            for future in concurrent.futures.as_completed(futures):
                symbol, long_amount, short_amount = future.result()
                if long_amount != 0:
                    symbol_long_amount[symbol] = long_amount
                if short_amount != 0:
                    symbol_short_amount[symbol] = short_amount

        return symbol_long_amount, symbol_short_amount
    
    def get_available_balance(self):
        balances = self.client.futures_account_balance()
        usdt_available_balance = next((item['balance'] for item in balances if item['asset'] == 'USDT'), None)
        return float(usdt_available_balance)