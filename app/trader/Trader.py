from app.general.ConfigManager import ConfigManager
from app.trader.binance_future_trader_service import BinanceFutureTrader
from default_lists import selected_coins
class Trader():
    def __init__(self, df, symbol_long_amount, symbol_short_amount):
        self.df = df
        self.symbol_long_amount = symbol_long_amount
        self.symbol_short_amount = symbol_short_amount
        config = ConfigManager()
        self.ML_SIGNAL_THRESHOLD = config.prediction_ml_signal_threshold
        self.ML_SIGNAL_THRESHOLD2 = config.prediction_ml_signal_threshold2
        self.binance_future_trader = BinanceFutureTrader()
        self.prediction_strategy_selection = config.prediction_strategy_selection
        
    def buy_sell(self):
        strong_buy_df = self.df[self.df["result"] >= self.ML_SIGNAL_THRESHOLD]
        weak_buy_df = self.df[(self.df["result"] < self.ML_SIGNAL_THRESHOLD) & (self.df["result"] >= self.ML_SIGNAL_THRESHOLD2)]
        weak_sell_df = self.df[(self.df["result"] > (-1 * self.ML_SIGNAL_THRESHOLD)) & (self.df["result"] <= self.ML_SIGNAL_THRESHOLD2)]
        strong_sell_df = self.df[self.df["result"] <= (-1 * self.ML_SIGNAL_THRESHOLD)]
        strong_buy_dict = strong_buy_df.to_dict(orient='records')
        weak_buy_dict = weak_buy_df.to_dict(orient='records')
        weak_sell_dict = weak_sell_df.to_dict(orient='records')
        strong_sell_dict = strong_sell_df.to_dict(orient='records')
        return strong_buy_dict, strong_sell_dict, weak_buy_dict, weak_sell_dict

    def create_order_strategy4(self, strong_buy_dict, strong_sell_dict, weak_buy_dict, weak_sell_dict):
        buy_amount = int(self.binance_future_trader.get_available_balance() / (len(selected_coins) + 1))
        for row in strong_buy_dict:
            if row["coin"] in self.symbol_short_amount.keys():
                quantity_for_pos_close = self.symbol_short_amount[row["coin"]]
                quantity = quantity_for_pos_close * 2
                self.binance_future_trader.long_order(symbol=row["coin"], quantity=quantity, pos="close_open")
            elif row["coin"] in self.symbol_long_amount.keys():
                pass
            else:
                quantity = self.binance_future_trader.calculate_quantity(row["coin"], buy_amount)
                self.binance_future_trader.long_order(symbol=row["coin"], quantity=quantity, pos="open")

        for row in strong_sell_dict:
            if row["coin"] in self.symbol_long_amount.keys():
                quantity_for_pos_close = self.symbol_long_amount[row["coin"]]
                quantity = quantity_for_pos_close * 2
                self.binance_future_trader.short_order(symbol=row["coin"], quantity=quantity, pos="close_open")
            elif row["coin"] in self.symbol_short_amount.keys():
                pass
            else:
                quantity = self.binance_future_trader.calculate_quantity(row["coin"], buy_amount)
                self.binance_future_trader.short_order(symbol=row["coin"], quantity=quantity, pos="open")
        
        if self.ML_SIGNAL_THRESHOLD2 != 0: # For Weak Signals
            for row in weak_buy_dict:
                if row["coin"] in self.symbol_short_amount.keys():
                    quantity_for_pos_close = self.symbol_short_amount[row["coin"]]
                    quantity = quantity_for_pos_close
                    self.binance_future_trader.long_order(symbol=row["coin"], quantity=quantity, pos="close")
                elif row["coin"] in self.symbol_long_amount.keys():
                    quantity_for_pos_close = self.symbol_long_amount[row["coin"]]
                    quantity = quantity_for_pos_close
                    self.binance_future_trader.short_order(symbol=row["coin"], quantity=quantity, pos="close")
            for row in weak_sell_dict:
                if row["coin"] in self.symbol_short_amount.keys():
                    quantity_for_pos_close = self.symbol_short_amount[row["coin"]]
                    quantity = quantity_for_pos_close
                    self.binance_future_trader.long_order(symbol=row["coin"], quantity=quantity, pos="close")
                elif row["coin"] in self.symbol_long_amount.keys():
                    quantity_for_pos_close = self.symbol_long_amount[row["coin"]]
                    quantity = quantity_for_pos_close
                    self.binance_future_trader.short_order(symbol=row["coin"], quantity=quantity, pos="close")



    def pred_order(self):
        match self.prediction_strategy_selection:
            case "strategy1":
                buy_dict, sell_dict = self.buy_sell()
                self.create_order_strategy1(buy_dict, sell_dict)
            case "strategy2":
                buy_dict, sell_dict = self.buy_sell()
                self.create_order_strategy2(buy_dict, sell_dict)
            case "strategy3":
                self.create_order_strategy3()
            case "strategy4":
                strong_buy_dict, strong_sell_dict, weak_buy_dict, weak_sell_dict = self.buy_sell()
                self.create_order_strategy4(strong_buy_dict, strong_sell_dict, weak_buy_dict, weak_sell_dict)