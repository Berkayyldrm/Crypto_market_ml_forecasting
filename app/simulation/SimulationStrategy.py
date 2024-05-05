import pandas as pd
from app.general.ConfigManager import ConfigManager
from app.general.CustomLogger import CustomLogger
from default_lists import selected_coins

logger = CustomLogger('simulation', 'log/simulation.log')
logger = logger.get_logger()

class SimulationStrategy():
    def __init__(self):
        self.asset_list = dict()
        for coin in selected_coins.keys():
            self.asset_list[coin] = {"amount": 0, "price": 0}
        config = ConfigManager()
        self.ML_SIGNAL_THRESHOLD = config.simulation_ml_signal_threshold
        self.ML_SIGNAL_THRESHOLD2 = config.simulation_ml_signal_threshold2
        self.BUDGET = config.simulation_budget / len(selected_coins)
        self.OLD_BUDGET = self.BUDGET
        self.FIRSTBUDGET = self.BUDGET
        self.MAXCOST = 0
        self.TRADE_COUNT = 0
        self.OLD_TRADE_COUNT = 0
        self.BUDGET_PERCENTAGE = {}
        self.engine = config.get_db_engine()

    def calc_last_budget(self, revenues, budget):
        last_budget = 0
        for coin in revenues.keys():
            last_budget += ((revenues[coin]/100) * budget) + budget
        return last_budget
            
    
    def strategy4(self, iterations):
        """
        open order when position closed - All in Strategy
        """
        logger.info(f"All in, open order when position closed strategy started.")
        for g, c, ts, p, symbol, d in iterations:
            df = pd.DataFrame(
            {'date': d,
            'coin': symbol,
            'general_ml_result': g,
            'coin_based_ml_result': c,
            'ts_result':ts,
            'close_price': p
            })
            df["open_price"] = df["close_price"].shift(1)
            df["general_ml_result"] = df["general_ml_result"].shift(1)
            df["coin_based_ml_result"] = df["coin_based_ml_result"].shift(1)
            df["result"] = df["general_ml_result"] + df["coin_based_ml_result"] + df["ts_result"]
            df = df.dropna().reset_index(drop=True)
            df.to_sql(symbol, self.engine, schema="simulation_strategy_data", if_exists='replace', index=False)
            logger.info(f"For {symbol}, simulation started.")
            for index, row in df.iterrows():
                if row["result"] >= self.ML_SIGNAL_THRESHOLD: # Long Signal
                    if self.asset_list[symbol]["amount"] < 0:
                        old_budget = self.asset_list[symbol]["amount"] * -1 * self.asset_list[symbol]["price"]
                        self.BUDGET += self.asset_list[symbol]["amount"] * (self.asset_list[symbol]["price"] - row["open_price"]) * -1 + old_budget
                        self.asset_list[symbol]["amount"] = 0
                        logger.info(f"{row['date']} -> Short Close: {self.BUDGET}, -> Trade Price: {row['open_price']}")

                        self.asset_list[symbol]["amount"] += self.BUDGET/row["open_price"]
                        self.asset_list[symbol]["price"] = row["open_price"]
                        self.BUDGET -= self.BUDGET
                        logger.info(f"{row['date']} -> Long Open: {self.BUDGET}, -> Trade Price: {row['open_price']}, {symbol} assets: {self.asset_list[symbol]}")
                        self.TRADE_COUNT += 2
                    elif self.asset_list[symbol]["amount"] > 0:
                        logger.info(f"{row['date']} -> Long Hold")
                    else:
                        self.asset_list[symbol]["amount"] += self.BUDGET/row["open_price"]
                        self.asset_list[symbol]["price"] = row["open_price"]
                        self.BUDGET -= self.BUDGET
                        logger.info(f"{row['date']} -> Long Open: {self.BUDGET}, -> Trade Price: {row['open_price']}, {symbol} assets: {self.asset_list[symbol]}")
                        self.TRADE_COUNT += 1

                elif row["result"] <= (-1 * self.ML_SIGNAL_THRESHOLD): # Short Signal
                    if self.asset_list[symbol]["amount"] > 0:
                        self.BUDGET += self.asset_list[symbol]["amount"] * row["open_price"] 
                        self.asset_list[symbol]["amount"] = 0
                        logger.info(f"{row['date']} -> Long Close: {self.BUDGET}, -> Trade Price: {row['open_price']}")

                        self.asset_list[symbol]["amount"] -= self.BUDGET/row["open_price"]
                        self.asset_list[symbol]["price"] = row["open_price"]
                        self.BUDGET -= self.BUDGET
                        logger.info(f"{row['date']} -> Short Open: {self.BUDGET}, -> Trade Price: {row['open_price']}, {symbol} assets: {self.asset_list[symbol]}")
                        self.TRADE_COUNT += 2
                    elif self.asset_list[symbol]["amount"] < 0:
                        logger.info(f"{row['date']} -> Short Hold")
                    else:
                        self.asset_list[symbol]["amount"] -= self.BUDGET/row["open_price"]
                        self.asset_list[symbol]["price"] = row["open_price"]
                        self.BUDGET -= self.BUDGET
                        logger.info(f"{row['date']} -> Short Open: {self.BUDGET}, -> Trade Price: {row['open_price']}, {symbol} assets: {self.asset_list[symbol]}")
                        self.TRADE_COUNT += 1
                    
                if self.ML_SIGNAL_THRESHOLD2 != 0: # For Weak Signals
                    if (row["result"] < self.ML_SIGNAL_THRESHOLD) & (row["result"] >= self.ML_SIGNAL_THRESHOLD2): # Weak Long Signal
                        logger.info(f"{row['date']} -> **********Weak Long Signal**********")
                        if self.asset_list[symbol]["amount"] < 0: # Close Short Position
                            old_budget = self.asset_list[symbol]["amount"] * -1 * self.asset_list[symbol]["price"]
                            self.BUDGET += self.asset_list[symbol]["amount"] * (self.asset_list[symbol]["price"] - row["open_price"]) * -1 + old_budget
                            self.asset_list[symbol]["amount"] = 0
                            logger.info(f"{row['date']} -> Short Close: {self.BUDGET}, -> Trade Price: {row['open_price']}")
                        elif self.asset_list[symbol]["amount"] > 0: # Close Long Position
                            self.BUDGET += self.asset_list[symbol]["amount"] * row["open_price"] 
                            self.asset_list[symbol]["amount"] = 0
                            logger.info(f"{row['date']} -> Long Close: {self.BUDGET}, -> Trade Price: {row['open_price']}")

                    elif (row["result"] <= (-1 * self.ML_SIGNAL_THRESHOLD2)) & (row["result"] > (-1 * self.ML_SIGNAL_THRESHOLD)): # Weak Short Signal
                        logger.info(f"{row['date']} -> **********Weak Short Signal**********")
                        if self.asset_list[symbol]["amount"] > 0: # Close Long Position
                            self.BUDGET += self.asset_list[symbol]["amount"] * row["open_price"] 
                            self.asset_list[symbol]["amount"] = 0
                            logger.info(f"{row['date']} -> Long Close: {self.BUDGET}, -> Trade Price: {row['open_price']}")
                        elif self.asset_list[symbol]["amount"] < 0: # Close Short Position
                            old_budget = self.asset_list[symbol]["amount"] * -1 * self.asset_list[symbol]["price"]
                            self.BUDGET += self.asset_list[symbol]["amount"] * (self.asset_list[symbol]["price"] - row["open_price"]) * -1 + old_budget
                            self.asset_list[symbol]["amount"] = 0
                            logger.info(f"{row['date']} -> Short Close: {self.BUDGET}, -> Trade Price: {row['open_price']}")

                if index == len(df) - 1:
                    if self.asset_list[symbol]["amount"] < 0:
                        old_budget = self.asset_list[symbol]["amount"] * -1 * self.asset_list[symbol]["price"]
                        self.BUDGET += self.asset_list[symbol]["amount"] * (self.asset_list[symbol]["price"] - row["open_price"]) * -1 + old_budget
                        logger.info(f"{row['date']} -> Last transaction for {symbol} Short closed: {self.BUDGET}, -> Trade Price: {row['open_price']}")
                        self.asset_list[symbol]["amount"] = 0
                        self.TRADE_COUNT += 1
                    else:
                        self.BUDGET += self.asset_list[symbol]["amount"] * row["open_price"] 
                        logger.info(f"{row['date']} -> Last transaction for {symbol} Long closed: {self.BUDGET}, -> Trade Price: {row['open_price']}")
                        self.asset_list[symbol]["amount"] = 0
                        self.TRADE_COUNT += 1
                    logger.info(f"TRADE COUNT FOR {symbol} -> {self.TRADE_COUNT - self.OLD_TRADE_COUNT}")
                    logger.info(f"BUDGET REVENUE PERCENT FOR {symbol} -> {((self.BUDGET * 100) / self.OLD_BUDGET) - 100}")
                    self.BUDGET_PERCENTAGE[symbol] = ((self.BUDGET * 100) / self.OLD_BUDGET) - 100
                    self.OLD_TRADE_COUNT = self.TRADE_COUNT
                    self.OLD_BUDGET = self.BUDGET
        logger.info("----------------------------------------------RESULTS----------------------------------------------------")        
        logger.info(f"TOTAL TRADE COUNT -> {self.TRADE_COUNT}")
        self.ESTIMATED_FEE = self.TRADE_COUNT * self.FIRSTBUDGET * 0.0005
        logger.info(f"Final Asset: {self.asset_list}")
        logger.info(f"BUDGET REVENUES: {self.BUDGET_PERCENTAGE}")
        last_budget = self.calc_last_budget(self.BUDGET_PERCENTAGE, self.FIRSTBUDGET)
        logger.info(f"LAST BUDGET: {last_budget}")
        #logger.info(f"Final BUDGET(NOT INCLUDED FEE): {self.BUDGET}")
        logger.info(f"ESTİMATED FEE -> {self.ESTIMATED_FEE}")
        logger.info("Simulation Mode Executed")