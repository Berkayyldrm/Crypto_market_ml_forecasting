import numpy as np
import pandas as pd

from app.general.CustomLogger import CustomLogger

logger = CustomLogger('general', 'log/general.log').get_logger()

class InterpretTechnicalData():
    def __init__(self):
        pass

    def rsi_interpret(self, series: pd.Series) -> np.ndarray:
        overbought = series > 70
        oversold = series < 30

        conditions = [overbought, oversold]
        choices = ["Overbought", "Oversold"]
        rsi_interpretation = np.select(conditions, choices, default="Neutral")

        return rsi_interpretation

    def stoch_interpret(self, df: pd.DataFrame) -> np.ndarray:
        buy = (df['STOCH_k'] < 20) & (df['STOCH_d'] < 20) & (df['STOCH_k'] >= df['STOCH_d']) & (df['STOCH_prev_k'] < df['STOCH_prev_d'])
        sell = (df['STOCH_k'] >= 80) & (df['STOCH_d'] >= 80) & (df['STOCH_k'] < df['STOCH_d']) & (df['STOCH_prev_k'] >= df['STOCH_prev_d']) 
        bullish_momentum = df['STOCH_k'] >= df['STOCH_d']
        bearish_momentum = df['STOCH_k'] < df['STOCH_d']

        conditions = [buy, sell, bullish_momentum, bearish_momentum]
        choices = ["Buy", "Sell", "Bullish-Momentum", "Bearish-Momentum"]
        stoch_interpretation = np.select(conditions, choices, default=None)
        return stoch_interpretation
    
    def stochrsi_interpret(self, df: pd.DataFrame) -> np.ndarray:
        buy = (df['STOCHRSI_k'] < 20) & (df['STOCHRSI_d'] < 20) & (df['STOCHRSI_k'] >= df['STOCHRSI_d']) & (df['STOCHRSI_prev_k'] < df['STOCHRSI_prev_d'])
        sell = (df['STOCHRSI_k'] >= 80) & (df['STOCHRSI_d'] >= 80) & (df['STOCHRSI_k'] < df['STOCHRSI_d']) & (df['STOCHRSI_prev_k'] >= df['STOCHRSI_prev_d']) 
        bullish_momentum = df['STOCHRSI_k'] >= df['STOCHRSI_d']
        bearish_momentum = df['STOCHRSI_k'] < df['STOCHRSI_d']

        conditions = [buy, sell, bullish_momentum, bearish_momentum]
        choices = ["Buy", "Sell", "Bullish-Momentum", "Bearish-Momentum"]
        stochrsi_interpretation = np.select(conditions, choices, default=None)
        return stochrsi_interpretation
    
    def macd_interpret(self, df: pd.DataFrame) -> np.ndarray:
        buy = (df["MACD"] >= df["MACD_s"]) & (df["MACD_prev"] < df["MACD_s_prev"])
        sell = (df["MACD"] <= df["MACD_s"]) & (df["MACD_prev"] > df["MACD_s_prev"])

        bullish_momentum = (df["MACD"] >= df["MACD_s"])
        bearish_momentum = (df["MACD"] < df["MACD_s"])

        conditions = [buy, sell, bullish_momentum, bearish_momentum]
        choices = ["Buy", "Sell", "Bullish-Momentum", "Bearish-Momentum"]
        macd_interpretation = np.select(conditions, choices, default=None)
        return macd_interpretation
    
    def bbands_interpret(self, df: pd.DataFrame, period: int, std: float, threshold: float) -> np.ndarray:
        threshold_1 = threshold
        _params = f"_{period}_{std}"
        bullish_tred = df[f"BBP{_params}"] > 1.0  
        bearish_trend = df[f"BBP{_params}"] < 0.0  
        buy = (df[f"BBP{_params}"] >= 0.0) & (df[f"BBP{_params}"].shift(1) < 0.0)
        sell = (df[f"BBP{_params}"] < 1.0) & (df[f"BBP{_params}"].shift(1) >= 1.0) 
        stabilized = ((df[f"BBP{_params}"] < threshold_1) & (df[f"BBP{_params}"].shift(1) >= threshold_1)) |((df[f"BBP{_params}"] >= threshold_1) & (df[f"BBP{_params}"].shift(1) < threshold_1))

        conditions = [bullish_tred, bearish_trend, buy, sell, stabilized]
        choices = ["Bullish-Trend", "Bearish-Trend", "Buy", "Sell", "Stabilized"]
        bollinger_interpretation = np.select(conditions, choices, default="Neutral")
        return bollinger_interpretation
        
    def adx_interpret(self, df: pd.DataFrame) -> np.ndarray:
        threshold = 25
        buy = (df["DMP"] >= df["DMN"]) & (df["ADX"] >= threshold)
        sell = (df["DMN"] > df["DMP"]) & (df["ADX"] >= threshold)
        neutral_buy = (df["DMP"] >= df["DMN"]) & (df["ADX"] < threshold)
        neutral_sell = (df["DMN"] > df["DMP"]) & (df["ADX"] < threshold)

        conditions = [buy, sell, neutral_buy, neutral_sell]
        choices = ["Buy", "Sell", "Neutral-Buy", "Neutral-Sell"]
        adx_interpretation = np.select(conditions, choices, default=None)
        return adx_interpretation
    
    def williams_r_interpret(self, series: pd.Series) -> np.ndarray:
        overbought = series > -20
        oversold = series < -80

        conditions = [overbought, oversold]
        choices = ["Overbought", "Oversold"]
        adx_interpretation = np.select(conditions, choices, default="Neutral")
        return adx_interpretation
    
    def cci_interpret(self, series: pd.Series) -> np.ndarray:
        overbought = series > 100
        oversold = series < -100

        conditions = [overbought, oversold]
        choices = ["Overbought", "Oversold"]
        cci_interpretation = np.select(conditions, choices, default="Neutral")
        return cci_interpretation
    
    def uo_interpret(self, series: pd.Series) -> np.ndarray:
        overbought = series > 70
        oversold = series < 30

        conditions = [overbought, oversold]
        choices = ["Overbought", "Oversold"]
        uo_interpretation = np.select(conditions, choices, default="Neutral")
        return uo_interpretation
    
    def bull_bear_power_interpret(self, series: pd.Series) -> np.ndarray:
        bullish_momentum = (series > 0) & (series.diff() > 0)
        bearish_momentum = (series < 0) & (series.diff() < 0)

        conditions = [bullish_momentum, bearish_momentum]
        choices = ["Bullish-Momentum", "Bearish-Momentum"]
        bbp_interpretation = np.select(conditions, choices, default="Neutral")
        return bbp_interpretation
    
    def ma_interpret(self, ma_short: pd.Series, ma_long: pd.Series) -> pd.Series:
        buy = (ma_short >= ma_long)
        sell = (ma_short < ma_long)
        conditions = [buy, sell]
        choices = ["Buy", "Sell"]
        ma_interpretation = np.select(conditions, choices, default=None)
        return ma_interpretation
        
    def pivot_interpret(self, df: pd.DataFrame):
        conditions = [
            df['Current_Price'] >= df['Resistance_Level_2'],
            df['Current_Price'] >= df['Resistance_Level_1'],
            df['Current_Price'] >= df['Pivot'],
            df['Current_Price'] >= df['Support_Level_1'],
            df['Current_Price'] >= df['Support_Level_2']
        ]
        choices = ['Strong-Sell', 'Sell', 'Neutral-Sell', 'Neutral-Buy', 'Buy']
        pivot_interpretation = np.select(conditions, choices, default='Strong-Buy')
        return pivot_interpretation
    
    def demark_pivot_interpret(self, df: pd.DataFrame):
        conditions = [
            df['Current_Price'] >= df['Resistance_Level_1'],
            df['Current_Price'] >= df['Pivot'],
            df['Current_Price'] >= df['Support_Level_1'],
        ]
        choices = ['Sell', 'Neutral-Sell', 'Neutral-Buy']
        demark_pivot_interpretation = np.select(conditions, choices, default='Buy')
        return demark_pivot_interpretation
    
    def kdj_interpret(self, df: pd.DataFrame, period: int, signal: int) -> np.ndarray:
        _params = f"_{period}_{signal}"
        buy = (df[f"J{_params}"] < 20) & (df[f"K{_params}"] >= df[f"D{_params}"]) & (df[f"K{_params}"].shift(1) < df[f"D{_params}"].shift(1))
        sell = (df[f"J{_params}"] > 80) & (df[f"K{_params}"] < df[f"D{_params}"]) & (df[f"K{_params}"].shift(1) >= df[f"D{_params}"].shift(1))
        bullish_momentum = df[f"K{_params}"] >= df[f"D{_params}"]
        bearish_momentum = df[f"K{_params}"] < df[f"D{_params}"]
        
        conditions = [buy, sell, bullish_momentum, bearish_momentum]
        choices = ["Buy", "Sell", "Bullish-Momentum", "Bearish-Momentum"]
        kdj_interpretation = np.select(conditions, choices, default=None)
        return kdj_interpretation

    def psar_interpret(self, psar_df: pd.DataFrame, df: pd.DataFrame, af0: float, max_af: float) -> np.ndarray:
        _params = f"_{af0}_{max_af}"
        buy = psar_df[f"PSARl{_params}"] < df['close']
        sell = psar_df[f"PSARs{_params}"] >= df['close']

        conditions = [buy, sell]
        choices = ["Buy", "Sell"]
        psar_interpretation = np.select(conditions, choices, default=None)
        return psar_interpretation

    def obv_interpret(self, series: pd.Series, close: pd.Series) -> np.ndarray:
        buy = (series >= series.shift(1)) & (close >= close.shift(1))
        sell = (series < series.shift(1)) & (close < close.shift(1))

        conditions = [buy, sell]
        choices = ["Buy", "Sell"]
        obv_interpretation = np.select(conditions, choices, default=None)
        return obv_interpretation

    def cmf_interpret(self, series: pd.Series) -> np.ndarray:
        bullish_momentum = (series > 0) & (series.diff() > 0)
        bearish_momentum = (series < 0) & (series.diff() < 0)

        conditions = [bullish_momentum, bearish_momentum]
        choices = ["Bullish-Momentum", "Bearish-Momentum"]
        cmf_interpretation = np.select(conditions, choices, default="Neutral")
        return cmf_interpretation
    
    def ao_interpret(self, series: pd.Series) -> np.ndarray:
        bullish_momentum = (series > 0) & (series.diff() > 0)
        bearish_momentum = (series < 0) & (series.diff() < 0)

        conditions = [bullish_momentum, bearish_momentum]
        choices = ["Bullish-Momentum", "Bearish-Momentum"]
        ao_interpretation = np.select(conditions, choices, default="Neutral")
        return ao_interpretation
    
    def keltner_ch_interpret(self, series: pd.Series, threshold: float) -> np.ndarray:
        threshold_1 = threshold
        
        bullish_tred = series > 1.0  
        bearish_trend = series < 0.0  
        buy = (series >= 0.0) & (series.shift(1) < 0.0)
        sell = (series < 1.0) & (series >= 1.0) 
        stabilized = ((series < threshold_1) & (series.shift(1) >= threshold_1)) |((series >= threshold_1) & (series.shift(1) < threshold_1))

        conditions = [bullish_tred, bearish_trend, buy, sell, stabilized]
        choices = ["Bullish-Trend", "Bearish-Trend", "Buy", "Sell", "Stabilized"]
        kc_interpretation = np.select(conditions, choices, default="Neutral")
        return kc_interpretation
    
    def super_trend_interpret(self, super_trend: pd.Series, close: pd.Series) -> np.ndarray:
        buy = super_trend > 0
        sell = super_trend < 0

        conditions = [buy, sell]
        choices = ["Buy", "Sell"]
        super_trend_interpretation = np.select(conditions, choices, default="Neutral")
        return super_trend_interpretation