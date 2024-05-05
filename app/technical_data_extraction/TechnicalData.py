import numpy as np
import pandas_ta as ta
import pandas as pd
import concurrent.futures
from app.general import CustomLogger

from default_lists import selected_coins
from app.general.CustomLogger import CustomLogger
from app.general.ConfigManager import ConfigManager
from app.technical_data_extraction.InterpretTechnicalData import InterpretTechnicalData
from app.technical_data_extraction.PlotTechnicalData import PlotTechnicalData
from app.data.dataframe_service import get_data_as_dataframe
from default_lists import parameters as p

logger = CustomLogger('general', 'log/general.log').get_logger()

class TechnicalData():
    def __init__(self, **kwargs):
        self.prediction_mode = kwargs.get("prediction_mode", None)
        self.simulation_mode = kwargs.get("simulation_mode", None)
        self.simulation_mode_update = kwargs.get("simulation_mode_update", None)
        config = ConfigManager()
        self.engine = config.get_db_engine()
        self.interval_type = config.interval_type
        self.technical_interpreter = InterpretTechnicalData()
        self.technical_plotter = PlotTechnicalData()

    def get_coin_data(self, symbol: str):
        if self.simulation_mode and self.simulation_mode_update:
            self.data = get_data_as_dataframe(schema_name="simulation_data", table_name=symbol)
        else:
            if self.prediction_mode:
                self.data = get_data_as_dataframe(schema_name="pred_data", table_name=symbol)
            else:
                self.data = get_data_as_dataframe(schema_name="train_data", table_name=symbol)
        
        self.get_general_technical_indicators(symbol)
    
    def get_general_technical_indicators(self, symbol: str):
        rsi = self.rsi_calc(period=p["rsi_period"])
        stoch = self.stoch_calc(k=p["stoch_k_period"], d=p["stoch_d"], smooth_k=p["stoch_smooth_k"])
        stochrsi = self.stochrsi_calc(period=p["stochrsi_period"], rsi_period=p["stochrsi_rsi_period"], k=p["stochrsi_k"], d=p["stochrsi_d"])
        macd = self.macd_calc(fast_period=p["macd_fast_period"], slow_period=p["macd_slow_period"], signal_period=p["macd_signal_period"])
        bbands = self.bbands_calc(period=p["bbands_period"], std=p["bbands_std"], threshold=p["bbands_threshold"])
        adx = self.adx_calc(period=p["adx_period"])
        williams_r = self.williams_r_calc(period=p["willr_period"])
        cci = self.cci_calc(period=p["cci_period"], c=p["cci_c"])
        atr = self.atr_calc(period=p["atr_period"])
        uo = self.uo_calc(short_period=p["uo_short_period"], medium_period=p["uo_medium_period"], long_period=p["uo_long_period"])
        bull_bear_power = self.bull_bear_power_calc(period=p["bullbear_period"])
        ma_1 = self.ma_calc(p["ma_short_period"], p["ma_mid_period"], p["ma_long_period"])
        ma_2 = self.ma_mix_calc(p["ma_short_period"], p["ma_mid_period"])
        classic_pivot, classic_pivot_levels, classic_pivot_2, classic_pivot_levels_2 = self.classic_pivot_calc(p["pivot_period"])
        fibonacci_pivot, fibonacci_pivot_levels, fibonacci_pivot_2, fibonacci_pivot_levels_2 = self.fibonacci_pivot_calc(p["pivot_period"])
        kdj = self.kdj_calc(period=p["kdj_period"], signal=p["kdj_signal_period"])
        psar = self.psar_calc(af=p["psar_af"], max_af=p["psar_max_af"])
        obv = self.obv_calc()
        cmf = self.cmf_calc(period=p["cmf_period"])
        ao = self.awesome_osc_calc(fast_period=p["ao_fast_period"], slow_period=p["ao_slow_period"])
        kc = self.keltner_ch_calc(period=p["kc_period"], scalar=p["kc_scalar"], threshold=p["kc_threshold"])
        super_trend = self.super_trend(period=p["super_trend_period"], multiplier=p["super_trend_multiplier"])
        df = pd.concat([self.data["date"], rsi, stoch, stochrsi, macd, bbands, adx, williams_r, cci, atr, uo, bull_bear_power,
                        ma_1, ma_2, classic_pivot, fibonacci_pivot, classic_pivot_2, fibonacci_pivot_2, 
                        kdj, psar, obv, cmf, ao, kc, super_trend], axis=1)
        df_pivot = pd.concat([self.data["date"], classic_pivot_levels, fibonacci_pivot_levels, 
                              classic_pivot_levels_2, fibonacci_pivot_levels_2], axis=1)
        self.df_to_sql(df, symbol)
        self.df_pivot_to_sql(df_pivot, symbol)

    def df_to_sql(self, df: pd.DataFrame, symbol:str):
        if self.simulation_mode and self.simulation_mode_update:
            df.to_sql(symbol, self.engine, schema="simulation_technical", if_exists='replace', index=False)
        else:
            if self.prediction_mode:
                df.to_sql(symbol, self.engine, schema="pred_technical", if_exists='replace', index=False)
            else:
                df.to_sql(symbol, self.engine, schema="train_technical", if_exists='replace', index=False)

    def df_pivot_to_sql(self, df: pd.DataFrame, symbol:str):
        if self.simulation_mode and self.simulation_mode_update:
            df.to_sql(f"{symbol}_pivot", self.engine, schema="simulation_technical", if_exists='replace', index=False)
        else:
            if self.prediction_mode:
                df.to_sql(f"{symbol}_pivot", self.engine, schema="pred_technical", if_exists='replace', index=False)
            else:
                df.to_sql(f"{symbol}_pivot", self.engine, schema="train_technical", if_exists='replace', index=False)

    def pivot_helper(self, df):
        selected_columns = df.columns
        df['hour'] = df['date'].dt.hour
        df['dt'] = df['date'].dt.date
        for col in ['Support_Level_1', 'Support_Level_2', 'Support_Level_3', 'Pivot', 'Resistance_Level_1', 'Resistance_Level_2', 'Resistance_Level_3']:
            try:
                two_am_values = df[df['hour'] == 23].set_index('dt')[col]
                df['update_key'] = df['dt'].where(df['hour'] >= 23, df['dt'] - pd.Timedelta(days=1))
                df[col] = df['update_key'].map(two_am_values)
            except:
                pass
        return df[selected_columns].drop(["date"], axis=1)

    def rsi_calc(self, period: int) -> pd.DataFrame:
        rsi = ta.rsi(self.data['close'], length=period)
        rsi_interpretation = self.technical_interpreter.rsi_interpret(rsi)
        rsi = pd.DataFrame({"RSI": rsi, "RSI_I": rsi_interpretation}).rename(columns={f"RSI_{period}": "RSI"})
        return rsi
    
    def stoch_calc(self, k: int, d: int, smooth_k: int) -> pd.DataFrame:
        stoch = ta.stoch(self.data["high"], self.data["low"], self.data["close"], k=k, d=d, smooth_k=smooth_k)
        stoch_k = stoch['STOCHk_{}_{}_{}'.format(k, d, smooth_k)]
        stoch_d = stoch['STOCHd_{}_{}_{}'.format(k, d, smooth_k)]
        stoch_prev_k = stoch['STOCHk_{}_{}_{}'.format(k, d, smooth_k)].shift(1)
        stoch_prev_d = stoch['STOCHd_{}_{}_{}'.format(k, d, smooth_k)].shift(1)

        stoch_df = pd.concat([
            stoch_k.rename('STOCH_k'),
            stoch_d.rename('STOCH_d'),
            stoch_prev_k.rename('STOCH_prev_k'),
            stoch_prev_d.rename('STOCH_prev_d')
        ], axis=1)

        #self.technical_plotter.plot_stoch(self.data["close"], stoch_df)
        stoch_interpretation = self.technical_interpreter.stoch_interpret(stoch_df)
        stoch = pd.DataFrame(stoch_interpretation, columns=["STOCH_I"], index=stoch_df.index)
        return stoch
        
    def stochrsi_calc(self, period: int, rsi_period: int, k: int, d: int) -> pd.DataFrame:
        stochrsi = ta.stochrsi(close=self.data['close'], rsi_length=rsi_period, length=period, k=k, smooth_d=d)
        stochrsi_k = stochrsi['STOCHRSIk_{}_{}_{}_{}'.format(rsi_period, period, k, d)]
        stochrsi_d = stochrsi['STOCHRSId_{}_{}_{}_{}'.format(rsi_period, period, k, d)]
        stochrsi_prev_k = stochrsi['STOCHRSIk_{}_{}_{}_{}'.format(rsi_period, period, k, d)].shift(1)
        stochrsi_prev_d = stochrsi['STOCHRSId_{}_{}_{}_{}'.format(rsi_period, period, k, d)].shift(1)

        stochrsi_df = pd.concat([
            stochrsi_k.rename('STOCHRSI_k'),
            stochrsi_d.rename('STOCHRSI_d'),
            stochrsi_prev_k.rename('STOCHRSI_prev_k'),
            stochrsi_prev_d.rename('STOCHRSI_prev_d')
        ], axis=1)
        #self.technical_plotter.plot_stochrsi(self.data["close"], stochrsi_df)
        stochrsi_interpretation = self.technical_interpreter.stochrsi_interpret(stochrsi_df)
        stochrsi = pd.DataFrame(stochrsi_interpretation, columns=["STOCHRSI_I"], index=stochrsi_df.index)
        return stochrsi

    def macd_calc(self, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
        macd = ta.macd(self.data["close"], fast=fast_period, slow=slow_period, signal=signal_period)
        macd_ = macd[f'MACD_{fast_period}_{slow_period}_{signal_period}']
        macd_h = macd[f'MACDh_{fast_period}_{slow_period}_{signal_period}']
        macd_s = macd[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
        macd_prev = macd[f'MACD_{fast_period}_{slow_period}_{signal_period}'].shift(1)
        macd_s_prev = macd[f'MACDs_{fast_period}_{slow_period}_{signal_period}'].shift(1)

        macd_df = pd.concat([
            macd_.rename('MACD'),
            macd_h.rename('MACD_h'),
            macd_s.rename('MACD_s'),
            macd_prev.rename('MACD_prev'),
            macd_s_prev.rename('MACD_s_prev')
        ], axis=1)

        #self.technical_plotter.plot_macd(self.data["close"], macd_df)
        macd_interpretation = self.technical_interpreter.macd_interpret(macd_df)
        macd = pd.DataFrame(macd_interpretation, columns=["MACD_I"])
        return macd
    
    def bbands_calc(self, period: int, std: float, threshold: float) -> pd.DataFrame:
        bbands = ta.bbands(self.data["close"], length=period, std=std)
        #self.technical_plotter.plot_bbands(self.data["close"], bbands, period, std)
        bbands_df = bbands[[f"BBP_{period}_{std}"]]
        bbands_interpretation = self.technical_interpreter.bbands_interpret(bbands_df, period, std, threshold)
        bbands = pd.DataFrame({"BBANDS_I": bbands_interpretation, "BBANDS_Bandwidth": bbands[f"BBB_{period}_{std}"]})
        return bbands
    
    def adx_calc(self, period: int) -> pd.DataFrame:
        adx = ta.adx(high=self.data['high'], low=self.data['low'], close=self.data['close'], length=period)
        
        adx_df = pd.DataFrame({
            'ADX': adx[f"ADX_{period}"],
            'DMP': adx[f"DMP_{period}"],
            'DMN': adx[f"DMN_{period}"]
        })
        adx_numerical = np.where(adx_df['DMP'] > adx_df['DMN'], adx_df['ADX'], -adx_df['ADX'])
        adx_interpretation = self.technical_interpreter.adx_interpret(adx_df)
        adx = pd.DataFrame({"ADX": adx_numerical, "ADX_I": adx_interpretation})
        return adx
    
    def williams_r_calc(self, period: int) -> pd.DataFrame:
        williams_r = ta.willr(high=self.data['high'], low=self.data['low'], close=self.data['close'], length=period)
        williams_r_interpretation = self.technical_interpreter.williams_r_interpret(williams_r)
        williams_r = pd.DataFrame({"WILLR": williams_r, "WILLR_I": williams_r_interpretation})
        return williams_r

    def cci_calc(self, period: int, c: float) -> pd.DataFrame:
        cci = ta.cci(high=self.data['high'], low=self.data['low'], close=self.data['close'], length=period, c=c)
        cci_interpretation = self.technical_interpreter.cci_interpret(cci)
        cci = pd.DataFrame({"CCI": cci, "CCI_I": cci_interpretation})
        return cci

    def atr_calc(self, period: int) -> pd.DataFrame:
        atr = ta.atr(high=self.data['high'], low=self.data['low'], close=self.data['close'], length=period, percent=False)
        atr = pd.DataFrame({"ATR_Percentage": atr})
        return atr
    
    def uo_calc(self, short_period: int, medium_period: int, long_period: int) -> pd.DataFrame:
        uo = ta.uo(high=self.data['high'], low=self.data['low'], close=self.data['close'], fast=short_period, medium=medium_period, slow=long_period)
        uo_interpretation = self.technical_interpreter.uo_interpret(uo)
        uo = pd.DataFrame({"UO": uo, "UO_I": uo_interpretation})
        return uo
    
    def bull_bear_power_calc(self, period: int) -> pd.DataFrame:
        bull_power = self.data["high"] - ta.ema(self.data["close"], length=period)
        bear_power = self.data["low"] - ta.ema(self.data["close"], length=period)
        bbp = bull_power + bear_power
        bbp_interpretation = self.technical_interpreter.bull_bear_power_interpret(bbp)
        bbp = pd.DataFrame({"BBP": bbp, "BBP_I": bbp_interpretation})
        return bbp
    
    def ma_calc(self, short_period: int, mid_period: int, long_period: int) -> pd.DataFrame:
        sma_short = ta.sma(self.data["close"], length=short_period)
        sma_mid = ta.sma(self.data["close"], length=mid_period)
        sma_long = ta.sma(self.data["close"], length=long_period)

        ma_interpretation_sm = self.technical_interpreter.ma_interpret(sma_short, sma_mid)
        ma_interpretation_sl = self.technical_interpreter.ma_interpret(sma_short, sma_long)
        ma_interpretation_ml = self.technical_interpreter.ma_interpret(sma_mid, sma_long)
        ma = pd.DataFrame({"MA_sm_I": ma_interpretation_sm, "MA_sl_I": ma_interpretation_sl, "MA_ml_I": ma_interpretation_ml})
        return ma
    
    def ma_mix_calc(self, short_period: int, long_period: int) -> pd.DataFrame:
        sma_long = ta.sma(self.data["close"], length=long_period)
        ema_short = ta.ema(self.data["close"], length=short_period)

        ma_interpretation = self.technical_interpreter.ma_interpret(ema_short, sma_long)
        ma = pd.DataFrame({"MA_mix_I": ma_interpretation})
        return ma
    
    def classic_pivot_calc(self, period: int) -> pd.DataFrame:
        high_max = self.data['high'].rolling(window=period).max()
        low_min = self.data['low'].rolling(window=period).min()
        close_last = self.data['close']

        pivot = (high_max + low_min + close_last) / 3
        r1 = 2 * pivot - low_min
        r2 = pivot + (high_max - low_min)
        r3 = high_max + 2 * (pivot - low_min)
        s1 = 2 * pivot - high_max
        s2 = pivot - (high_max - low_min)
        s3 = low_min - 2 * (high_max - pivot)

        classic_pivot_levels = pd.DataFrame({
            'Support_Level_1': s1,
            'Support_Level_2': s2,
            'Support_Level_3': s3,
            'Pivot': pivot,
            'Resistance_Level_1': r1,
            'Resistance_Level_2': r2,
            'Resistance_Level_3': r3,
            'Current_Price': close_last,
            'date': self.data["date"]
        })
        classic_pivot_4h = self.technical_interpreter.pivot_interpret(classic_pivot_levels)
        classic_pivot_4h = pd.DataFrame(classic_pivot_4h, columns=["Classic_Pivot_4H"])
        classic_pivot_levels_4h = classic_pivot_levels.add_prefix('Classic_4H_')

        classic_pivot_levels_1d = self.pivot_helper(classic_pivot_levels)
        classic_pivot_1d = self.technical_interpreter.pivot_interpret(classic_pivot_levels_1d)
        classic_pivot_1d = pd.DataFrame(classic_pivot_1d, columns=["Classic_Pivot_1D"])
        classic_pivot_levels_1d = classic_pivot_levels_1d.add_prefix('Classic_1D_')

        return classic_pivot_1d, classic_pivot_levels_1d, classic_pivot_4h, classic_pivot_levels_4h

    def fibonacci_pivot_calc(self, period: int) -> pd.DataFrame:
        high = self.data['high'].rolling(window=period).max()
        low = self.data['low'].rolling(window=period).min()
        close_last = self.data['close']

        pivot = (high + low + close_last) / 3
        r1 = pivot + (0.382 * (high - low))
        r2 = pivot + (0.618 * (high - low))
        r3 = pivot + (1 * (high - low))
        s1 = pivot - (0.382 * (high - low))
        s2 = pivot - (0.618 * (high - low))
        s3 = pivot - (1 * (high - low))

        fibonacci_pivot_levels = pd.DataFrame({
            'Support_Level_1': s1,
            'Support_Level_2': s2,
            'Support_Level_3': s3,
            'Pivot': pivot,
            'Resistance_Level_1': r1,
            'Resistance_Level_2': r2,
            'Resistance_Level_3': r3,
            'Current_Price': close_last,
            'date': self.data["date"]
        })
        fibonacci_pivot_levels_1d = self.pivot_helper(fibonacci_pivot_levels)
        fibonacci_pivot_1d = self.technical_interpreter.pivot_interpret(fibonacci_pivot_levels_1d)
        fibonacci_pivot_1d = pd.DataFrame(fibonacci_pivot_1d, columns=["Fibonacci_Pivot_1D"])
        fibonacci_pivot_levels_1d = fibonacci_pivot_levels_1d.add_prefix('Fibonacci_1D_')

        fibonacci_pivot_4h = self.technical_interpreter.pivot_interpret(fibonacci_pivot_levels)
        fibonacci_pivot_4h = pd.DataFrame(fibonacci_pivot_4h, columns=["Fibonacci_Pivot_4H"])
        fibonacci_pivot_levels_4h = fibonacci_pivot_levels.add_prefix('Fibonacci_4H_')
        return fibonacci_pivot_1d, fibonacci_pivot_levels_1d, fibonacci_pivot_4h, fibonacci_pivot_levels_4h
    
    
    def kdj_calc(self, period: int, signal: int) -> pd.DataFrame:
        kdj = ta.kdj(high=self.data["high"], low=self.data["low"], close=self.data["close"], length=period, signal=signal)
        _params = f"_{period}_{signal}"
        #self.technical_plotter.plot_kdj(self.data["close"], kdj, period, signal)
        kdj_interpretation = self.technical_interpreter.kdj_interpret(kdj, period=period, signal=signal)
        kdj = pd.DataFrame({"KDJ": kdj[f"J{_params}"], "KDJ_I": kdj_interpretation})
        return kdj
    
    def psar_calc(self, af: float, max_af: float) -> pd.DataFrame:
        psar = ta.psar(high=self.data["high"], low=self.data["low"], close=self.data["close"], af=af, max_af=max_af)
        _params = f"_{af}_{max_af}"
        #self.technical_plotter.plot_psar(self.data["close"], psar, _params)
        psar_interpretation = self.technical_interpreter.psar_interpret(psar, self.data, af, max_af)
        psar = pd.DataFrame(psar_interpretation, columns=["PSAR_I"])
        return psar
    
    def obv_calc(self) -> pd.DataFrame:
        obv = ta.obv(close=self.data["close"], volume=self.data["volume"])
        obv_interpretation = self.technical_interpreter.obv_interpret(obv, self.data["close"])
        obv = pd.DataFrame({"OBV": obv, "OBV_I": obv_interpretation})
        return obv

    def cmf_calc(self, period: int) -> pd.DataFrame:
        cmf = ta.cmf(high=self.data["high"], low=self.data["low"], close=self.data["close"], volume=self.data["volume"], open_=self.data["open"], length=period)
        cmf_interpretation = self.technical_interpreter.cmf_interpret(cmf)
        cmf = pd.DataFrame({"CMF": cmf, "CMF_I": cmf_interpretation})
        return cmf

    def awesome_osc_calc(self, fast_period: int, slow_period: int) -> pd.DataFrame:
        ao = ta.ao(high=self.data["high"], low=self.data["low"], fast=fast_period, slow=slow_period)
        ao_interpretation = self.technical_interpreter.ao_interpret(ao)
        ao = pd.DataFrame({"AO": ao, "AO_I": ao_interpretation})
        return ao

    def keltner_ch_calc(self, period: int, scalar: float, threshold: float) -> pd.DataFrame:
        kc = ta.kc(high=self.data["high"], low=self.data["low"], close=self.data["close"], length=period, scalar=scalar)
        _params = f"_{period}_{scalar}"
        #self.technical_plotter.plot_keltner_ch(self.data["close"], kc, _params)
        kc_percent = ta.non_zero_range(self.data["close"], kc[f"KCLe{_params}"]) / ta.non_zero_range(kc[f"KCUe{_params}"], kc[f"KCLe{_params}"])
        kc_interpretation = self.technical_interpreter.keltner_ch_interpret(kc_percent, threshold)
        kc = pd.DataFrame({"KC": kc_percent, "KC_I": kc_interpretation})
        return kc
    
    def super_trend(self, period: int, multiplier: float) -> pd.DataFrame:
        super_trend = ta.supertrend(high=self.data['high'], low=self.data['low'], close=self.data['close'], length=period, multiplier=multiplier)
        super_trend_interpretation = self.technical_interpreter.super_trend_interpret(super_trend[f"SUPERTd_{period}_{multiplier}"], self.data['close'])
        return pd.DataFrame({"SuperTrend_I": super_trend_interpretation})
    
    def extract_data(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor: # Don't change
            future_to_symbol = {executor.submit(self.get_coin_data, symbol): symbol for symbol in selected_coins.keys()}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                     data = future.result()
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    logger.info(f'An error occured for {symbol} in data extraction service: {e}')