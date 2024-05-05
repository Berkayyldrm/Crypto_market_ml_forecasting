from matplotlib import pyplot as plt


class PlotTechnicalData():
    def __init__(self):
        pass

    def plot_stochrsi(self, close_prices, stochrsi_df):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Kapanış fiyatlarının çizilmesi
        ax1.plot(close_prices, label='Close Price')
        ax1.set_title('Close Prices')
        ax1.legend()

        # StochRSI ve sinyal çizgilerinin çizilmesi
        ax2.plot(stochrsi_df['STOCHRSI_k'], label='STOCHRSI_k')
        ax2.plot(stochrsi_df['STOCHRSI_d'], label='STOCHRSI_d')
        ax2.axhline(80, color='red', linestyle='--')  # Aşırı alım bölgesi
        ax2.axhline(20, color='green', linestyle='--') # Aşırı satım bölgesi
        ax2.set_title('Stochastic RSI')
        ax2.legend()

        plt.show()

    def plot_stoch(self, close_prices, stoch_df):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Kapanış fiyatlarının çizilmesi
        ax1.plot(close_prices, label='Close Price')
        ax1.set_title('Close Prices')
        ax1.legend()

        # Stoch ve sinyal çizgilerinin çizilmesi
        ax2.plot(stoch_df['STOCH_k'], label='STOCH_k')
        ax2.plot(stoch_df['STOCH_d'], label='STOCH_d')
        ax2.axhline(80, color='red', linestyle='--')  # Aşırı alım bölgesi
        ax2.axhline(20, color='green', linestyle='--') # Aşırı satım bölgesi
        ax2.set_title('Stochastic')
        ax2.legend()

        plt.show()

    def plot_macd(self, close_prices, macd_df):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Kapanış fiyatlarının çizilmesi
        ax1.plot(close_prices, label='Close Price')
        ax1.set_title('Close Prices')
        ax1.legend()

        # StochRSI ve sinyal çizgilerinin çizilmesi
        ax2.plot(macd_df['MACD'], label='MACD Line')
        ax2.plot(macd_df['MACD_s'], label='Signal Line')

        # MACD histogramının çizilmesi
        ax2.bar(macd_df.index, macd_df['MACD_h'], label='Histogram', color='grey')

        ax2.set_title('MACD Indicator')
        ax2.legend()

        plt.show()

    def plot_bbands(self, close_prices, bbands, period, std):
        _params = f"{period}_{std}"
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Kapanış fiyatlarının çizilmesi
        ax.plot(close_prices, label='Close Price', color='blue')

        # Bollinger Bantlarının çizilmesi
        ax.plot(bbands[f'BBL_{_params}'], label='Lower Bollinger Band', color='red')
        ax.plot(bbands[f'BBM_{_params}'], label='Middle Bollinger Band', color='black')
        ax.plot(bbands[f'BBU_{_params}'], label='Upper Bollinger Band', color='green')

        ax.set_title('Bollinger Bands')
        ax.legend()

        plt.show()

    def plot_kdj(self, close_prices, kdj, period, signal):
        _params = f"_{period}_{signal}"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(close_prices, label='Close Price')
        ax1.set_title('Close Prices')
        ax1.legend()

        ax2.plot(kdj[f'K{_params}'], label='K Line')
        ax2.plot(kdj[f'D{_params}'], label='D Line')
        ax2.plot(kdj[f'J{_params}'], label='J Line')
        ax2.axhline(80, color='red', linestyle='--')  # Aşırı alım bölgesi
        ax2.axhline(20, color='green', linestyle='--') # Aşırı satım bölgesi
        ax2.set_title('KDJ')
        ax2.legend()

        plt.show()

    def plot_ichimoku(self, close_prices, ichimoku_df):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Kapanış fiyatlarının çizilmesi
        ax.plot(close_prices, label='Close Price', color='black')

        # Ichimoku bileşenlerinin çizilmesi
        ax.plot(ichimoku_df['ITS_9'], label='Tenkan-sen', color='red', linestyle='--')
        ax.plot(ichimoku_df['IKS_26'], label='Kijun-sen', color='blue', linestyle='--')
        ax.plot(ichimoku_df['ISA_9'], label='Senkou Span A', color='green')
        ax.plot(ichimoku_df['ISB_26'], label='Senkou Span B', color='brown')

        # Chikou Span'ın çizilmesi
        ax.plot(ichimoku_df['ICS_26'], label='Chikou Span', color='purple', linestyle=':')

        # Ichimoku Bulutunun (Kumo) çizilmesi
        ax.fill_between(ichimoku_df.index, ichimoku_df['ISA_9'], ichimoku_df['ISB_26'], where=ichimoku_df['ISA_9'] >= ichimoku_df['ISB_26'], color='lightgreen', alpha=0.3)
        ax.fill_between(ichimoku_df.index, ichimoku_df['ISA_9'], ichimoku_df['ISB_26'], where=ichimoku_df['ISA_9'] < ichimoku_df['ISB_26'], color='lightcoral', alpha=0.3)

        ax.set_title('Ichimoku Cloud')
        ax.legend()

        plt.show()
    
    def plot_psar(self, close_prices, psar, params):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(close_prices, label='Close Price', color='black')

        ax.scatter(psar.index, psar[f'PSARl{params}'], label='Parabolic SAR Long', color='green', marker='o', s=1)

        ax.scatter(psar.index, psar[f'PSARs{params}'], label='Parabolic SAR Short', color='red', marker='x', s=1)

        ax.set_title('Parabolic SAR')
        ax.legend()

        plt.show()

    def plot_keltner_ch(self, close_prices, kc, params):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Kapanış fiyatlarının çizilmesi
        ax.plot(close_prices, label='Close Price', color='black')

        # Keltner Channel üst bantının çizilmesi
        ax.plot(kc[f"KCUe{params}"], label='Upper Band', color='green')

        # Keltner Channel orta bantının (hareketli ortalama) çizilmesi
        ax.plot(kc[f"KCBe{params}"], label='Middle Line', color='blue')

        # Keltner Channel alt bantının çizilmesi
        ax.plot(kc[f"KCLe{params}"], label='Lower Band', color='red')

        # Keltner Channel'ın görselleştirilmesi
        ax.fill_between(kc.index, kc[f"KCUe{params}"], kc[f"KCLe{params}"], color='grey', alpha=0.3)

        ax.set_title('Keltner Channel')
        ax.legend()

        plt.show()