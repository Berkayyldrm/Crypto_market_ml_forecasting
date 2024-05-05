import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import mean_absolute_error, mean_squared_error
from app.general.ConfigManager import ConfigManager
from abc import ABC, abstractmethod
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
import tensorflow as tf
from keras import layers
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd


class TimeSeriesFinance(ABC):
    def __init__(self):
        config = ConfigManager()
        self.engine = config.get_db_engine()

    @abstractmethod
    def create_model(self, data, order):
        pass

    @abstractmethod
    def forecast_data(self, data, model, scaler):
        pass

    def load_model(self, symbol):
        filename = f'app/ml_models/coin/ts_model_{symbol}.pkl'
        model = pickle.load(open(filename, 'rb'))
        return model

    def save_model(self, symbol, model):
        filename = f'app/ml_models/coin/ts_model_{symbol}.pkl'
        pickle.dump(model, open(filename, 'wb'))

    def load_scaler(self, symbol):
        filename = f'app/ml_models/coin/ts_model_scaler_{symbol}.pkl'
        scaler = pickle.load(open(filename, 'rb'))
        return scaler

    def save_scaler(self, symbol, scaler):
        filename = f'app/ml_models/coin/ts_model_scaler_{symbol}.pkl'
        pickle.dump(scaler, open(filename, 'wb'))
    
    def data_creator(self, data_set, time_step, scale=True):
        scaler = MinMaxScaler(feature_range=(0, 1))
        if scale:
            data_set = scaler.fit_transform(data_set['close'].values.reshape(-1,1))
        def create_dataset(data_set, time_step=1):
            dataX, dataY = [], []
            for i in range(len(data_set) - time_step):
                a = data_set[i:(i + time_step)]
                dataX.append(a)
                dataY.append(data_set[i + time_step])
            return np.array(dataX), np.array(dataY)

        X_data, y_data = create_dataset(data_set, time_step)
        return X_data, y_data, scaler
    
    def data_creator_prediction(self, data_set, time_step, scaler, scale=True):
        if scale:
            data_set_scaled = scaler.transform(data_set['close'].values.reshape(-1,1))
        def create_dataset(data_set, time_step=1):
            dataX = []
            for i in range(len(data_set) - time_step + 1):
                a = data_set[i:(i + time_step)]
                dataX.append(a)
            return np.array(dataX)

        X_data = create_dataset(data_set_scaled, time_step)
        return X_data
    
    def data_creator_simulation(self, train_set, test_set, time_step, scale=True):
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_set = train_set['close'].values.reshape(-1,1)
        test_set = test_set['close'].values.reshape(-1,1)
        if scale:
            scaler = scaler.fit(train_set)
            train_set = scaler.transform(train_set)
            test_set = scaler.transform(test_set)
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step):
                a = dataset[i:(i + time_step)]
                dataX.append(a)
                dataY.append(dataset[i + time_step])
            return np.array(dataX), np.array(dataY)
        
        X_train, y_train = create_dataset(train_set, time_step)
        X_test, y_test = create_dataset(test_set, time_step)
        return X_train, y_train, X_test, y_test, scaler
    
    def visualize_result(self, forecast, train_set, test_set, model_name, symbol, time_step=0):
        try:
            if forecast.shape[0] != test_set['date'].shape[0]:
                test_set = test_set.iloc[-forecast.shape[0]:, :]
        except:
            pass
        plt.figure(figsize=(15, 6))
        plt.plot(train_set['date'], train_set['close'], label='Real Close Values(Train)')
        plt.plot(test_set['date'], test_set['close'], label='Real Close Values(Test)')
        plt.plot(test_set['date'], forecast, label=f'Forecast Values {model_name}')
        if time_step != 0:
            plt.title(f'{symbol} Close Values (Train-Test Split (Time Step {time_step})')
        else:
            plt.title(f'{symbol} Close Values (Train-Test Split')
        plt.xlabel('Date')
        plt.ylabel('Close Value (USDT)')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
    
    def evaluate_result(self, real_value, forecasts):
        mae = mean_absolute_error(real_value, forecasts)
        rmse = mean_squared_error(real_value, forecasts, squared=False)
        print("mae: ", mae)
        print("rmse: ", rmse)
        return mae, rmse
    
    def convert_2_percentage(self, forecast, close_reel, time_step, prediction_mode=False):
        forecast = pd.Series(forecast.flatten())
        if not prediction_mode:
            nan_series = pd.Series([np.nan] * (time_step)) # Because first values empty for time step size
            forecast = pd.concat([nan_series, forecast], ignore_index=True)
        forecast = pd.concat([close_reel, forecast], axis=1)
        forecast.columns = ["close_reel", "forecast"]
        forecast.close_reel = forecast.close_reel.shift(1)
        perc_forecast = pd.DataFrame()
        perc_forecast['percentage'] = ((forecast['forecast'] - forecast['close_reel']) / forecast['close_reel']) * 100
        return perc_forecast["percentage"], forecast["forecast"]
    
    def get_additional_data_for_pred(self, train_data, pred_data):
        max_train_date = train_data['date'].max()
        if max_train_date not in pred_data['date'].values:
            raise ValueError(f"Date: {max_train_date} doesn't exist in prediction data. Please activate train mode.")
        additional_data = pred_data[pred_data['date'] > max_train_date]
        if additional_data.empty:
            raise ValueError(f"Please wait one hour or one minutes for prediction(Based on used interval time). Time series model is inactive")
        return additional_data

class XGBoostFinance(TimeSeriesFinance):
    def __init__(self):
        super().__init__()
    
    def create_model(self, X_data, y_data, time_step, **params):
        X_data = X_data.reshape(X_data.shape[0], -1)
        model_xgb_optimized = XGBRegressor(**params)
        model_xgb_optimized.fit(X_data, y_data)
        return model_xgb_optimized
    
    def forecast_data(self, data, model, scaler):
        data = data.reshape(data.shape[0], -1)
        forecast_scaled = model.predict(data)
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))
        return forecast, forecast_scaled

class LSTMFinance(TimeSeriesFinance):
    def __init__(self):
        super().__init__()
    
    def create_model(self, X_data, y_data, time_step, **params):
        learning_rate = params["learning_rate"]
        batch_size = params["batch_size"]
        units = params["units"]
        activation = params["activation"]
        optimizer = params["optimizer"]

        model_lstm = Sequential()
        model_lstm.add(Bidirectional(LSTM(units, activation=activation, return_sequences=False, input_shape=(X_data.shape[1], 1))))
        model_lstm.add(Dense(1, activation="sigmoid"))

        if optimizer == 'adam':
            model_lstm.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error", metrics=['mae'])
        else:
            model_lstm.compile(optimizer=SGD(learning_rate=learning_rate), loss="mean_squared_error", metrics=['mae'])

        model_lstm.fit(X_data, y_data, epochs=20, batch_size=batch_size)
        return model_lstm
    
    def forecast_data(self, data, model, scaler):
        forecast_scaled = model.predict(data)
        forecast = scaler.inverse_transform(forecast_scaled)
        return forecast, forecast_scaled   

class TransformerFinance(TimeSeriesFinance):
    def __init__(self):
        super().__init__()
    
    def create_model(self, X_data, y_data, time_step, **params):
        self.X_data = X_data
        self.y_data = y_data
        head_size = params["head_size"]
        num_heads = params["num_heads"]
        ff_dim = params["ff_dim"]
        nums_trans_blocks = params["nums_trans_blocks"]
        mlp_units = params["mlp_units"]
        dropout = params["dropout"]
        mlp_dropout = params["mlp_dropout"]
        lr = params["learning_rate"]
        batch_size = params["batch_size"]

        transformer = self.build_transformer(time_step=time_step, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, num_trans_blocks=nums_trans_blocks, mlp_units=mlp_units, mlp_dropout=mlp_dropout, dropout=dropout, attention_axes=1)
        early_stopping = EarlyStopping(monitor='loss', patience=5)
        transformer = self.fit_transformer(transformer, lr, batch_size, early_stopping)
        return transformer
    
    def forecast_data(self, data, model, scaler):
        forecast_scaled = model.predict(data)
        forecast = scaler.inverse_transform(forecast_scaled)
        return forecast, forecast_scaled
    
    def build_transformer(self, time_step, head_size, num_heads, ff_dim, num_trans_blocks, mlp_units, dropout=0, mlp_dropout=0, attention_axes=None, epsilon=1e-6, kernel_size=1):
        """
        Creates final model by building many transformer blocks.
        """
        n_timesteps, n_features, n_outputs = time_step, 1, 1
        inputs = tf.keras.Input(shape=(n_timesteps, n_features))
        x = inputs #+ positional_encoding(n_timesteps, n_features)
        for _ in range(num_trans_blocks):
            x = self.transformer_encoder(x, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, attention_axes=attention_axes, kernel_size=kernel_size, epsilon=epsilon)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)

        outputs = layers.Dense(n_outputs)(x)
        return tf.keras.Model(inputs, outputs)
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0, epsilon=1e-6, attention_axes=None, kernel_size=1):
        """
        Creates a single transformer block.
        """
        x = layers.LayerNormalization(epsilon=epsilon)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            attention_axes=attention_axes
            )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=epsilon)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=kernel_size, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=kernel_size)(x)
        return x + res
    
    def fit_transformer(self, transformer: tf.keras.Model, lr, batch_size, early_stopping):
        """
        Compiles and fits our transformer.
        """
        transformer.compile(
        loss="mae",
        optimizer=Adam(learning_rate=lr))

        transformer.fit(self.X_data, self.y_data, batch_size=batch_size, epochs=20, verbose=True, callbacks=[early_stopping])
        return transformer
