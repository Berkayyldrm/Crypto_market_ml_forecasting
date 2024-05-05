import pandas as pd
import xgboost as xgb
import pickle
from app.general.ConfigManager import ConfigManager
from app.ml.data_preprocess_service import data_preprocess_service
from app.data.dataframe_service import get_technical_data_as_dataframe, get_data_as_dataframe
from default_lists import selected_coins
import warnings
warnings.filterwarnings('ignore')

def ml_model_service():
    config = ConfigManager()
    engine = config.get_db_engine()
    data_df = pd.DataFrame()

    for symbol in selected_coins.keys():
        data_technical_info = get_technical_data_as_dataframe(schema_name="train_technical", table_name=symbol)
        data_basic_info = get_data_as_dataframe(schema_name="train_data", table_name=symbol)
        data_temp_df = data_technical_info.merge(data_basic_info, on="date", how="left")
        data_temp_df["percentage"] = data_temp_df["percentage"].shift(-1)
        data_df = pd.concat([data_df, data_temp_df]).reset_index(drop=True)
        data_df = data_df.dropna()

    data = data_preprocess_service(data_df)
    
    data.to_sql("general", engine, schema="train_ml_data", if_exists='replace', index=False)
    y = data["percentage"]
    X = data.drop(["date", "percentage"], axis=1)

    xgb_clf = xgb.XGBRegressor()
    xgb_clf.fit(X, y)

    filename = 'app/ml_models/coin/general_ml.pkl'
    pickle.dump(xgb_clf, open(filename, 'wb'))
        
