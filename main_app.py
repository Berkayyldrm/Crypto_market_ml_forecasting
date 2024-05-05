from app.general.CustomLogger import CustomLogger
from app.technical_data_extraction.TechnicalData import TechnicalData
from app.data_extraction.DataExtractionService import DataExtractionService
from app.trader.MlSignalCalculator import MlSignalCalculator
from app.simulation.SimulationService import SimulationService
from app.trader.Trader import Trader
from app.trader.binance_future_trader_service import BinanceFutureTrader
from datetime import datetime, timedelta
import asyncio
from app.ml.ml_model_service import ml_model_service
from app.ml.ml_model_stock_based_service import ml_model_stock_based_service
from app.general.ConfigManager import ConfigManager
from app.time_series.time_series_service import time_series_service

logger = CustomLogger('general', 'log/general.log').get_logger()

config = ConfigManager()
INTERVAL_TYPE = config.interval_type
SIMULATION_MODE = config.simulation_mode
SIMULATION_MODE_UPDATE = config.simulation_mode_update
SIMULATION_START_DATE = config.start_date
SIMULATION_END_DATE = config.end_date
PREDICTION_MODE = config.prediction_mode
TRAIN_MODE = config.train_mode

binance_future_trader = BinanceFutureTrader()

if SIMULATION_MODE_UPDATE and not SIMULATION_MODE:
    raise ValueError("SIMULATION_MODE_UPDATE is True but SIMULATION_MODE is False. Exiting...")

if sum([PREDICTION_MODE, SIMULATION_MODE, TRAIN_MODE]) != 1:
    raise ValueError("Only one of PREDICTION_MODE, SIMULATION_MODE, or TRAIN_MODE should be True. Exiting...")

async def main():
    if TRAIN_MODE:
        logger.info("Training Mode Active")
        data_extraction_service = DataExtractionService(prediction_mode=PREDICTION_MODE, interval_type=INTERVAL_TYPE, simulation_mode=SIMULATION_MODE, simulation_mode_update=SIMULATION_MODE_UPDATE)
        technical_data_service = TechnicalData(prediction_mode=PREDICTION_MODE, simulation_mode=SIMULATION_MODE, simulation_mode_update=SIMULATION_MODE_UPDATE)
        logger.info("Data Extraction Task Starting")
        data_extraction_service.extract_data()
        logger.info("Data Extraction Task Executed")
        logger.info("Technical Data Tasks Starting")
        technical_data_service.extract_data()
        logger.info("Technical Data Task Executed")
        logger.info("ML Model Task Starting")
        ml_model_service()
        logger.info("ML Model Task Executed")
        logger.info("ML Stock Based Model Task Starting")
        ml_model_stock_based_service()
        logger.info("ML Stock Based Model Task Executed")
        logger.info("Time Series Model Task Starting")
        time_series_service()
        logger.info("Time Series Model Task Executed")
        logger.info("Training Mode End")

    if SIMULATION_MODE:
        if SIMULATION_MODE_UPDATE:
            logger.info("Simulation Update Mode Active")
            data_extraction_service = DataExtractionService(prediction_mode=PREDICTION_MODE, interval_type=INTERVAL_TYPE, simulation_mode=SIMULATION_MODE, simulation_mode_update=SIMULATION_MODE_UPDATE)
            technical_data_service = TechnicalData(prediction_mode=PREDICTION_MODE, simulation_mode=SIMULATION_MODE, simulation_mode_update=SIMULATION_MODE_UPDATE)
            logger.info("Data Extraction Task Starting")
            data_extraction_service.extract_data()
            logger.info("Data Extraction Task Executed")
            logger.info("Technical Data Task Starting")
            technical_data_service.extract_data()
            logger.info("Technical Data Task Executed")
            logger.info("Simulation Task Starting")
            simulation_service = SimulationService(simulation_start_date=SIMULATION_START_DATE, simulation_end_date=SIMULATION_END_DATE)
            simulation_service.run_simulation()
            logger.info("Simulation Task Executed")
        else:
            logger.info("Simulation Mode Active")
            logger.info("Simulation Task Starting")
            simulation_service = SimulationService(simulation_start_date=SIMULATION_START_DATE, simulation_end_date=SIMULATION_END_DATE)
            simulation_service.run_simulation()
            logger.info("Simulation Task Executed")
            logger.info("Simulation Mode End")

    if PREDICTION_MODE:
        logger.info("Prediction Mode Active")
        while True:
            now = datetime.now()
            if INTERVAL_TYPE == 'hourly':
                next_target_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            elif INTERVAL_TYPE == 'minutely':
                next_target_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            elif INTERVAL_TYPE == "15minutes":
                minutes_to_add = 15 - (now.minute % 15)
                next_target_time = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
            elif INTERVAL_TYPE == "4hours":
                hours_to_add = 4 - (now.hour % 4)
                next_target_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
            sleep_seconds = (next_target_time - now).total_seconds()
            await asyncio.sleep(sleep_seconds)

            data_extraction_service = DataExtractionService(prediction_mode=PREDICTION_MODE, interval_type=INTERVAL_TYPE, simulation_mode=SIMULATION_MODE, simulation_mode_update=SIMULATION_MODE_UPDATE)
            technical_data_service = TechnicalData(prediction_mode=PREDICTION_MODE, simulation_mode=SIMULATION_MODE, simulation_mode_update=SIMULATION_MODE_UPDATE)
            ml_signal_calculator = MlSignalCalculator()
            logger.info("Data Extraction Task Starting")
            data_extraction_service.extract_data()
            logger.info("Data Extraction Task Executed")
            logger.info("Technical Data Task Starting")
            technical_data_service.extract_data()
            logger.info("Technical Data Task Executed")
            logger.info("Scraping Positions Task Starting")
            symbol_long_amount, symbol_short_amount = binance_future_trader.get_positions()
            logger.info("Scraping Positions Task Executed")
            logger.info("ML Signal Task Starting")
            df = ml_signal_calculator.calculate_signals()
            logger.info("ML Signal Task Executed")
            trader = Trader(df, symbol_long_amount, symbol_short_amount)
            #trader.pred_order()
            logger.info("Orders Completed")

if __name__ == "__main__":
    asyncio.run(main())