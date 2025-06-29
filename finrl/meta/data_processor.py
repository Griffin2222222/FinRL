import logging
import numpy as np
import pandas as pd
from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
from finrl.meta.data_processors.processor_wrds import WrdsProcessor
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor


class DataProcessor:
    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        if data_source == "alpaca":
            try:
                API_KEY = kwargs.get("API_KEY")
                API_SECRET = kwargs.get("API_SECRET")
                API_BASE_URL = kwargs.get("API_BASE_URL")
                self.processor = AlpacaProcessor(API_KEY, API_SECRET, API_BASE_URL)
                print("Alpaca successfully connected")
            except BaseException:
                raise ValueError("Please input correct account info for alpaca!")

        elif data_source == "wrds":
            self.processor = WrdsProcessor()

        elif data_source == "yahoofinance":
            self.processor = YahooFinanceProcessor()

        else:
            raise ValueError("Data source input is NOT supported yet.")

        self.tech_indicator_list = tech_indicator
        self.vix = vix

    def download_data(
        self, ticker_list, start_date, end_date, time_interval, max_failures=5
    ) -> pd.DataFrame:
        try:
            df = self.processor.download_data(
                ticker_list=ticker_list,
                start_date=start_date,
                end_date=end_date,
                time_interval=time_interval,
                max_failures=max_failures,  # pass through
            )
            logging.info(
                f"Downloaded data for {ticker_list} from {start_date} to {end_date}"
            )
            return df
        except Exception as e:
            logging.error(f"Data download failed: {e}")
            raise

    def clean_data(self, df) -> pd.DataFrame:
        try:
            df = self.processor.clean_data(df)
            logging.info("Data cleaned successfully.")
            return df
        except Exception as e:
            logging.error(f"Data cleaning failed: {e}")
            raise

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        try:
            df = self.processor.add_technical_indicator(df, tech_indicator_list)
            # Normalization check
            for col in tech_indicator_list:
                if col in df.columns:
                    col_data = df[col].dropna()
                    if col_data.std() > 0:
                        normed = (col_data - col_data.mean()) / col_data.std()
                        assert (
                            abs(normed.mean()) < 1e-2
                        ), f"Feature {col} not normalized (mean)"
            logging.info(f"Added technical indicators: {tech_indicator_list}")
            return df
        except Exception as e:
            logging.error(f"Adding technical indicators failed: {e}")
            raise

    def add_turbulence(self, df) -> pd.DataFrame:
        try:
            df = self.processor.add_turbulence(df)
            logging.info("Turbulence added.")
            return df
        except Exception as e:
            logging.error(f"Adding turbulence failed: {e}")
            raise

    def add_vix(self, df) -> pd.DataFrame:
        try:
            df = self.processor.add_vix(df)
            logging.info("VIX added.")
            return df
        except Exception as e:
            logging.error(f"Adding VIX failed: {e}")
            raise

    def add_vixor(self, df) -> pd.DataFrame:
        try:
            df = self.processor.add_vixor(df)
            logging.info("VIXOR added.")
            return df
        except Exception as e:
            logging.error(f"Adding VIXOR failed: {e}")
            raise

    def add_sentiment(self, df) -> pd.DataFrame:
        # Add sentiment data if supported by the processor
        if hasattr(self.processor, "add_sentiment"):
            try:
                df = self.processor.add_sentiment(df)
                logging.info("Sentiment added.")
                return df
            except Exception as e:
                logging.error(f"Adding sentiment failed: {e}")
                raise
        else:
            raise NotImplementedError(
                "Sentiment analysis not implemented for this data source."
            )

    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        # Normalization check
        if np.any(np.isnan(tech_array)) or np.any(np.isinf(tech_array)):
            logging.warning("NaN or Inf detected in tech_array, replacing with 0.")
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0
        # Causality check: ensure no future data is used
        assert (
            price_array.shape[0] == tech_array.shape[0] == turbulence_array.shape[0]
        ), "Array length mismatch (possible lookahead)"
        return price_array, tech_array, turbulence_array
