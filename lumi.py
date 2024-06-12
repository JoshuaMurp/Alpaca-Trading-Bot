import alpaca_trade_api as tradeapi
import pandas as pd 
from alpaca.data import TimeFrame
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta
import time



API_KEY = "PK9NMDGNF0EHKDM4NA4A" 
API_SECRET = "XqJsIBj8a5suSbsvpZyRAPrKonE4F4MfZAmmujkU" 
BASE_URL = "https://paper-api.alpaca.markets"



def initialize_api(api_key, api_secret, base_url='https://paper-api.alpaca.markets'):
    """
    Initialize and return the Alpaca API client.
    """
    return tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

class TradingBot:
    def __init__(self, api_key, api_secret, base_url='https://paper-api.alpaca.markets'):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.trading_client = TradingClient(api_key, api_secret, paper=True)

    def get_current_price(self, ticker):
        try:
            latest_trade = self.api.get_latest_trade(ticker)
            current_price = latest_trade.price
            return current_price
        except tradeapi.rest.APIError as api_error:
            print(f"API error fetching current price for {ticker}: {api_error}")
            return None
        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
            return None

    def get_historical_data(self, ticker, start_date, end_date, timeframe=TimeFrame.Minute, limit=1000):
        try:
            barset = self.api.get_bars(
                symbol=ticker,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                limit=limit
            ).df

            if not barset.empty:
                return barset
            else:
                print(f"No historical data found for {ticker} between {start_date} and {end_date}.")
                return None

        except tradeapi.rest.APIError as api_error:
            print(f"API error fetching historical data for {ticker}: {api_error}")
            return None
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return None


    def get_volume(self, ticker, start_date, end_date, timeframe=TimeFrame.Minute, limit=1000):
        """
        Fetch volume data for a given ticker between start_date and end_date.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        start_date (str): The start date in 'YYYY-MM-DDTHH:MM:SSZ' format.
        end_date (str): The end date in 'YYYY-MM-DDTHH:MM:SSZ' format.
        timeframe (str): The timeframe for the bars (e.g., '1Min', '5Min', '15Min', 'day').
        limit (int): The maximum number of bars to fetch.

        Returns:
        pd.DataFrame: A DataFrame containing the volume data.
        """
        try:
            barset = self.api.get_bars(
                symbol=ticker,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                limit=limit
            ).df

            if not barset.empty:
                volume_data = barset['volume']
                return volume_data
            else:
                print(f"No volume data found for {ticker} between {start_date} and {end_date}.")
                return None

        except tradeapi.rest.APIError as api_error:
            print(f"API error fetching volume data for {ticker}: {api_error}")
            return None
        except Exception as e:
            print(f"Error fetching volume data for {ticker}: {e}")
            return None

    def calculate_sma(self, ticker, period, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Calculate the Simple Moving Average (SMA) for a given ticker and period.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        period (int): The period for calculating the SMA.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.Series: A Series containing the SMA values.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None:
                # Calculate SMA using rolling window
                sma = historical_data['close'].rolling(window=period).mean()

                return sma
            else:
                print(f"No historical data found for {ticker} between {start_date} and {end_date}.")
                return None

        except Exception as e:
            print(f"Error calculating SMA for {ticker}: {e}")
            return None

    def calculate_ema(self, ticker, period, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Calculate the Exponential Moving Average (EMA) for a given ticker and period.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        period (int): The period for calculating the EMA.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.Series: A Series containing the EMA values.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None:
                # Calculate EMA using the ewm (exponential weighted mean) method
                ema = historical_data['close'].ewm(span=period, adjust=False).mean()

                return ema
            else:
                print(f"No historical data found for {ticker} between {start_date} and {end_date}.")
                return None

        except Exception as e:
            print(f"Error calculating EMA for {ticker}: {e}")
            return None
        

    def calculate_rsi(self, ticker, period, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Calculate the Relative Strength Index (RSI) for a given ticker and period.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        period (int): The period for calculating the RSI.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.Series: A Series containing the RSI values.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None:
                # Calculate the differences in closing prices
                delta = historical_data['close'].diff()

                # Calculate gains (positive differences) and losses (negative differences)
                gain = (delta.where(delta > 0, 0)).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)

                # Calculate the average gain and loss
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()

                # Calculate the relative strength (RS)
                rs = avg_gain / avg_loss

                # Calculate the RSI
                rsi = 100 - (100 / (1 + rs))

                return rsi
            else:
                print(f"No historical data found for {ticker} between {start_date} and {end_date}.")
                return None

        except Exception as e:
            print(f"Error calculating RSI for {ticker}: {e}")
            return None

    def calculate_macd(self, ticker, short_period, long_period, signal_period, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Calculate the Moving Average Convergence Divergence (MACD) for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        short_period (int): The period for the short EMA.
        long_period (int): The period for the long EMA.
        signal_period (int): The period for the signal line EMA.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.DataFrame: A DataFrame containing the MACD line, signal line, and MACD histogram.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None:
                # Calculate the short and long EMAs
                short_ema = historical_data['close'].ewm(span=short_period, adjust=False).mean()
                long_ema = historical_data['close'].ewm(span=long_period, adjust=False).mean()

                # Calculate the MACD line
                macd_line = short_ema - long_ema

                # Calculate the signal line
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

                # Calculate the MACD histogram
                macd_histogram = macd_line - signal_line

                # Create a DataFrame to hold the MACD components
                macd_df = pd.DataFrame({
                    'MACD_Line': macd_line,
                    'Signal_Line': signal_line,
                    'MACD_Histogram': macd_histogram
                })

                return macd_df
            else:
                print(f"No historical data found for {ticker} between {start_date} and {end_date}.")
                return None

        except Exception as e:
            print(f"Error calculating MACD for {ticker}: {e}")
            return None

    def calculate_bollinger_bands(self, ticker, period, std_dev, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Calculate the Bollinger Bands for a given ticker, period, and standard deviation.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        period (int): The period for calculating the middle band (SMA).
        std_dev (float): The number of standard deviations for the upper and lower bands.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.DataFrame: A DataFrame containing the middle band (SMA), upper band, and lower band.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None:
                # Calculate the middle band (SMA)
                middle_band = historical_data['close'].rolling(window=period).mean()

                # Calculate the rolling standard deviation
                rolling_std_dev = historical_data['close'].rolling(window=period).std()

                # Calculate the upper and lower bands
                upper_band = middle_band + (rolling_std_dev * std_dev)
                lower_band = middle_band - (rolling_std_dev * std_dev)

                # Create a DataFrame to hold the Bollinger Bands components
                bollinger_bands_df = pd.DataFrame({
                    'Middle_Band': middle_band,
                    'Upper_Band': upper_band,
                    'Lower_Band': lower_band
                })

                return bollinger_bands_df
            else:
                print(f"No historical data found for {ticker} between {start_date} and {end_date}.")
                return None

        except Exception as e:
            print(f"Error calculating Bollinger Bands for {ticker}: {e}")
            return None

    def calculate_fibonacci_retracement(self, high_price, low_price):
        """
        Calculate the Fibonacci retracement levels given a high and low price.

        Parameters:
        high_price (float): The high price.
        low_price (float): The low price.

        Returns:
        dict: A dictionary containing the Fibonacci retracement levels.
        """
        try:
            # Calculate the difference between the high and low prices
            price_range = high_price - low_price

            # Calculate Fibonacci levels
            level_0 = high_price
            level_23_6 = high_price - (price_range * 0.236)
            level_38_2 = high_price - (price_range * 0.382)
            level_50 = high_price - (price_range * 0.5)
            level_61_8 = high_price - (price_range * 0.618)
            level_100 = low_price

            # Store Fibonacci levels in a dictionary
            fibonacci_levels = {
                '0%': level_0,
                '23.6%': level_23_6,
                '38.2%': level_38_2,
                '50%': level_50,
                '61.8%': level_61_8,
                '100%': level_100
            }

            return fibonacci_levels

        except Exception as e:
            print(f"Error calculating Fibonacci retracement levels: {e}")
            return None

    def calculate_stochastic_oscillator(self, ticker, period, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Calculate the Stochastic Oscillator for a given ticker and period.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        period (int): The period for calculating the Stochastic Oscillator.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.DataFrame: A DataFrame containing the %K and %D values of the Stochastic Oscillator.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) >= period:
                # Calculate the lowest low and highest high over the specified period
                low_min = historical_data['low'].rolling(window=period).min()
                high_max = historical_data['high'].rolling(window=period).max()

                # Calculate the %K (fast stochastic)
                k_percent = 100 * ((historical_data['close'] - low_min) / (high_max - low_min))

                # Calculate the %D (slow stochastic), which is the 3-period moving average of %K
                d_percent = k_percent.rolling(window=3).mean()

                # Create a DataFrame to hold the Stochastic Oscillator components
                stochastic_oscillator_df = pd.DataFrame({
                    '%K': k_percent,
                    '%D': d_percent
                }, index=historical_data.index)

                return stochastic_oscillator_df
            else:
                print(f"Not enough data to calculate Stochastic Oscillator for {ticker} with period {period}.")
                return None

        except Exception as e:
            print(f"Error calculating Stochastic Oscillator for {ticker}: {e}")
            return None

    def calculate_obv(self, ticker, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Calculate the On-Balance Volume (OBV) for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.Series: A Series containing the OBV values.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) > 1:
                # Initialize OBV and set the first OBV value to zero
                obv = [0]

                # Loop through the historical data to calculate OBV
                for i in range(1, len(historical_data)):
                    if historical_data['close'].iloc[i] > historical_data['close'].iloc[i - 1]:
                        obv.append(obv[-1] + historical_data['volume'].iloc[i])
                    elif historical_data['close'].iloc[i] < historical_data['close'].iloc[i - 1]:
                        obv.append(obv[-1] - historical_data['volume'].iloc[i])
                    else:
                        obv.append(obv[-1])

                # Create a Series to hold the OBV values
                obv_series = pd.Series(obv, index=historical_data.index)

                return obv_series
            else:
                print(f"Not enough data to calculate OBV for {ticker}.")
                return None

        except Exception as e:
            print(f"Error calculating OBV for {ticker}: {e}")
            return None



    def calculate_vwap(self, ticker, period, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Calculate the Volume Weighted Average Price (VWAP) for a given ticker and period.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        period (int): The period for calculating the VWAP.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.Series: A Series containing the VWAP values.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) >= period:
                # Calculate VWAP
                vwap = []
                for i in range(period, len(historical_data)):
                    volume_sum = sum(historical_data['volume'].iloc[i - period:i])
                    if volume_sum != 0:
                        vwap.append(sum(historical_data['close'].iloc[i - period:i] * historical_data['volume'].iloc[i - period:i]) / volume_sum)
                    else:
                        vwap.append(None)

                # Create a Series to hold the VWAP values
                vwap_series = pd.Series(vwap, index=historical_data.index[period:])

                return vwap_series
            else:
                print(f"Not enough data to calculate VWAP for {ticker} with period {period}.")
                return None

        except Exception as e:
            print(f"Error calculating VWAP for {ticker}: {e}")
            return None

    def generate_moving_average_signal(self, ticker, short_period, long_period, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Generate a moving average crossover signal for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        short_period (int): The period for the short-term moving average.
        long_period (int): The period for the long-term moving average.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.DataFrame: A DataFrame containing the moving average crossover signals.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) >= long_period:
                # Calculate short-term moving average
                short_ma = historical_data['close'].rolling(window=short_period).mean()

                # Calculate long-term moving average
                long_ma = historical_data['close'].rolling(window=long_period).mean()

                # Generate signal (1 for buy, -1 for sell, 0 for hold)
                signal = np.where(short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0))

                # Create a DataFrame to hold the moving average signals
                signal_df = pd.DataFrame({
                    'Short_MA': short_ma,
                    'Long_MA': long_ma,
                    'Signal': signal
                }, index=historical_data.index)

                return signal_df
            else:
                print(f"Not enough data to generate moving average signal for {ticker}.")
                return None

        except Exception as e:
            print(f"Error generating moving average signal for {ticker}: {e}")
            return None

    def generate_rsi_signal(self, ticker, period=14, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Generate an RSI signal for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        period (int): The period for calculating the RSI. Default is 14.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.DataFrame: A DataFrame containing the RSI values and corresponding signals.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) > period:
                # Calculate price changes
                delta = historical_data['close'].diff()

                # Calculate gain and loss
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)

                # Calculate average gain and average loss over the period
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()

                # Calculate RS (Relative Strength)
                rs = avg_gain / avg_loss

                # Calculate RSI (Relative Strength Index)
                rsi = 100 - (100 / (1 + rs))

                # Generate signal (1 for buy, -1 for sell, 0 for hold)
                signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))

                # Create a DataFrame to hold the RSI values and signals
                rsi_signal_df = pd.DataFrame({
                    'RSI': rsi,
                    'Signal': signal
                }, index=historical_data.index)

                return rsi_signal_df
            else:
                print(f"Not enough data to generate RSI signal for {ticker}.")
                return None

        except Exception as e:
            print(f"Error generating RSI signal for {ticker}: {e}")
            return None

    def generate_macd_signal(self, ticker, short_period=12, long_period=26, signal_period=9, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Generate a MACD signal for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        short_period (int): The period for the short-term EMA. Default is 12.
        long_period (int): The period for the long-term EMA. Default is 26.
        signal_period (int): The period for the signal line. Default is 9.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.DataFrame: A DataFrame containing the MACD values and corresponding signals.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) >= long_period:
                # Calculate short-term EMA
                short_ema = historical_data['close'].ewm(span=short_period, adjust=False).mean()

                # Calculate long-term EMA
                long_ema = historical_data['close'].ewm(span=long_period, adjust=False).mean()

                # Calculate MACD line
                macd_line = short_ema - long_ema

                # Calculate signal line
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

                # Generate signal (1 for buy, -1 for sell, 0 for hold)
                signal = np.where(macd_line > signal_line, 1, np.where(macd_line < signal_line, -1, 0))

                # Create a DataFrame to hold the MACD values and signals
                macd_signal_df = pd.DataFrame({
                    'MACD': macd_line,
                    'Signal Line': signal_line,
                    'Signal': signal
                }, index=historical_data.index)

                return macd_signal_df
            else:
                print(f"Not enough data to generate MACD signal for {ticker}.")
                return None

        except Exception as e:
            print(f"Error generating MACD signal for {ticker}: {e}")
            return None


# Example of initializing the bot with API credentials

    def generate_bollinger_band_signal(self, ticker, period=20, std_dev=2, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Generate a Bollinger Bands signal for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        period (int): The period for calculating the moving average and standard deviation. Default is 20.
        std_dev (int): The number of standard deviations for the width of the bands. Default is 2.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min').

        Returns:
        pd.DataFrame: A DataFrame containing the Bollinger Bands values and corresponding signals.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) >= period:
                # Calculate moving average
                ma = historical_data['close'].rolling(window=period).mean()

                # Calculate standard deviation
                std = historical_data['close'].rolling(window=period).std()

                # Calculate upper and lower bands
                upper_band = ma + std * std_dev
                lower_band = ma - std * std_dev

                # Generate signal (1 for buy, -1 for sell, 0 for hold)
                signal = np.where(historical_data['close'] > upper_band, -1,
                                  np.where(historical_data['close'] < lower_band, 1, 0))

                # Create a DataFrame to hold the Bollinger Bands values and signals
                bollinger_band_signal_df = pd.DataFrame({
                    'Upper Band': upper_band,
                    'Lower Band': lower_band,
                    'Signal': signal
                }, index=historical_data.index)

                return bollinger_band_signal_df
            else:
                print(f"Not enough data to generate Bollinger Bands signal for {ticker}.")
                return None

        except Exception as e:
            print(f"Error generating Bollinger Bands signal for {ticker}: {e}")
            return None

    def generate_fibonacci_signal(self, ticker, start_date=None, end_date=None, timeframe=TimeFrame.Minute, proximity_threshold=0.01):
        """
        Generate a Fibonacci retracement signal for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (TimeFrame): The timeframe for the bars. Default is TimeFrame.Minute.
        proximity_threshold (float): The proximity threshold to consider for generating the signal. Default is 0.01 (1%).

        Returns:
        pd.DataFrame: A DataFrame containing the Fibonacci retracement signal.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) >= 2:
                # Identify swing high and swing low
                swing_high = historical_data['high'].max()
                swing_low = historical_data['low'].min()

                # Calculate Fibonacci retracement levels
                retracement_levels = [0.236, 0.382, 0.5, 0.618, 1.0]  # Fibonacci ratios
                retracement_values = [swing_high - (ratio * (swing_high - swing_low)) for ratio in retracement_levels]

                # Current price
                current_price = historical_data['close'].iloc[-1]

                # Determine if the current price is within the proximity threshold of any retracement level
                signal = any(abs(current_price - level) / level <= proximity_threshold for level in retracement_values)

                # Create a DataFrame to hold the Fibonacci retracement signal
                fibonacci_signal_df = pd.DataFrame(index=historical_data.index)
                fibonacci_signal_df['Signal'] = 1 if signal else -1

                return fibonacci_signal_df
            else:
                print(f"Not enough data to generate Fibonacci retracement signal for {ticker}.")
                return None

        except Exception as e:
            print(f"Error generating Fibonacci retracement signal for {ticker}: {e}")
            return None


    def generate_stochastic_signal(self, ticker, k_period=14, d_period=3, start_date=None, end_date=None, timeframe=TimeFrame.Minute):
        """
        Generate a Stochastic Oscillator signal for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        k_period (int): The lookback period for %K calculation. Default is 14.
        d_period (int): The smoothing period for %D calculation. Default is 3.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min'). Default is '1D'.

        Returns:
        pd.DataFrame: A DataFrame containing the Stochastic Oscillator values and corresponding signals.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) >= k_period + d_period:
                # Calculate %K
                lowest_low = historical_data['low'].rolling(window=k_period).min()
                highest_high = historical_data['high'].rolling(window=k_period).max()
                stochastic_k = ((historical_data['close'] - lowest_low) / (highest_high - lowest_low)) * 100

                # Calculate %D
                stochastic_d = stochastic_k.rolling(window=d_period).mean()

                # Generate signal (1 for buy, -1 for sell, 0 for hold)
                signal = np.where((stochastic_k > stochastic_d) & (stochastic_k < 80), 1,
                                np.where((stochastic_k < stochastic_d) & (stochastic_k > 20), -1, 0))

                # Create a DataFrame to hold the Stochastic Oscillator values and signals
                stochastic_signal_df = pd.DataFrame({
                    '%K': stochastic_k,
                    '%D': stochastic_d,
                    'Signal': signal
                }, index=historical_data.index)

                return stochastic_signal_df
            else:
                print(f"Not enough data to generate Stochastic Oscillator signal for {ticker}.")
                return None

        except Exception as e:
            print(f"Error generating Stochastic Oscillator signal for {ticker}: {e}")
            return None

    def generate_volume_signal(self, ticker, ma_period=20, start_date=None, end_date=None, timeframe=TimeFrame.Day):
        """
        Generate a volume signal for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        ma_period (int): The period for calculating the moving average of volume. Default is 20.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DDTHH:MM:SSZ' format (optional).
        timeframe (str): The timeframe for the bars (e.g., '1D', '1Min', '5Min', '15Min'). Default is '1D'.

        Returns:
        pd.DataFrame: A DataFrame containing the volume values, moving average, and corresponding signals.
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)

            if historical_data is not None and len(historical_data) >= ma_period:
                # Calculate moving average of volume
                volume_ma = historical_data['volume'].rolling(window=ma_period).mean()

                # Generate signal based on volume and its moving average
                signal = np.where(historical_data['volume'] > volume_ma, 1, -1)

                # Create a DataFrame to hold the volume values, moving average, and signals
                volume_signal_df = pd.DataFrame({
                    'Volume': historical_data['volume'],
                    'Volume_MA': volume_ma,
                    'Signal': signal
                }, index=historical_data.index)

                return volume_signal_df
            else:
                print(f"Not enough data to generate volume signal for {ticker}.")
                return None

        except Exception as e:
            print(f"Error generating volume signal for {ticker}: {e}")
            return None

    def generate_ema_signal(self, ticker, period, start_date, end_date, timeframe=TimeFrame.Minute):
        try:
            historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe)
            ema = historical_data['close'].ewm(span=period, adjust=False).mean()
            signal = pd.DataFrame(index=historical_data.index)
            signal['Signal'] = np.where(historical_data['close'] > ema, 1, -1)
            return signal
        except Exception as e:
            print(f"Error generating EMA signal for {ticker}: {e}")
            return None
            
    def aggregate_signals(self, ticker, start_date, end_date, timeframe=TimeFrame.Minute, threshold=0.2):
        """
        Aggregate signals generated by different technical indicators for a given ticker.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        start_date (str): The start date for historical data retrieval.
        end_date (str): The end date for historical data retrieval.
        timeframe (TimeFrame): The timeframe for the historical data.
        threshold (float): The minimum percentage of signals required to make a buy/sell decision.

        Returns:
        pd.Series: A Series containing the final aggregated signals based on a majority vote.
        """
        try:
            # Generate individual signals with their respective parameters
            ma_signal = self.generate_moving_average_signal(ticker, short_period=10, long_period=50, start_date=start_date, end_date=end_date, timeframe=timeframe)
            ema_signal = self.generate_ema_signal(ticker, period=20, start_date=start_date, end_date=end_date, timeframe=timeframe)
            rsi_signal = self.generate_rsi_signal(ticker, period=14, start_date=start_date, end_date=end_date, timeframe=timeframe)
            macd_signal = self.generate_macd_signal(ticker, short_period=12, long_period=26, signal_period=9, start_date=start_date, end_date=end_date, timeframe=timeframe)
            bollinger_band_signal = self.generate_bollinger_band_signal(ticker, period=20, std_dev=2, start_date=start_date, end_date=end_date, timeframe=timeframe)
            fibonacci_signal = self.generate_fibonacci_signal(ticker, start_date=start_date, end_date=end_date, timeframe=timeframe)
            stochastic_signal = self.generate_stochastic_signal(ticker, k_period=14, d_period=3, start_date=start_date, end_date=end_date, timeframe=timeframe)
            volume_signal = self.generate_volume_signal(ticker, start_date=start_date, end_date=end_date, timeframe=timeframe)

            # Check if any signal is None
            if any(signal is None for signal in [ma_signal, ema_signal, rsi_signal, macd_signal, bollinger_band_signal, fibonacci_signal, stochastic_signal, volume_signal]):
                print(f"Not all signals could be generated for {ticker}.")
                return None

            # Merge signals into a single DataFrame
            aggregated_signals_df = pd.concat([
                ma_signal['Signal'],
                ema_signal['Signal'],
                rsi_signal['Signal'],
                macd_signal['Signal'],
                bollinger_band_signal['Signal'],
                fibonacci_signal['Signal'],
                stochastic_signal['Signal'],
                volume_signal['Signal']
            ], axis=1)

            # Rename columns for clarity
            aggregated_signals_df.columns = [
                'MA_Signal',
                'EMA_Signal',
                'RSI_Signal',
                'MACD_Signal',
                'Bollinger_Band_Signal',
                'Fibonacci_Signal',
                'Stochastic_Signal',
                'Volume_Signal'
            ]

            # Determine the final signal based on a majority vote with a threshold
            def majority_vote(row, threshold):
                total_signals = len(row)
                buy_signals = sum(row == 1)
                sell_signals = sum(row == -1)
                buy_percentage = buy_signals / total_signals
                sell_percentage = sell_signals / total_signals

                if buy_percentage >= threshold:
                    return 1
                elif sell_percentage >= threshold:
                    return -1
                else:
                    return 0

            aggregated_signals_df['Final_Signal'] = aggregated_signals_df.apply(majority_vote, axis=1, threshold=threshold)

            return aggregated_signals_df['Final_Signal']

        except Exception as e:
            print(f"Error aggregating signals for {ticker}: {e}")
            return None

    def calculate_position_size(self, ticker, account_balance, risk_per_trade, stop_loss_percentage):
        """
        Calculate the position size for a trade based on account balance and risk per trade.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        account_balance (float): The total account balance.
        risk_per_trade (float): The percentage of the account balance to risk on a single trade.
        stop_loss_percentage (float): The percentage distance between the entry price and the stop loss.

        Returns:
        float: The number of shares to buy or sell.
        """
        try:
            # Fetch the current price of the ticker
            current_price = self.get_current_price(ticker)

            if current_price is None:
                raise ValueError(f"Could not fetch the current price for {ticker}")

            # Calculate the dollar amount to risk per trade
            dollar_risk_per_trade = account_balance * risk_per_trade

            # Calculate the stop loss price
            stop_loss_price = current_price * (1 - stop_loss_percentage)

            # Calculate the dollar risk per share
            dollar_risk_per_share = current_price - stop_loss_price

            # Calculate the position size
            position_size = dollar_risk_per_trade / dollar_risk_per_share

            return position_size

        except Exception as e:
            print(f"Error calculating position size for {ticker}: {e}")
            return None

    def set_stop_loss(self, ticker, entry_price, stop_loss_level, quantity):
        """
        Set a stop-loss order for a given position.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        entry_price (float): The entry price of the trade.
        stop_loss_level (float): The stop-loss price level.
        quantity (int): The number of shares to apply the stop-loss to.

        Returns:
        str: Order ID of the stop-loss order.
        """
        try:
            # Calculate the stop-loss price
            stop_loss_price = entry_price * (1 - stop_loss_level)

            # Place the stop-loss order
            stop_loss_order = self.api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='sell',
                type='stop',
                time_in_force='gtc',
                stop_price=stop_loss_price
            )

            return stop_loss_order.id

        except Exception as e:
            print(f"Error setting stop-loss for {ticker}: {e}")
            return None

    def set_take_profit(self, ticker, entry_price, take_profit_level, quantity):
        """
        Set a take-profit order for a given position.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        entry_price (float): The entry price of the trade.
        take_profit_level (float): The take-profit price level.
        quantity (int): The number of shares to apply the take-profit to.

        Returns:
        str: Order ID of the take-profit order.
        """
        try:
            # Calculate the take-profit price
            take_profit_price = entry_price * (1 + take_profit_level)

            # Place the take-profit order
            take_profit_order = self.api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='sell',
                type='limit',
                time_in_force='gtc',
                limit_price=take_profit_price
            )

            return take_profit_order.id

        except Exception as e:
            print(f"Error setting take-profit for {ticker}: {e}")
            return None




    def place_market_order(self, ticker, side, quantity):
        try:
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            market_order = self.trading_client.submit_order(order_data=market_order_data)
            print("Market order placed successfully:", market_order)
            return market_order
        except Exception as e:
            print(f"Error placing market order for {ticker}: {e}")
            return None

    def get_current_price(self, ticker):
        try:
            latest_trade = self.api.get_latest_trade(ticker)
            current_price = latest_trade.price
            return current_price
        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
            return None
    
    def place_market_order(self, ticker, side, quantity):
        try:
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            market_order = self.trading_client.submit_order(order_data=market_order_data)
            print("Market order placed successfully:", market_order)
            return market_order
        except Exception as e:
            print(f"Error placing market order for {ticker}: {e}")
            return None

    def place_limit_order(self, ticker, side, quantity, price):
        try:
            limit_order_data = LimitOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                limit_price=price,
                time_in_force=TimeInForce.DAY
            )
            limit_order = self.trading_client.submit_order(order_data=limit_order_data)
            print("Limit order placed successfully:", limit_order)
            return limit_order
        except Exception as e:
            print(f"Error placing limit order for {ticker}: {e}")
            return None

    def place_stop_loss_order(self, ticker, quantity, stop_price):
        try:
            stop_loss_data = StopLossRequest(
                stop_price=stop_price
            )
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_loss=stop_loss_data
            )
            stop_loss_order = self.trading_client.submit_order(order_data=market_order_data)
            print("Stop-loss order placed successfully:", stop_loss_order)
            return stop_loss_order
        except Exception as e:
            print(f"Error placing stop-loss order for {ticker}: {e}")
            return None


    def place_take_profit_order(self, ticker, quantity, take_profit_price):
        try:
            take_profit_data = TakeProfitRequest(
                limit_price=take_profit_price
            )
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                take_profit=take_profit_data
            )
            take_profit_order = self.trading_client.submit_order(order_data=market_order_data)
            print("Take-profit order placed successfully:", take_profit_order)
            return take_profit_order
        except Exception as e:
            print(f"Error placing take-profit order for {ticker}: {e}")
            return None

    def cancel_order(self, order_id):
        try:
            cancel_result = self.trading_client.cancel_order(order_id)
            print(f"Order {order_id} canceled successfully:", cancel_result)
            return cancel_result
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return None
# Example parameters for placing a limit order with take-profit and stop-loss

    def modify_order(self, order_id, new_price):
        try:
            # Cancel the existing order
            self.cancel_order(order_id)

            # Retrieve the details of the canceled order to replicate it with a new price
            order = self.api.get_order(order_id)

            if order:
                # Place a new order with the same details but a different price
                if order.order_type == 'limit':
                    modified_order_data = LimitOrderRequest(
                        symbol=order.symbol,
                        qty=order.qty,
                        side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                        limit_price=new_price,
                        time_in_force=order.time_in_force
                    )
                    modified_order = self.trading_client.submit_order(order_data=modified_order_data)
                    print("Order modified successfully:", modified_order)
                    return modified_order
                else:
                    print("Only limit orders can be modified for now.")
                    return None
            else:
                print(f"Order {order_id} not found.")
                return None
        except Exception as e:
            print(f"Error modifying order {order_id}: {e}")
            return None

    def get_account_balance(self):
        try:
            account = self.api.get_account()
            balance = {
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity)
            }
            return balance
        except Exception as e:
            print(f"Error fetching account balance: {e}")
            return None

    def get_current_holdings(self):
        try:
            positions = self.api.list_positions()
            holdings = []
            for position in positions:
                holding = {
                    "ticker": position.symbol,
                    "quantity": float(position.qty),
                    "avg_entry_price": float(position.avg_entry_price),
                    "market_value": float(position.market_value),
                    "current_price": float(position.current_price),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc)
                }
                holdings.append(holding)
            return holdings
        except Exception as e:
            print(f"Error fetching current holdings: {e}")
            return None

    def backtest_strategy(self, ticker, strategy, historical_data):
        """
        Backtests a trading strategy on historical data.

        Parameters:
        ticker (str): The ticker symbol for the asset.
        strategy (function): The strategy function that takes historical data as input and returns buy/sell signals.
        historical_data (pd.DataFrame): Historical OHLCV data.

        Returns:
        pd.DataFrame: DataFrame containing the backtest results.
        """
        try:
            # Apply the strategy function to generate signals
            signals = strategy(historical_data)

            # Simulate trading based on the signals
            backtest_results = pd.DataFrame(index=historical_data.index)
            backtest_results['price'] = historical_data['close']
            backtest_results['signal'] = signals

            # Calculate the positions based on signals
            backtest_results['position'] = backtest_results['signal'].apply(
                lambda signal: 1 if signal == 'buy' else (-1 if signal == 'sell' else 0)
            )

            # Calculate returns based on positions
            backtest_results['returns'] = backtest_results['price'].pct_change() * backtest_results['position'].shift(1)

            # Calculate cumulative returns
            backtest_results['cumulative_returns'] = (1 + backtest_results['returns']).cumprod()

            return backtest_results
        except Exception as e:
            print(f"Error in backtesting strategy for {ticker}: {e}")
            return None  

    def simulate_trading(self, tickers, start_date, end_date, take_profit_pct=0.1, stop_loss_pct=0.05, trailing_stop_loss_pct=0.03, buy_budget_pct=0.8, min_hold_time_minutes=10):
        """
        Simulates trading based on a combined strategy and historical data within a specified date range for multiple tickers.

        Parameters:
        tickers (list): List of ticker symbols for the assets.
        start_date (str): The start date for historical data retrieval in 'YYYY-MM-DD' format.
        end_date (str): The end date for historical data retrieval in 'YYYY-MM-DD' format.
        take_profit_pct (float): The percentage increase in price for taking profit.
        stop_loss_pct (float): The percentage decrease in price for setting a stop loss.
        trailing_stop_loss_pct (float): The percentage decrease in price for setting a trailing stop loss.
        buy_budget_pct (float): The percentage of available cash to use for buying.
        min_hold_time_minutes (int): The minimum time to hold a position before selling, in minutes.

        Returns:
        dict: Dictionary containing final cash, positions, and portfolio value.
        list: List of tuples containing trades.
        """
        try:
            # Initialize account
            initial_cash = 100000
            cash = initial_cash
            positions = {ticker: 0 for ticker in tickers}
            trades = []  # To keep track of trades
            stop_loss_prices = {ticker: None for ticker in tickers}
            trailing_stop_loss_prices = {ticker: None for ticker in tickers}
            entry_times = {ticker: None for ticker in tickers}  # To track the entry times of positions

            # Simulate trading for each ticker
            for ticker in tickers:
                # Fetch historical data
                historical_data = self.get_historical_data(ticker, start_date, end_date, timeframe=TimeFrame.Minute, limit=100000000000000000000000000)
                if historical_data is None or len(historical_data) == 0:
                    print(f"No historical data found for {ticker} between {start_date} and {end_date}. Skipping this ticker.")
                    continue

                # Generate aggregated signals
                aggregated_signals = self.aggregate_signals(ticker, start_date, end_date)
                if aggregated_signals is None or len(aggregated_signals) == 0:
                    print(f"No aggregated signals generated for {ticker} between {start_date} and {end_date}. Skipping this ticker.")
                    continue

                entry_price = None
                for i in range(1, len(aggregated_signals)):
                    current_price = historical_data['close'].iloc[i]
                    current_time = historical_data.index[i]

                    if aggregated_signals.iloc[i] == 1.0 and cash >= current_price:
                        # Buy using a percentage of available cash
                        budget = cash * buy_budget_pct
                        quantity = budget // current_price
                        if quantity > 0:
                            cash -= quantity * current_price
                            positions[ticker] += quantity
                            entry_price = current_price
                            stop_loss_prices[ticker] = entry_price * (1 - stop_loss_pct)
                            trailing_stop_loss_prices[ticker] = entry_price * (1 - trailing_stop_loss_pct)
                            entry_times[ticker] = current_time
                            trades.append((current_time, ticker, 'buy', quantity, current_price))

                    elif aggregated_signals.iloc[i] == -1.0 and positions[ticker] > 0:
                        hold_time = (current_time - entry_times[ticker]).total_seconds() / 60
                        if hold_time >= min_hold_time_minutes:
                            # Sell all holdings
                            cash += positions[ticker] * current_price
                            trades.append((current_time, ticker, 'sell', positions[ticker], current_price))
                            positions[ticker] = 0
                            stop_loss_prices[ticker] = None
                            trailing_stop_loss_prices[ticker] = None
                            entry_times[ticker] = None

                    # Check for take profit
                    if positions[ticker] > 0 and entry_price and current_price >= entry_price * (1 + take_profit_pct):
                        hold_time = (current_time - entry_times[ticker]).total_seconds() / 60
                        if hold_time >= min_hold_time_minutes:
                            cash += positions[ticker] * current_price
                            trades.append((current_time, ticker, 'sell', positions[ticker], current_price))
                            positions[ticker] = 0
                            stop_loss_prices[ticker] = None
                            trailing_stop_loss_prices[ticker] = None
                            entry_times[ticker] = None

                    # Check for stop loss
                    if positions[ticker] > 0 and current_price <= stop_loss_prices[ticker]:
                        hold_time = (current_time - entry_times[ticker]).total_seconds() / 60
                        if hold_time >= min_hold_time_minutes:
                            cash += positions[ticker] * current_price
                            trades.append((current_time, ticker, 'sell', positions[ticker], current_price))
                            positions[ticker] = 0
                            stop_loss_prices[ticker] = None
                            trailing_stop_loss_prices[ticker] = None
                            entry_times[ticker] = None

                    # Check for trailing stop loss
                    if positions[ticker] > 0:
                        if current_price > trailing_stop_loss_prices[ticker]:
                            trailing_stop_loss_prices[ticker] = max(trailing_stop_loss_prices[ticker], current_price * (1 - trailing_stop_loss_pct))
                        elif current_price <= trailing_stop_loss_prices[ticker]:
                            hold_time = (current_time - entry_times[ticker]).total_seconds() / 60
                            if hold_time >= min_hold_time_minutes:
                                cash += positions[ticker] * current_price
                                trades.append((current_time, ticker, 'sell', positions[ticker], current_price))
                                positions[ticker] = 0
                                stop_loss_prices[ticker] = None
                                trailing_stop_loss_prices[ticker] = None
                                entry_times[ticker] = None

            # Calculate portfolio value at the end of the period
            final_value = cash + sum(positions[ticker] * self.get_historical_data(ticker, start_date, end_date, timeframe=TimeFrame.Minute)['close'].iloc[-1] for ticker in tickers if positions[ticker] > 0)

            result = {
                'cash': cash,
                'positions': positions,
                'value': final_value
            }

            return result, trades

        except Exception as e:
            print(f"Error simulating trading: {e}")
            return None, None

    def live_trading(self, tickers, interval=60, threshold=0.2, min_hold_time_minutes=60):
        try:
            while True:
                for ticker in tickers:
                    # Get current price
                    current_price = self.get_current_price(ticker)
                    if current_price is None:
                        continue

                    # Get aggregated signals
                    end_date = pd.Timestamp.utcnow().isoformat()
                    start_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).isoformat()
                    aggregated_signals = self.aggregate_signals(ticker, start_date, end_date, threshold=threshold)
                    if aggregated_signals is None:
                        continue

                    # Determine the latest signal
                    latest_signal = aggregated_signals.iloc[-1]
                    
                    # Get account balance and holdings
                    balance = self.get_account_balance()
                    holdings = self.get_current_holdings()
                    
                    # Ensure we have balance and holdings data
                    if balance is None or holdings is None:
                        continue
                    
                    # Check if we already hold the ticker
                    holding = next((h for h in holdings if h['ticker'] == ticker), None)
                    
                    if latest_signal == 1 and (holding is None or holding['quantity'] == 0):
                        # Buy signal and we do not hold the ticker
                        budget = balance['cash'] * 0.2  # 20% of cash balance
                        quantity = budget // current_price
                        if quantity > 0:
                            self.place_market_order(ticker, 'buy', quantity)
                    elif latest_signal == -1 and holding is not None and holding['quantity'] > 0:
                        # Sell signal and we hold the ticker
                        self.place_market_order(ticker, 'sell', holding['quantity'])

                # Sleep until the next check
                time.sleep(interval)
        except Exception as e:
            print(f"Error in live trading: {e}")


# Example usage
bot = TradingBot(API_KEY, API_SECRET, BASE_URL)
start_date = '2024-03-01T16:15:00Z'
end_date = '2024-06-01T16:20:00Z'

tickers = [
    "MMM", "ABT", "ACN", "T", "BAC", "BA", "KO", "XOM", "GE", "IBM",
    "AAPL", "MSFT", "JNJ", "JPM", "WMT", "PG", "DIS", "HD", "VZ", "UNH",
    "CVX", "PFE", "PEP", "MA", "V", "MRK", "NKE", "INTC", "MCD", "WFC",
    "GS", "AXP", "CAT", "CSCO", "ABT", "SLB", "KO", "XOM", "MO", "BMY",
    "BK", "CL", "COP", "DHR", "DOW", "DUK", "EMR", "ETN", "FDX", "GD",
    "GILD", "GM", "HAL", "HON", "HPQ", "JCI", "KMB", "LMT", "LVS", "MET",
    "MMM", "MON", "NEE", "NSC", "ORCL", "PM", "PNC", "PRU", "RTX", "SO",
    "SPG", "TGT", "TXN", "UNP", "USB", "UTX", "VLO", "VRTX", "WBA", "WDC",
    "WM", "XEL"
]

bot.live_trading(tickers)



