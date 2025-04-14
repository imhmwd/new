import pandas as pd
import numpy as np
import talib
from typing import Optional, List
import logging

from agents.base.agent import Agent, Signal

class VWAPAgent(Agent):
    """Agent that makes decisions based on Volume Weighted Average Price (VWAP)"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 lookback_periods: List[int] = [1, 5, 20]):
        """
        Initialize the VWAP Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            lookback_periods: Periods to calculate VWAP for (default: [1, 5, 20] days)
        """
        super().__init__(name="VWAP", symbol=symbol, timeframe=timeframe)
        self.lookback_periods = lookback_periods
        self.current_price = None
        self.vwap_values = {}
        self.confidence = 0.0
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data using VWAP
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for VWAP analysis")
            return Signal.NEUTRAL
        
        df = self.preprocess_data(data)
        
        # Current price is the latest close
        self.current_price = df['close'].iloc[-1]
        
        # Calculate VWAP for different periods
        for period in self.lookback_periods:
            if len(df) >= period:
                period_df = df.iloc[-period:]
                self.vwap_values[period] = self._calculate_vwap(period_df)
            else:
                self.vwap_values[period] = None
        
        # Generate signal
        signal = self._generate_signal()
        
        # Calculate confidence
        self._calculate_confidence()
        
        return signal
    
    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """
        Calculate VWAP for the given data period
        
        Args:
            data: DataFrame containing OHLCV data for a specific period
            
        Returns:
            float: VWAP value
        """
        # Calculate typical price: (high + low + close) / 3
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate VWAP
        data['tp_volume'] = data['typical_price'] * data['volume']
        vwap = data['tp_volume'].sum() / data['volume'].sum() if data['volume'].sum() > 0 else data['typical_price'].mean()
        
        return vwap
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on VWAP
        
        Returns:
            Signal: Trading signal enum
        """
        if (self.current_price is None or not self.vwap_values):
            return Signal.NEUTRAL
        
        # Get the short and long period VWAPs
        short_vwap = self.vwap_values.get(self.lookback_periods[0])
        mid_vwap = self.vwap_values.get(self.lookback_periods[1]) if len(self.lookback_periods) > 1 else None
        long_vwap = self.vwap_values.get(self.lookback_periods[-1])
        
        if short_vwap is None or long_vwap is None:
            return Signal.NEUTRAL
        
        # Define price distance from VWAP as a percentage
        short_distance = (self.current_price - short_vwap) / short_vwap
        long_distance = (self.current_price - long_vwap) / long_vwap
        
        # Strong buy signals
        if short_vwap > long_vwap and self.current_price < short_vwap * 0.995:
            # Price dipped below short VWAP in an uptrend (potential buying opportunity)
            return Signal.STRONG_BUY
        elif self.current_price < long_vwap * 0.98:
            # Price is significantly below long VWAP (potential value)
            return Signal.STRONG_BUY
            
        # Buy signals
        elif short_vwap > long_vwap and self.current_price > short_vwap and short_distance < 0.01:
            # Price is just above short VWAP in an uptrend
            return Signal.BUY
        elif self.current_price < short_vwap and short_distance > -0.01:
            # Price is just below short VWAP (potential support)
            return Signal.BUY
            
        # Strong sell signals
        elif short_vwap < long_vwap and self.current_price > short_vwap * 1.005:
            # Price spiked above short VWAP in a downtrend (potential selling opportunity)
            return Signal.STRONG_SELL
        elif self.current_price > long_vwap * 1.02:
            # Price is significantly above long VWAP (potential overvalued)
            return Signal.STRONG_SELL
            
        # Sell signals
        elif short_vwap < long_vwap and self.current_price < short_vwap and short_distance > -0.01:
            # Price is just below short VWAP in a downtrend
            return Signal.SELL
        elif self.current_price > short_vwap and short_distance < 0.01:
            # Price is just above short VWAP (potential resistance)
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on VWAP values"""
        if (self.current_price is None or not self.vwap_values):
            self.confidence = 0.0
            return
        
        # Get the short and long period VWAPs
        short_vwap = self.vwap_values.get(self.lookback_periods[0])
        long_vwap = self.vwap_values.get(self.lookback_periods[-1])
        
        if short_vwap is None or long_vwap is None:
            self.confidence = 0.0
            return
        
        # Calculate distance from VWAPs
        short_distance = abs(self.current_price - short_vwap) / short_vwap
        long_distance = abs(self.current_price - long_vwap) / long_vwap
        
        # Calculate VWAP trend strength
        vwap_trend = abs(short_vwap - long_vwap) / long_vwap
        
        # Confidence is higher when:
        # 1. Price is further from VWAPs (stronger signal)
        # 2. Short and long VWAPs show a clear trend (greater difference)
        self.confidence = min(1.0, (short_distance * 20) + (long_distance * 10) + (vwap_trend * 30))
            
    def get_confidence(self) -> float:
        """
        Return the confidence level of the agent's prediction
        
        Returns:
            float: Confidence level between 0.0 and 1.0
        """
        return self.confidence
    
    def get_explanation(self) -> str:
        """
        Get explanation of the agent's reasoning
        
        Returns:
            str: Explanation text
        """
        if (self.current_price is None or not self.vwap_values):
            return "VWAP values not available for analysis"
        
        signal = self._generate_signal()
        
        # Format values for display
        price_value = f"{self.current_price:.2f}"
        vwap_strings = []
        
        for period in sorted(self.vwap_values.keys()):
            if self.vwap_values[period] is not None:
                vwap_strings.append(f"VWAP({period}) = {self.vwap_values[period]:.2f}")
        
        vwap_text = ", ".join(vwap_strings)
        
        # Get the short and long period VWAPs
        short_vwap = self.vwap_values.get(self.lookback_periods[0])
        long_vwap = self.vwap_values.get(self.lookback_periods[-1])
        
        trend_text = ""
        if short_vwap is not None and long_vwap is not None:
            if short_vwap > long_vwap:
                trend_text = "uptrend"
            elif short_vwap < long_vwap:
                trend_text = "downtrend"
            else:
                trend_text = "neutral trend"
        
        if signal == Signal.STRONG_BUY:
            return f"Price ({price_value}) shows a strong buying opportunity relative to VWAP values ({vwap_text}). {trend_text.capitalize()}."
        elif signal == Signal.BUY:
            return f"Price ({price_value}) shows a potential buying opportunity near VWAP values ({vwap_text}). {trend_text.capitalize()}."
        elif signal == Signal.STRONG_SELL:
            return f"Price ({price_value}) shows a strong selling opportunity relative to VWAP values ({vwap_text}). {trend_text.capitalize()}."
        elif signal == Signal.SELL:
            return f"Price ({price_value}) shows a potential selling opportunity near VWAP values ({vwap_text}). {trend_text.capitalize()}."
        else:
            return f"Price ({price_value}) is neutral relative to VWAP values ({vwap_text}). {trend_text.capitalize()}." 