from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid

@dataclass
class Position:
    """
    Represents a trading position with entry, exit, and risk management details.
    """
    symbol: str                     # Trading pair symbol (e.g., 'BTC/USDT')
    side: str                       # 'long' or 'short'
    entry_price: float             # Position entry price
    quantity: float                # Position size in base currency
    stop_loss: float              # Stop loss price
    take_profit: float            # Take profit price
    entry_time: datetime          # Position entry timestamp
    position_id: str = None       # Unique position identifier
    exit_price: Optional[float] = None    # Position exit price
    exit_time: Optional[datetime] = None  # Position exit timestamp
    trailing_stop: Optional[float] = None # Trailing stop price
    pnl: Optional[float] = None          # Realized profit/loss
    status: str = 'open'                 # Position status: 'open' or 'closed'
    
    def __post_init__(self):
        """Initialize position with a unique ID if not provided."""
        if self.position_id is None:
            self.position_id = str(uuid.uuid4())
    
    def update_trailing_stop(self, current_price: float, trailing_pct: float) -> None:
        """
        Update the trailing stop price based on current price and trailing percentage.
        
        Args:
            current_price (float): Current market price
            trailing_pct (float): Trailing stop percentage
        """
        if self.side == 'long':
            new_stop = current_price * (1 - trailing_pct)
            if self.trailing_stop is None or new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        else:  # short position
            new_stop = current_price * (1 + trailing_pct)
            if self.trailing_stop is None or new_stop < self.trailing_stop:
                self.trailing_stop = new_stop
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized profit/loss based on current market price.
        
        Args:
            current_price (float): Current market price
            
        Returns:
            float: Unrealized PnL
        """
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:  # short position
            return (self.entry_price - current_price) * self.quantity
    
    def close_position(self, exit_price: float, exit_time: datetime) -> None:
        """
        Close the position and calculate realized PnL.
        
        Args:
            exit_price (float): Position exit price
            exit_time (datetime): Position exit timestamp
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = 'closed'
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # short position
            self.pnl = (self.entry_price - exit_price) * self.quantity
    
    def should_close(self, current_price: float) -> tuple[bool, str]:
        """
        Check if position should be closed based on stop loss, take profit, or trailing stop.
        
        Args:
            current_price (float): Current market price
            
        Returns:
            tuple[bool, str]: (should_close, reason)
        """
        if self.side == 'long':
            if current_price <= self.stop_loss:
                return True, 'stop_loss'
            if self.trailing_stop and current_price <= self.trailing_stop:
                return True, 'trailing_stop'
            if current_price >= self.take_profit:
                return True, 'take_profit'
        else:  # short position
            if current_price >= self.stop_loss:
                return True, 'stop_loss'
            if self.trailing_stop and current_price >= self.trailing_stop:
                return True, 'trailing_stop'
            if current_price <= self.take_profit:
                return True, 'take_profit'
        
        return False, None
    
    def get_risk_reward_ratio(self) -> float:
        """
        Calculate the risk-reward ratio for the position.
        
        Returns:
            float: Risk-reward ratio
        """
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk != 0 else 0
    
    def to_dict(self) -> dict:
        """
        Convert position to dictionary format.
        
        Returns:
            dict: Position details as dictionary
        """
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'entry_time': self.entry_time.isoformat(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'status': self.status
        } 