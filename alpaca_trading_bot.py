import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple, Optional
import asyncio
import websocket
import json
from concurrent.futures import ThreadPoolExecutor
import talib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlpacaTradingBot:
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        """
        Initialize the Alpaca trading bot
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper_trading: Whether to use paper trading (default: True)
        """
        base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        # Trading parameters
        self.position_size = 0.02  # 2% of portfolio per trade
        self.max_positions = 10
        self.risk_per_trade = 0.01  # 1% risk per trade
        
        # Strategy parameters
        self.momentum_lookback = 20
        self.mean_reversion_lookback = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Data storage
        self.positions = {}
        self.market_data = {}
        self.running = False
        
        # Watchlist for trading
        self.watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
        
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'day_trade_count': account.day_trade_count
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_market_data(self, symbol: str, timeframe: str = '1Min', limit: int = 1000) -> pd.DataFrame:
        """Get historical market data"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=limit
            )
            
            df = pd.DataFrame([{
                'timestamp': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            } for bar in bars])
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if df.empty or len(df) < 50:
            return df
            
        try:
            # Price-based indicators
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
            df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
            
            # Momentum indicators
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            
            # Volatility indicators
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Volume indicators
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            
            # Price momentum
            df['price_change'] = df['close'].pct_change()
            df['momentum'] = df['close'] / df['close'].shift(self.momentum_lookback) - 1
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def momentum_strategy(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """
        Momentum trading strategy
        Buy when price breaks above resistance with strong volume
        """
        if df.empty or len(df) < 50:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        try:
            # Momentum conditions
            momentum_up = latest['momentum'] > 0.02  # 2% momentum
            price_above_sma = latest['close'] > latest['sma_20']
            volume_surge = latest['volume'] > latest['volume_sma'] * 1.5
            rsi_not_overbought = latest['rsi'] < 75
            
            # Mean reversion conditions
            rsi_oversold = latest['rsi'] < self.rsi_oversold
            price_below_bb = latest['close'] < latest['bb_lower']
            
            if momentum_up and price_above_sma and volume_surge and rsi_not_overbought:
                return 'BUY_MOMENTUM'
            elif rsi_oversold and price_below_bb:
                return 'BUY_MEAN_REVERSION'
            elif latest['rsi'] > self.rsi_overbought and latest['close'] > latest['bb_upper']:
                return 'SELL'
                
        except Exception as e:
            logger.error(f"Error in momentum strategy for {symbol}: {e}")
            
        return None
    
    def arbitrage_opportunity(self, symbol: str) -> Optional[Dict]:
        """
        Simple arbitrage detection (price discrepancies)
        In practice, this would compare prices across different exchanges
        """
        try:
            # Get current quote
            quote = self.api.get_latest_quote(symbol)
            bid = quote.bid_price
            ask = quote.ask_price
            spread = ask - bid
            
            # Check for unusually wide spreads that might indicate opportunity
            if spread > 0.05:  # 5 cent spread threshold
                return {
                    'symbol': symbol,
                    'bid': bid,
                    'ask': ask,
                    'spread': spread,
                    'opportunity': 'WIDE_SPREAD'
                }
                
        except Exception as e:
            logger.error(f"Error checking arbitrage for {symbol}: {e}")
            
        return None
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return 0
                
            equity = account_info['equity']
            risk_amount = equity * self.risk_per_trade
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share <= 0:
                return 0
                
            # Calculate shares to buy
            shares = int(risk_amount / risk_per_share)
            
            # Check if we have enough buying power
            cost = shares * entry_price
            if cost > account_info['buying_power']:
                shares = int(account_info['buying_power'] * self.position_size / entry_price)
                
            return max(1, shares)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def place_order(self, symbol: str, side: str, shares: int, order_type: str = 'market') -> bool:
        """Place a trading order"""
        try:
            if shares <= 0:
                return False
                
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side=side,
                type=order_type,
                time_in_force='day'
            )
            
            logger.info(f"Order placed: {side} {shares} shares of {symbol} - Order ID: {order.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return False
    
    def manage_positions(self):
        """Manage existing positions - stop losses, take profits"""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                qty = int(position.qty)
                current_price = float(position.current_price)
                unrealized_pl = float(position.unrealized_pl)
                
                # Get market data for the position
                df = self.get_market_data(symbol, '1Min', 100)
                if df.empty:
                    continue
                    
                df = self.calculate_technical_indicators(df)
                latest = df.iloc[-1]
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # Stop loss: 2% loss
                if unrealized_pl < -0.02 * float(position.market_value):
                    should_exit = True
                    exit_reason = "Stop Loss"
                
                # Take profit: 3% gain
                elif unrealized_pl > 0.03 * float(position.market_value):
                    should_exit = True
                    exit_reason = "Take Profit"
                
                # Technical exit conditions
                elif qty > 0 and latest['rsi'] > 75:  # Long position, RSI overbought
                    should_exit = True
                    exit_reason = "RSI Overbought"
                
                elif qty < 0 and latest['rsi'] < 25:  # Short position, RSI oversold
                    should_exit = True
                    exit_reason = "RSI Oversold"
                
                if should_exit:
                    side = 'sell' if qty > 0 else 'buy'
                    if self.place_order(symbol, side, abs(qty)):
                        logger.info(f"Position closed: {symbol} - Reason: {exit_reason}")
                        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    def scan_opportunities(self):
        """Scan for trading opportunities"""
        opportunities = []
        
        for symbol in self.watchlist:
            try:
                # Get market data
                df = self.get_market_data(symbol, '1Min', 200)
                if df.empty:
                    continue
                    
                df = self.calculate_technical_indicators(df)
                
                # Check for momentum/mean reversion signals
                signal = self.momentum_strategy(df, symbol)
                if signal:
                    opportunities.append({
                        'symbol': symbol,
                        'signal': signal,
                        'data': df.iloc[-1]
                    })
                
                # Check for arbitrage opportunities
                arb = self.arbitrage_opportunity(symbol)
                if arb:
                    opportunities.append({
                        'symbol': symbol,
                        'signal': 'ARBITRAGE',
                        'data': arb
                    })
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
                
        return opportunities
    
    def execute_trades(self, opportunities: List[Dict]):
        """Execute trades based on opportunities"""
        current_positions = len(self.api.list_positions())
        
        for opp in opportunities:
            if current_positions >= self.max_positions:
                break
                
            symbol = opp['symbol']
            signal = opp['signal']
            
            try:
                # Check if we already have a position
                try:
                    position = self.api.get_position(symbol)
                    if position:
                        continue  # Skip if we already have a position
                except:
                    pass  # No position exists, continue
                
                # Get current price
                quote = self.api.get_latest_quote(symbol)
                current_price = (quote.bid_price + quote.ask_price) / 2
                
                if signal in ['BUY_MOMENTUM', 'BUY_MEAN_REVERSION']:
                    # Calculate stop loss (2% below entry)
                    stop_loss = current_price * 0.98
                    
                    # Calculate position size
                    shares = self.calculate_position_size(symbol, current_price, stop_loss)
                    
                    if shares > 0:
                        if self.place_order(symbol, 'buy', shares):
                            logger.info(f"Bought {shares} shares of {symbol} at ${current_price:.2f} - Signal: {signal}")
                            current_positions += 1
                
                elif signal == 'ARBITRAGE':
                    # Simple arbitrage execution
                    shares = 100  # Fixed size for arbitrage
                    if self.place_order(symbol, 'buy', shares):
                        logger.info(f"Arbitrage trade: {symbol} - {opp['data']}")
                        current_positions += 1
                        
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
    
    def run_trading_loop(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting trading bot...")
        
        while self.running:
            try:
                # Check market hours
                clock = self.api.get_clock()
                if not clock.is_open:
                    logger.info("Market is closed, waiting...")
                    time.sleep(60)
                    continue
                
                # Manage existing positions
                self.manage_positions()
                
                # Scan for new opportunities
                opportunities = self.scan_opportunities()
                
                if opportunities:
                    logger.info(f"Found {len(opportunities)} opportunities")
                    self.execute_trades(opportunities)
                
                # Wait before next iteration
                time.sleep(30)  # Run every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Stopping trading bot...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def backtest_strategy(self, symbol: str, days: int = 30) -> Dict:
        """Simple backtesting functionality"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.get_market_data(symbol, '1Day', days)
            if df.empty:
                return {}
                
            df = self.calculate_technical_indicators(df)
            
            # Simulate trades
            trades = []
            position = 0
            entry_price = 0
            
            for i in range(50, len(df)):
                current_data = df.iloc[i]
                signal = self.momentum_strategy(df.iloc[:i+1], symbol)
                
                if signal == 'BUY_MOMENTUM' and position == 0:
                    position = 1
                    entry_price = current_data['close']
                    trades.append({
                        'date': current_data.name,
                        'action': 'BUY',
                        'price': entry_price,
                        'signal': signal
                    })
                
                elif signal == 'SELL' and position == 1:
                    position = 0
                    exit_price = current_data['close']
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({
                        'date': current_data.name,
                        'action': 'SELL',
                        'price': exit_price,
                        'pnl': pnl,
                        'signal': signal
                    })
            
            # Calculate performance metrics
            if trades:
                total_pnl = sum([t.get('pnl', 0) for t in trades])
                win_rate = len([t for t in trades if t.get('pnl', 0) > 0]) / len([t for t in trades if 'pnl' in t])
                
                return {
                    'symbol': symbol,
                    'total_trades': len(trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'trades': trades
                }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            
        return {}

# Example usage
if __name__ == "__main__":
    # Initialize bot (use your actual API keys)
    bot = AlpacaTradingBot(
        api_key="YOUR_API_KEY",
        secret_key="YOUR_SECRET_KEY",
        paper_trading=True  # Start with paper trading
    )
    
    # Check account info
    account = bot.get_account_info()
    print(f"Account Equity: ${account.get('equity', 0):,.2f}")
    
    # Run backtest
    backtest_result = bot.backtest_strategy('AAPL', days=60)
    if backtest_result:
        print(f"Backtest Results for {backtest_result['symbol']}:")
        print(f"Total Trades: {backtest_result['total_trades']}")
        print(f"Total P&L: {backtest_result['total_pnl']:.2%}")
        print(f"Win Rate: {backtest_result['win_rate']:.2%}")
    
    # Uncomment to run the trading loop
    # bot.run_trading_loop()
