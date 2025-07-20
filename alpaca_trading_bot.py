#!/usr/bin/env python3
"""
Server-optimized Alpaca Trading Bot for OCI VM deployment
Modified to use 'ta' library instead of TA-Lib
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
import os
import json
import signal
import sys
from typing import Dict, List, Tuple, Optional
import asyncio
import websocket
from concurrent.futures import ThreadPoolExecutor
import ta  # Using 'ta' instead of talib
from dotenv import load_dotenv
import schedule
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

class TradingBotLogger:
    def __init__(self, log_file='trading_bot.log'):
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger

class DatabaseManager:
    def __init__(self, db_path='trading_bot.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing trades and performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                strategy TEXT NOT NULL,
                pnl REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN'
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE DEFAULT CURRENT_DATE,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                account_value REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_trade(self, symbol, side, quantity, price, strategy):
        """Log a trade to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, side, quantity, price, strategy)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, side, quantity, price, strategy))
        conn.commit()
        conn.close()
    
    def update_trade_pnl(self, symbol, pnl, status='CLOSED'):
        """Update trade P&L when position is closed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE trades SET pnl = ?, status = ?
            WHERE symbol = ? AND status = 'OPEN'
            ORDER BY timestamp DESC LIMIT 1
        ''', (pnl, status, symbol))
        conn.commit()
        conn.close()

class EmailNotifier:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email = os.getenv('NOTIFICATION_EMAIL')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.enabled = self.email and self.password
    
    def send_notification(self, subject: str, message: str):
        """Send email notification"""
        if not self.enabled:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
                
        except Exception as e:
            print(f"Failed to send notification: {e}")

class ServerTradingBot:
    def __init__(self):
        self.logger = TradingBotLogger().get_logger()
        self.db = DatabaseManager()
        self.notifier = EmailNotifier()
        
        # Initialize Alpaca API
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment variables")
        
        base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        # Trading parameters (configurable via environment)
        self.position_size = float(os.getenv('POSITION_SIZE', '0.02'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '10'))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.01'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.02'))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '0.03'))
        
        # Strategy parameters
        self.momentum_lookback = int(os.getenv('MOMENTUM_LOOKBACK', '20'))
        self.rsi_oversold = int(os.getenv('RSI_OVERSOLD', '30'))
        self.rsi_overbought = int(os.getenv('RSI_OVERBOUGHT', '70'))
        
        # Watchlist
        watchlist_str = os.getenv('WATCHLIST', 'AAPL,GOOGL,MSFT,AMZN,TSLA,META,NVDA,SPY,QQQ')
        self.watchlist = [symbol.strip() for symbol in watchlist_str.split(',')]
        
        # State management
        self.running = False
        self.daily_pnl = 0
        self.daily_max_loss = float(os.getenv('DAILY_MAX_LOSS', '0.05'))  # 5% max daily loss
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Trading bot initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def get_account_info(self) -> Dict:
        """Get account information with error handling"""
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'day_trade_count': account.day_trade_count,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        account_info = self.get_account_info()
        if not account_info:
            return False
        
        # Calculate daily P&L (simplified - you might want to track this more precisely)
        current_equity = account_info['equity']
        
        # If loss exceeds daily limit, stop trading
        if self.daily_pnl < -self.daily_max_loss * current_equity:
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            self.notifier.send_notification(
                "Trading Bot - Daily Loss Limit",
                f"Daily loss limit reached: ${self.daily_pnl:.2f}. Trading suspended."
            )
            return True
        
        return False
    
    def get_market_data(self, symbol: str, timeframe: str = '1Min', limit: int = 1000) -> pd.DataFrame:
        """Get historical market data with caching"""
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
            
            if not bars:
                return pd.DataFrame()
            
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
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using 'ta' library instead of talib"""
        if df.empty or len(df) < 50:
            return df
            
        try:
            # Moving averages using ta library
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd_line = ta.trend.macd(df['close'])
            macd_signal = ta.trend.macd_signal(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_line - macd_signal
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            
            # Average True Range
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Volume indicators
            df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
            
            # Custom indicators
            df['price_change'] = df['close'].pct_change()
            df['momentum'] = df['close'] / df['close'].shift(self.momentum_lookback) - 1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def momentum_strategy(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """Enhanced momentum strategy with multiple confirmations"""
        if df.empty or len(df) < 50:
            return None
            
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Check for NaN values
            required_fields = ['momentum', 'close', 'sma_20', 'volume', 'volume_sma', 'rsi', 'bb_lower', 'bb_upper']
            if any(pd.isna(latest[field]) for field in required_fields):
                return None
            
            # Momentum conditions
            momentum_strong = latest['momentum'] > 0.02  # 2% momentum
            price_above_sma = latest['close'] > latest['sma_20']
            volume_surge = latest['volume'] > latest['volume_sma'] * 1.5
            rsi_not_overbought = latest['rsi'] < 75
            macd_bullish = latest['macd'] > latest['macd_signal'] if not pd.isna(latest['macd']) else False
            
            # Mean reversion conditions
            rsi_oversold = latest['rsi'] < self.rsi_oversold
            price_below_bb = latest['close'] < latest['bb_lower']
            oversold_bounce = latest['close'] > prev['close'] and rsi_oversold
            
            # Exit conditions
            rsi_overbought_exit = latest['rsi'] > self.rsi_overbought
            price_above_bb = latest['close'] > latest['bb_upper']
            
            # Entry signals
            if momentum_strong and price_above_sma and volume_surge and rsi_not_overbought and macd_bullish:
                return 'BUY_MOMENTUM'
            elif oversold_bounce and price_below_bb:
                return 'BUY_MEAN_REVERSION'
            elif rsi_overbought_exit and price_above_bb:
                return 'SELL'
                
        except Exception as e:
            self.logger.error(f"Error in momentum strategy for {symbol}: {e}")
            
        return None
    
    def place_order(self, symbol: str, side: str, shares: int, strategy: str) -> bool:
        """Place order with database logging"""
        try:
            if shares <= 0:
                return False
            
            # Get current price for logging
            quote = self.api.get_latest_quote(symbol)
            current_price = (quote.bid_price + quote.ask_price) / 2
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            # Log to database
            self.db.log_trade(symbol, side, shares, current_price, strategy)
            
            self.logger.info(f"Order placed: {side} {shares} shares of {symbol} at ${current_price:.2f} - Strategy: {strategy}")
            
            # Send notification for significant trades
            if shares * current_price > 1000:  # Notify for trades > $1000
                self.notifier.send_notification(
                    f"Trading Bot - Large Trade",
                    f"{side.upper()} {shares} shares of {symbol} at ${current_price:.2f}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing order for {symbol}: {e}")
            return False
    
    def manage_positions(self):
        """Enhanced position management with database updates"""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                qty = int(position.qty)
                current_price = float(position.current_price)
                unrealized_pl = float(position.unrealized_pl)
                cost_basis = float(position.cost_basis)
                
                # Calculate percentage P&L
                pnl_pct = unrealized_pl / abs(cost_basis) if cost_basis != 0 else 0
                
                # Get technical data
                df = self.get_market_data(symbol, '1Min', 100)
                if df.empty:
                    continue
                    
                df = self.calculate_technical_indicators(df)
                if df.empty or len(df) < 2:
                    continue
                    
                latest = df.iloc[-1]
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # Stop loss
                if pnl_pct < -self.stop_loss_pct:
                    should_exit = True
                    exit_reason = "Stop Loss"
                
                # Take profit
                elif pnl_pct > self.take_profit_pct:
                    should_exit = True
                    exit_reason = "Take Profit"
                
                # Technical exits
                elif qty > 0 and not pd.isna(latest['rsi']) and latest['rsi'] > 75:
                    should_exit = True
                    exit_reason = "RSI Overbought"
                
                elif qty < 0 and not pd.isna(latest['rsi']) and latest['rsi'] < 25:
                    should_exit = True
                    exit_reason = "RSI Oversold"
                
                if should_exit:
                    side = 'sell' if qty > 0 else 'buy'
                    if self.place_order(symbol, side, abs(qty), f"EXIT_{exit_reason}"):
                        # Update database with final P&L
                        self.db.update_trade_pnl(symbol, unrealized_pl, 'CLOSED')
                        self.daily_pnl += unrealized_pl
                        
                        self.logger.info(f"Position closed: {symbol} - P&L: ${unrealized_pl:.2f} - Reason: {exit_reason}")
                        
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
    
    def health_check(self):
        """Perform system health checks"""
        try:
            # Check API connection
            self.api.get_clock()
            
            # Check account status
            account = self.get_account_info()
            if not account:
                raise Exception("Cannot access account information")
            
            # Check if account is restricted
            if account.get('trading_blocked', False):
                raise Exception("Account trading is blocked")
            
            # Check buying power
            if account.get('buying_power', 0) < 100:
                self.logger.warning("Low buying power detected")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.notifier.send_notification(
                "Trading Bot - Health Check Failed",
                f"Bot health check failed: {str(e)}"
            )
            return False
    
    def daily_report(self):
        """Generate and send daily performance report"""
        try:
            account_info = self.get_account_info()
            positions = self.api.list_positions()
            
            report = f"""
Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}

Account Summary:
- Equity: ${account_info.get('equity', 0):,.2f}
- Cash: ${account_info.get('cash', 0):,.2f}
- Day Trade Count: {account_info.get('day_trade_count', 0)}

Active Positions: {len(positions)}
Daily P&L: ${self.daily_pnl:.2f}

Bot Status: {'Running' if self.running else 'Stopped'}
"""
            
            self.logger.info("Daily report generated")
            self.notifier.send_notification("Trading Bot - Daily Report", report)
            
            # Reset daily P&L
            self.daily_pnl = 0
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    def scan_and_trade(self):
        """Main scanning and trading logic"""
        try:
            # Check if we should continue trading
            if self.check_daily_loss_limit():
                return
            
            # Manage existing positions first
            self.manage_positions()
            
            # Check current position count
            current_positions = len(self.api.list_positions())
            if current_positions >= self.max_positions:
                self.logger.info(f"Maximum positions ({self.max_positions}) reached")
                return
            
            # Scan for opportunities
            opportunities = []
            
            for symbol in self.watchlist:
                try:
                    # Skip if we already have a position
                    try:
                        position = self.api.get_position(symbol)
                        if position:
                            continue
                    except:
                        pass  # No position, continue
                    
                    # Get market data
                    df = self.get_market_data(symbol, '1Min', 200)
                    if df.empty:
                        continue
                    
                    df = self.calculate_technical_indicators(df)
                    if df.empty:
                        continue
                    
                    # Check for signals
                    signal = self.momentum_strategy(df, symbol)
                    if signal and signal.startswith('BUY'):
                        opportunities.append({
                            'symbol': symbol,
                            'signal': signal,
    except Exception as e:
        print(f"Failed to start trading bot: {e}")
        sys.exit(1)
