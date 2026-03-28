# app.py (versão monolítica - apenas para teste local)
from flask import Flask, request, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np
import threading
import queue
import uuid
import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIG ====================
class Config:
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
    MAX_CONCURRENT_TRADES = int(os.getenv('MAX_CONCURRENT_TRADES', 5))
    ATR_PERIOD = int(os.getenv('ATR_PERIOD', 14))
    ATR_FACTOR = float(os.getenv('ATR_FACTOR', 2.0))
    DEFAULT_RISK = float(os.getenv('DEFAULT_RISK', 1.0))
    RISK_LEVELS = {8.0: 2.0, 6.0: 1.2, 4.0: 0.6, 0.0: 0.5}
    TP_DISTRIBUTION = {1: 0.20, 2: 0.30, 3: 0.30, 4: 0.20}
    MONITOR_INTERVAL = 1
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ==================== LOGGER ====================
class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
    def info(self, message, extra=None):
        log_entry = {"message": message}
        if extra:
            log_entry.update(extra)
        self.logger.info(json.dumps(log_entry))
    def error(self, message, extra=None):
        log_entry = {"message": message}
        if extra:
            log_entry.update(extra)
        self.logger.error(json.dumps(log_entry))
    def warning(self, message, extra=None):
        log_entry = {"message": message}
        if extra:
            log_entry.update(extra)
        self.logger.warning(json.dumps(log_entry))

logger = StructuredLogger("bot")

# ==================== INDICATORS ====================
class ATRCalculator:
    def __init__(self):
        self.client = Client(Config.BINANCE_API_KEY, Config.BINANCE_SECRET_KEY)
    def get_atr(self, symbol=Config.SYMBOL, period=Config.ATR_PERIOD):
        klines = self.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=period+1)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df[['high', 'low', 'close']] = df[['high', 'low', 'close']].astype(float)
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        atr = df['tr'].rolling(window=period).mean().iloc[-1]
        return atr
    def get_candles(self, symbol=Config.SYMBOL, limit=50):
        klines = self.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=limit)
        return [{'timestamp': k[0], 'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])} for k in klines]

# ==================== RISK ====================
class RiskManager:
    def __init__(self):
        self.atr_calc = ATRCalculator()
    def calculate_position_size(self, score, account_balance):
        risk_percent = 1.0
        for threshold, risk in sorted(Config.RISK_LEVELS.items(), reverse=True):
            if score >= threshold:
                risk_percent = risk
                break
        return account_balance * (risk_percent / 100)
    def calculate_smart_stop(self, entry_price, direction, atr, candles):
        atr_stop = entry_price - (atr * Config.ATR_FACTOR) if direction == 'BUY' else entry_price + (atr * Config.ATR_FACTOR)
        if direction == 'BUY':
            technical_stop = min([c['low'] for c in candles[-10:]])
        else:
            technical_stop = max([c['high'] for c in candles[-10:]])
        return {'atr_stop': atr_stop, 'technical_stop': technical_stop, 'initial_sl': atr_stop if direction == 'BUY' else technical_stop}
    def calculate_take_profits(self, entry_price, direction, atr):
        if direction == 'BUY':
            return [entry_price + (atr * 1.0), entry_price + (atr * 1.5), entry_price + (atr * 2.0), entry_price + (atr * 3.0)]
        else:
            return [entry_price - (atr * 1.0), entry_price - (atr * 1.5), entry_price - (atr * 2.0), entry_price - (atr * 3.0)]

# ==================== STATE ====================
class TradeState:
    def __init__(self):
        self.active_trades = {}
        self.lock = threading.Lock()
    def create_trade(self, trade_data):
        with self.lock:
            trade_id = str(uuid.uuid4())
            self.active_trades[trade_id] = {'id': trade_id, 'created_at': datetime.now(), 'status': 'ACTIVE', 'sl_state': 'NORMAL', 'tp_hit': [], 'partial_fills': [], 'reentry': False, **trade_data}
            return trade_id
    def get_trade(self, trade_id):
        with self.lock:
            return self.active_trades.get(trade_id)
    def update_trade(self, trade_id, updates):
        with self.lock:
            if trade_id in self.active_trades:
                self.active_trades[trade_id].update(updates)
    def remove_trade(self, trade_id):
        with self.lock:
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
    def get_all_active(self):
        with self.lock:
            return {k: v for k, v in self.active_trades.items() if v['status'] == 'ACTIVE'}
    def get_active_count(self):
        with self.lock:
            return len([t for t in self.active_trades.values() if t['status'] == 'ACTIVE'])

# ==================== QUEUE ====================
class OrderQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = False
    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info("Order queue started")
    def stop(self):
        self.running = False
    def add_order(self, order_func, order_data):
        self.queue.put((order_func, order_data))
        logger.info(f"Order added: {order_data}")
    def _process_queue(self):
        while self.running:
            try:
                order_func, order_data = self.queue.get(timeout=1)
                try:
                    result = order_func(order_data)
                    logger.info(f"Order executed: {result}")
                except Exception as e:
                    logger.error(f"Order failed: {e}")
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue

# ==================== TRADER ENGINE ====================
class TradingEngine:
    def __init__(self):
        self.client = Client(Config.BINANCE_API_KEY, Config.BINANCE_SECRET_KEY)
        self.risk_manager = RiskManager()
        self.atr_calculator = ATRCalculator()
        self.state = TradeState()
        self.order_queue = OrderQueue()
        self.running = False
        self.order_queue.start()
        self.start_monitor()
    def process_signal(self, signal_data):
        try:
            if self.state.get_active_count() >= Config.MAX_CONCURRENT_TRADES:
                logger.warning("Max trades reached")
                return
            atr = self.atr_calculator.get_atr()
            candles = self.atr_calculator.get_candles()
            account_info = self.client.get_account()
            usdt_balance = 0.0
            for balance in account_info['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            position_size = self.risk_manager.calculate_position_size(signal_data['score'], usdt_balance)
            ticker = self.client.get_symbol_ticker(symbol=Config.SYMBOL)
            current_price = float(ticker['price'])
            stop_info = self.risk_manager.calculate_smart_stop(current_price, signal_data['action'], atr, candles)
            tp_levels = self.risk_manager.calculate_take_profits(current_price, signal_data['action'], atr)
            trade_data = {
                'action': signal_data['action'], 'score': signal_data['score'], 'hunt': signal_data.get('hunt', False),
                'trend': signal_data.get('trend', 'normal'), 'volatility': signal_data.get('volatility', 'normal'),
                'entry_price': current_price, 'position_size': position_size, 'atr': atr,
                'sl_atr': stop_info['atr_stop'], 'sl_technical': stop_info['technical_stop'],
                'current_sl': stop_info['initial_sl'], 'tp_levels': tp_levels, 'tp_hit_count': 0,
                'be_activated': False, 'trailing_activated': False, 'sl_state': 'NORMAL', 'candles_in_tolerance': 0
            }
            trade_id = self.state.create_trade(trade_data)
            self.order_queue.add_order(self._execute_entry, {'trade_id': trade_id, 'action': signal_data['action'], 'size': position_size})
            logger.info(f"Trade created: {trade_id}")
        except Exception as e:
            logger.error(f"Signal error: {e}")
    def _execute_entry(self, order_data):
        trade = self.state.get_trade(order_data['trade_id'])
        if not trade:
            raise ValueError("Trade not found")
        side = Client.SIDE_BUY if trade['action'] == 'BUY' else Client.SIDE_SELL
        order = self.client.create_order(symbol=Config.SYMBOL, side=side, type=Client.ORDER_TYPE_MARKET, quantity=self._adjust_quantity(trade['position_size']))
        self._place_stop_loss(order_data['trade_id'], trade)
        return {'order_id': order['orderId']}
    def _place_stop_loss(self, trade_id, trade):
        side = Client.SIDE_SELL if trade['action'] == 'BUY' else Client.SIDE_BUY
        self.client.create_order(symbol=Config.SYMBOL, side=side, type=Client.ORDER_TYPE_STOP_LOSS_LIMIT, quantity=self._adjust_quantity(trade['position_size']), price=trade['current_sl'], stopPrice=trade['current_sl'], timeInForce=Client.TIME_IN_FORCE_GTC)
    def start_monitor(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    def _monitor_loop(self):
        while self.running:
            try:
                for trade_id, trade in self.state.get_all_active().items():
                    self._check_take_profits(trade_id, trade)
                    self._check_stop_loss(trade_id, trade)
                    self._update_trailing(trade_id, trade)
                time.sleep(Config.MONITOR_INTERVAL)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    def _check_take_profits(self, trade_id, trade):
        current_price = self._get_current_price()
        direction = 1 if trade['action'] == 'BUY' else -1
        for level in range(trade['tp_hit_count'] + 1, 5):
            tp_price = trade['tp_levels'][level - 1]
            if (direction == 1 and current_price >= tp_price) or (direction == -1 and current_price <= tp_price):
                close_size = trade['position_size'] * Config.TP_DISTRIBUTION[level]
                self._close_position(trade_id, close_size, f"TP{level}")
                trade['tp_hit_count'] = level
                logger.info(f"TP{level} HIT", extra={'trade_id': trade_id})
                if level == 2 and not trade['be_activated']:
                    self._activate_break_even(trade_id, trade)
                if level == 3 and not trade['trailing_activated']:
                    trade['trailing_activated'] = True
                break
        if trade['tp_hit_count'] >= 4:
            self._close_position(trade_id, trade['position_size'], "FULL CLOSE")
            self.state.remove_trade(trade_id)
    def _check_stop_loss(self, trade_id, trade):
        current_price = self._get_current_price()
        direction = 1 if trade['action'] == 'BUY' else -1
        if (direction == 1 and current_price <= trade['sl_atr']) or (direction == -1 and current_price >= trade['sl_atr']):
            self._close_position(trade_id, trade['position_size'], "ATR STOP")
            self.state.remove_trade(trade_id)
            return
        if (direction == 1 and current_price <= trade['sl_technical']) or (direction == -1 and current_price >= trade['sl_technical']):
            if trade['sl_state'] == 'NORMAL':
                trade['sl_state'] = 'TOLERANCIA_ATIVA'
                trade['candles_in_tolerance'] = 0
            elif trade['sl_state'] == 'TOLERANCIA_ATIVA':
                trade['candles_in_tolerance'] += 1
                if trade['candles_in_tolerance'] >= 3:
                    self._close_position(trade_id, trade['position_size'], "TECH STOP")
                    self.state.remove_trade(trade_id)
        elif (direction == 1 and current_price > trade['entry_price']) or (direction == -1 and current_price < trade['entry_price']):
            if trade['sl_state'] == 'TOLERANCIA_ATIVA':
                trade['sl_state'] = 'NORMAL'
    def _activate_break_even(self, trade_id, trade):
        trade['be_activated'] = True
        trade['current_sl'] = trade['entry_price']
        self._update_stop_loss(trade_id, trade['entry_price'])
        logger.info(f"BE activated", extra={'trade_id': trade_id})
    def _update_trailing(self, trade_id, trade):
        if not trade['trailing_activated']:
            return
        candles = self.atr_calculator.get_candles(limit=20)
        if trade['action'] == 'BUY':
            highest_low = max([c['low'] for c in candles[-10:]])
            if highest_low > trade['current_sl']:
                trade['current_sl'] = highest_low
                self._update_stop_loss(trade_id, highest_low)
        else:
            lowest_high = min([c['high'] for c in candles[-10:]])
            if lowest_high < trade['current_sl']:
                trade['current_sl'] = lowest_high
                self._update_stop_loss(trade_id, lowest_high)
    def _update_stop_loss(self, trade_id, new_sl):
        trade = self.state.get_trade(trade_id)
        if not trade:
            return
        side = Client.SIDE_SELL if trade['action'] == 'BUY' else Client.SIDE_BUY
        open_orders = self.client.get_open_orders(symbol=Config.SYMBOL)
        for order in open_orders:
            if order['type'] == 'STOP_LOSS_LIMIT':
                self.client.cancel_order(symbol=Config.SYMBOL, orderId=order['orderId'])
        self.client.create_order(symbol=Config.SYMBOL, side=side, type=Client.ORDER_TYPE_STOP_LOSS_LIMIT, quantity=self._adjust_quantity(trade['position_size']), price=new_sl, stopPrice=new_sl, timeInForce=Client.TIME_IN_FORCE_GTC)
    def _close_position(self, trade_id, size, reason):
        trade = self.state.get_trade(trade_id)
        if not trade:
            return
        side = Client.SIDE_SELL if trade['action'] == 'BUY' else Client.SIDE_BUY
        self.client.create_order(symbol=Config.SYMBOL, side=side, type=Client.ORDER_TYPE_MARKET, quantity=self._adjust_quantity(size))
        logger.info(f"Closed: {reason}", extra={'trade_id': trade_id})
    def _get_current_price(self):
        return float(self.client.get_symbol_ticker(symbol=Config.SYMBOL)['price'])
    def _adjust_quantity(self, size):
        return round(size, 5)
    def get_active_count(self):
        return self.state.get_active_count()

# ==================== FLASK APP ====================
app = Flask(__name__)
engine = TradingEngine()

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "running", "active_trades": engine.get_active_count()}), 200

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400
        if 'action' not in data or 'score' not in data:
            return jsonify({"error": "Missing fields"}), 400
        if data['action'].upper() not in ['BUY', 'SELL']:
            return jsonify({"error": "Invalid action"}), 400
        engine.process_signal(data)
        return jsonify({"status": "queued"}), 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
import threading

def bot_loop():
    print("🔥 BOT rodando...")
    while True:
        pass  # depois coloca lógica

threading.Thread(target=bot_loop, daemon=True).start()
