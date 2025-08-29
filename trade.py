import os
import re
import hmac
import time
import math
import json
import base64
import hashlib
import requests
import pandas as pd
from glob import glob
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

SYMBOL = "BTC"
PAIR = "XBTUSDC" # Kraken spot pair
BASE = "XXBT"   # Kraken asset code for BTC in balances
QUOTE = "USDC"  # Kraken asset code for USD in balances
API_URL = "https://api.kraken.com"

# Reads keys from environment
API_KEY = os.getenv("KRAKEN_API_KEY", "")
API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

# Defines a safety buffer on spend to allow fees/slippage
SPEND_BUFFER = 0.98

# ---------------------------------------------------------------------------------------
# EXECUTION PLAN
#----------------------------------------------------------------------------------------

def get_latest_predictions_file(results_dir):
    """
    Given the results directory if exists returns the latest predictions file in it.
    """
    files = sorted(results_dir.glob("predictions_*.csv"))

    return files[-1] if files else None

def parse_quantiles(qstr: str):
    """
    Parse (lq, uq) string into (lq, uq) floats like.
    """
    clean = qstr.strip()
    lqs, uqs = clean.split("_")

    return float(lqs.strip()), float(uqs.strip())
    
def select_best_advisor(df: pd.DataFrame) -> pd.Series:
    """
    Picks the best model (advisor) based on validation metrics. Returns them too.
    """
    # Calculates quantile width
    lqs, uqs = [], []

    for q in df["quantile"]:
        lq, uq = parse_quantiles(str(q))
        lqs.append(lq)
        uqs.append(uq)

    df = df.copy()
    df["lq"] = lqs
    df["uq"] = uqs
    df["quantile_width"] = df["uq"] - df["lq"]

    # Sorts advisors following arbitrary logic
    df_sorted = df.sort_values(
        by=["quantile_width", "val_drawdown_ewma"],
        ascending=[True, True],
        kind="mergesort"
    )

    return df_sorted.iloc[0]

def calculate_leverage(row: pd.Series) -> float:
    """
    Calculates leverage based on validation metrics so that 2 * max_drawdown * L <= 1.
    """
    min_leverage = 1.0
    max_leverage = 10.0

    # Calculates leverage bound
    L = 1.0 / (2.0 * row["val_drawdown_ewma"])

    if L <= min_leverage:
        return min_leverage
    
    elif L >= max_leverage:
        return max_leverage
    
    else:
        return float(math.floor(L))

def build_execution_plan(results_dir):
    """
    Given the results directory checks if some model has made some decision (trading advice),
    if so, selects the best model based on validation metrics and computes a safe leverage,
    which takes into account maximum drawdown of the strategy.
    """
    # Obtains the daily best models results
    results_path = get_latest_predictions_file(results_dir)
    results = pd.read_csv(results_path)

    # Considers only non-neutral predictions
    mask = (results["decision"].str.lower() != "neutral")
    models_advice = results.loc[mask].copy()

    # All models decide to be Neutral
    if models_advice.empty:
        return None, "Neutral", 1.0

    advisor_info = select_best_advisor(models_advice)

    # Calculates leverage based on advisor validation metrics
    leverage = calculate_leverage(advisor_info)

    return advisor_info, advisor_info["decision"], leverage

# ---------------------------------------------------------------------------------------
# KRAKEN CLIENT
#----------------------------------------------------------------------------------------

class KrakenClient:
    def __init__(self, key: str, secret: str, api_url: str = API_URL):
        self.key = key
        self.secret = base64.b64decode(secret) if secret else b""
        self.api_url = api_url

    def _sign(self, urlpath: str, data: dict) -> str:
        # nonce must be included in data before calling
        postdata = requests.compat.urlencode(data)
        message = (str(data['nonce']) + postdata).encode()
        sha256 = hashlib.sha256(message).digest()
        mac = hmac.new(self.secret, urlpath.encode() + sha256, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())

        return sigdigest.decode()

    def _private(self, method: str, data: Optional[dict] = None):
        urlpath = f"/0/private/{method}"
        url = self.api_url + urlpath
        pdata = data.copy() if data else {}
        pdata["nonce"] = int(time.time() * 1000)

        headers = {
            "API-Key": API_KEY,
            "API-Sign": self._sign(urlpath, pdata)
        }

        resp = requests.post(url, headers=headers, data=pdata, timeout=30)
        resp.raise_for_status()
        out = resp.json()

        if out.get("error"):
            raise RuntimeError(str(out["error"]))
        
        return out.get("result", {})

    def _public(self, method: str, params: Optional[dict] = None):
        url = f"{self.api_url}/0/public/{method}"
        resp = requests.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        out = resp.json()

        if out.get("error"):
            raise RuntimeError(str(out["error"]))
        
        return out.get("result", {})

    def get_balance(self) -> dict:
        return self._private("Balance")

    def get_trade_balance(self) -> dict:
        return self._private("TradeBalance")

    def get_open_positions(self, txid: str, docalcs: bool = True) -> dict:
        data = {"docalcs": "true" if docalcs else "false"}

        if txid:
            data["txid"] = txid

        return self._private("OpenPositions", data)

    def get_open_orders(self) -> dict:
        return self._private("OpenOrders")

    def get_closed_orders(self) -> dict:
        return self._private("ClosedOrders")

    def add_order(self, pair: str, side: str, ordertype: str, volume: float,
                  leverage: float, oflags: str, reduce_only: bool = False):
        data = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": f"{volume:.10f}"
        }

        if leverage and leverage > 1.0:
            # Kraken expects strings like "3:1"
            data["leverage"] = f"{int(round(leverage))}:1"

            # reduce_only only applies to margin orders
            if reduce_only:
                data["reduce_only"] = "true"

        else:
            # spot order without leverage
            data["leverage"] = "none"
            
            if oflags:
                data["oflags"] = oflags

        return self._private("AddOrder", data)

    def cancel_all_orders(self):
        return self._private("CancelAll")

    def get_ticker_price(self, pair: str) -> float:
        res = self._public("Ticker", {"pair": pair})
        k = next(iter(res))
        price = float(res[k]["c"][0])  # Last trade closed

        return price
    
# ---------------------------------------------------------------------------------------
# AUXILIAR FUNCTIONS
#----------------------------------------------------------------------------------------

def now_iso():
    # timezone-aware UTC timestamp
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_state(state_path: str):
    """
    Loads strategy state if exists or creates a clear new one.
    """
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        
        except Exception:
            pass

    return {
        "is_open": False,
        "side": None,
        "open_time": None,
        "days_open": 1,
        "target_days": 0,
        "leverage": 1.0
    }

def save_state(state_path: str, state: dict):
    """
    Gievn a strategy state, saves it.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))

def parse_target_days(target_str: str) -> int:
    """
    Parses target holding period based on the model forward prediction horizon.
    """
    s = str(target_str).strip()
    m = re.search(r'(\d+)', s)
    iv = int(m.group(1))
    
    return iv if iv > 0 else None

# ---------------------------------------------------------------------------------------
# TRADING LOGIC
#----------------------------------------------------------------------------------------

def calculate_leveraged_order_volume(k: KrakenClient, leverage: float) -> float:
    """
    Given Kraken Exchange status (namely balance) and leverage, calculates the right order 
    volume to consider when using leverage.
    """
    bal = k.get_balance()
    quote_avail = float(bal.get(QUOTE, "0"))

    if quote_avail <= 0:
        return 0.0
    
    px = k.get_ticker_price(PAIR)
    spend = quote_avail * SPEND_BUFFER

    return (spend * max(leverage, 1.0)) / px

def calculate_order_volume(k: KrakenClient) -> float:
    """
    Given Kraken Exchange status (namely balance), calculates the right order volume to 
    consider when not using leverage.
    """
    bal = k.get_balance()
    base_avail = float(bal.get(BASE, "0"))

    return base_avail * SPEND_BUFFER

def close_existing_position(k: KrakenClient, state: dict):
    """
    Given Kraken Exchange status and 
    """
    # Tries to detect open margin position first
    try:
        pos = k.get_open_positions(docalcs=True)

        for _, p in pos.items():
            if p.get("pair", "").upper() == PAIR and float(p.get("vol", 0)) > 0:
                vol_to_close = float(p["vol"])
                side = "buy" if p.get("type") == "short" else "sell"
                k.add_order(PAIR, side, "market", vol_to_close, leverage=max(state.get("leverage", 1.0), 1.0), reduce_only=True)

                return
            
    except Exception:
        pass

    # Now spot open positions (closed buying the inverse order)
    if state.get("side") == "buy":
        vol = calculate_order_volume(k)


        if vol > 0:
            k.add_order(PAIR, "sell", "market", vol, leverage=None)

    elif state.get("side") == "sell":
        vol = calculate_leveraged_order_volume(k, leverage=max(state.get("leverage", 1.0), 1.0))

        if vol > 0:
            k.add_order(PAIR, "buy", "market", vol, leverage=max(state.get("leverage", 1.0), 1.0), reduce_only=True)

def open_new_position(k: KrakenClient, side: str, leverage: float):
    """
    
    """
    if side == "buy":
        vol = calculate_leveraged_order_volume(k, leverage=leverage)

        if vol > 0:
            k.add_order(PAIR, "buy", "market", vol, leverage=leverage if leverage and leverage > 1.0 else None,
                        oflags=None if (leverage and leverage > 1.0) else "viqc")
    else:
        if leverage and leverage > 1.0:
            vol = calculate_leveraged_order_volume(k, leverage=leverage)

            if vol > 0:
                k.add_order(PAIR, "sell", "market", vol, leverage=leverage)

        else:
            vol = calculate_order_volume(k)

            if vol > 0:
                k.add_order(PAIR, "sell", "market", vol, leverage=None)

def trade(advisor_info: pd.Series, decision: str, leverage: float, state_path: str):
    """
    
    """
    k = KrakenClient(API_KEY, API_SECRET)
    dec = str(decision).strip().lower()
    side = "buy" if dec == "long" else ("sell" if dec == "short" else None)
    state = load_state(state_path)
    is_open = bool(state.get("is_open", False))
    cur_side = state.get("side")
    days_open = int(state.get("days_open", 0))
    cur_tgt = int(state.get("target_days", 0))

    # Neutral logic
    if dec == "neutral":
        if not is_open:
            return  # Nothing to do
        
        # Order life has expired, order is closed
        if days_open >= max(cur_tgt, 1) - 1:
            close_existing_position(k, state)
            state.update(
                {
                    "is_open": False,
                    "side": None,
                    "open_time": None,
                    "days_open": 0,
                    "target_days": 0,
                    "leverage": 1.0
                }
            )

        # Order life and target horizon are updated
        else:
            state["days_open"] = days_open + 1

        save_state(state_path, state)

        return
    
    else:
        target_days = parse_target_days(advisor_info["target"])

        # Long / Short logic
        if is_open:
            # Same side signal, order life is reinitialized
            if cur_side == side:
                state["days_open"] = 1
                state["target_days"] = max(int(target_days), 1)

            else:
                # Flips order: closes old, opens new, resets life
                close_existing_position(k, state)
                open_new_position(k, side, leverage)
                state.update({
                    "is_open": True,
                    "side": side,
                    "open_time": now_iso(),
                    "days_open": 1,
                    "target_days": max(int(target_days), 1),
                    "leverage": float(leverage)
                })
        else:
            # No existing position: opens new, sets horizon
            open_new_position(k, side, leverage)
            state.update(
                {
                    "is_open": True,
                    "side": side,
                    "open_time": now_iso(),
                    "days_open": 1,
                    "target_days": max(int(target_days), 1),
                    "leverage": float(leverage)
                }
            )

    save_state(state_path, state)

# ---------------------------------------------------------------------------------------
# MAIN
#----------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Defines paths
    logs_dir = Path("logs")
    results_dir = Path(f"{logs_dir}/{SYMBOL}/results")
    state_path = Path(f"{logs_dir}/{SYMBOL}/state.json")

    # Builds execution plan based on best advisor decision and validation metrics
    advisor_info, decision, leverage = build_execution_plan(results_dir)

    # Trades based on the decision plan, managing open orders
    trade(advisor_info, decision, leverage, state_path)