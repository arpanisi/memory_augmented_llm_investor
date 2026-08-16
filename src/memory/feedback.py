"""
Step 4 Feedback Loop module.
Applies L=5 day cumulative percentage return look-back feedback (+-18) to retrieved memory IDs.
"""
import pandas as pd
from config.settings import FEEDBACK_LOOKBACK_DAYS, FEEDBACK_DELTA_IMPORTANCE
from src.memory.store import AssetMemoryStore

class FeedbackTracker:
    """
    Maintains rolling records of decisions and retrieved memory IDs for L-day look-back feedback.
    """
    def __init__(self, lookback_days: int = FEEDBACK_LOOKBACK_DAYS):
        self.lookback_days = lookback_days
        # History keyed by date: {date_str: {asset: {"memory_ids": [...], "position": 1.0/-1.0/0.0, "price": float}}}
        self.decision_history: dict[str, dict[str, dict]] = {}
        self.trading_days: list[str] = []

    def record_decision(self, date_str: str, asset: str, memory_ids: list[str], direction: str, close_price: float):
        """
        Records the memory IDs, position sign, and closing price for asset on decision date.
        """
        if date_str not in self.decision_history:
            self.decision_history[date_str] = {}
            if date_str not in self.trading_days:
                self.trading_days.append(date_str)
                self.trading_days.sort()

        pos_map = {"long": 1.0, "short": -1.0, "flat": 0.0}
        self.decision_history[date_str][asset] = {
            "memory_ids": memory_ids,
            "position": pos_map.get(direction.lower(), 0.0),
            "close_price": close_price
        }

    def process_feedback_for_day(self, current_date_str: str, asset_stores: dict[str, AssetMemoryStore], price_pivot: pd.DataFrame):
        """
        Processes feedback for the decision made L days prior to current_date_str.

        The full chronological trading-day sequence is tracked independently of whether today's own
        decision has been recorded yet: current_date_str is appended to self.trading_days before the
        lookback arithmetic, so feedback is evaluated as step (1) of a decision day — before that
        day's decision is recorded (the call order used by the orchestrator).
        """
        if current_date_str not in self.trading_days:
            self.trading_days.append(current_date_str)
            self.trading_days.sort()

        current_idx = self.trading_days.index(current_date_str)
        if current_idx < self.lookback_days:
            return  # Not enough days elapsed yet

        eval_date_str = self.trading_days[current_idx - self.lookback_days]
        eval_record = self.decision_history.get(eval_date_str, {})

        # Evaluate window [eval_idx, current_idx]
        window_dates = self.trading_days[current_idx - self.lookback_days : current_idx + 1]

        for asset, store in asset_stores.items():
            asset_dec = eval_record.get(asset)
            if not asset_dec or not asset_dec["memory_ids"] or asset_dec["position"] == 0.0:
                continue

            pos = asset_dec["position"]
            memory_ids = asset_dec["memory_ids"]

            # Compute cumulative percentage return over window_dates
            cum_pct_return = 0.0
            for i in range(len(window_dates) - 1):
                d_start = window_dates[i]
                d_end = window_dates[i + 1]
                
                p_start = None
                p_end = None
                
                if price_pivot is not None and asset in price_pivot.columns:
                    p_start = price_pivot.loc[d_start, asset] if d_start in price_pivot.index else None
                    p_end = price_pivot.loc[d_end, asset] if d_end in price_pivot.index else None
                
                if p_start and p_end and p_start > 0:
                    daily_ret = (p_end - p_start) / p_start
                    cum_pct_return += daily_ret * pos

            # Apply feedback update
            if cum_pct_return > 0:
                store.apply_feedback(memory_ids, +FEEDBACK_DELTA_IMPORTANCE)
            elif cum_pct_return < 0:
                store.apply_feedback(memory_ids, -FEEDBACK_DELTA_IMPORTANCE)
