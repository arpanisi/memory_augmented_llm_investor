"""
Daily Orchestration Loop module tying Steps 1-9 together into a deterministic chronological daily runbook.
Fully compliant with Step 8 specification in coding-plan.md.
"""
import logging

import pandas as pd
import numpy as np
from config.settings import (
    WARMUP_START_DATE, WARMUP_END_DATE, TEST_START_DATE, TEST_END_DATE,
    LAMBDA_GRID, REFLECTION_INTERVAL_DAYS, FACTOR_ESTIMATION_UNIVERSE,
    FACTOR_ESTIMATION_UNIVERSE_SIZE
)
from src.data.universe import build_full_universe
from src.data.estimation_universe import build_estimation_universe
from src.data.prices import get_pivot_close_prices
from src.data.edgar import (
    get_point_in_time_fundamentals, get_10k_filing_documents, get_filing_text
)
from src.data.barra_estimation import compute_barra_model
from src.data.barra_exposures import compute_asset_factor_exposures
from src.memory.store import AssetMemoryStore
from src.memory.migration import run_layer_migration
from src.memory.feedback import FeedbackTracker
from src.memory.reflection import process_reflection_cycle
from src.agent.view_formation import call_view_formation_agent
from src.graph.relationships import (
    CompanyRelationshipGraph, build_candidate_tickers, extract_relationships_from_filing
)
from src.portfolio.shrinkage import compute_ledoit_wolf_constant_correlation
from src.portfolio.optimizer import solve_portfolio_optimization
from src.reporting.metrics import (
    compute_portfolio_breadth, compute_realized_factor_exposures, compute_peer_concentration_exposure
)

logger = logging.getLogger(__name__)


def derive_decision_calendar(price_pivot: pd.DataFrame, assets: list[str], reference_assets: list[str]) -> list[str]:
    """
    Build the daily decision calendar from a real equity/ETF trading series, not from the
    unioned multi-asset pivot. Crypto may have weekend rows, but those rows must not become
    portfolio decision days.
    """
    for asset in reference_assets:
        if asset in assets and asset in price_pivot.columns:
            series = price_pivot[asset].dropna()
            if not series.empty:
                return [str(d) for d in series.index.tolist()]

    intersection_dates: set[str] | None = None
    for asset in assets:
        if asset not in price_pivot.columns:
            continue
        dates = {str(d) for d in price_pivot.index[price_pivot[asset].notna()].tolist()}
        if not dates:
            continue
        intersection_dates = dates if intersection_dates is None else intersection_dates.intersection(dates)
    return sorted(intersection_dates or [])


def convert_weights_to_shares(w: np.ndarray, portfolio_value: float, close_prices: np.ndarray) -> np.ndarray:
    """
    Converts target weights w_t into actual share counts using day-t closing prices and the current
    portfolio value (share_i = w_i * value / close_i). Assets with a missing/non-positive close price
    on day t get zero shares (a position cannot be established). This lets the system track an actual
    evolving portfolio rather than only a sequence of weight vectors.
    """
    w_arr = np.asarray(w, dtype=float)
    close_arr = np.asarray(close_prices, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(np.isfinite(close_arr) & (close_arr > 0), w_arr * portfolio_value / close_arr, 0.0)
    return shares


class DailyOrchestrator:
    """
    Executes the deterministic daily loop for warmup and test windows.
    """
    def __init__(self, model_id: str = "openai/gpt-4o-mini", risk_aversion_lambda: float = 1.0):
        self.model_id = model_id
        self.risk_aversion_lambda = risk_aversion_lambda
        self.universe = build_full_universe()
        self.assets = self.universe["full_universe"]
        self.stores = {a: AssetMemoryStore(a) for a in self.assets}
        self.feedback_tracker = FeedbackTracker(lookback_days=5)

        # Step 1 factor-estimation universe: the mechanically-selected ~300-name cross-section
        # used for the daily Barra factor-return regression — separate from the traded book.
        self.estimation_universe = build_estimation_universe(FACTOR_ESTIMATION_UNIVERSE_SIZE)

        # Step 9 company relationship graph.
        self._relationship_candidates = build_candidate_tickers(
            self.universe["equities"], FACTOR_ESTIMATION_UNIVERSE
        )
        self.relationship_graph = CompanyRelationshipGraph(self._relationship_candidates)
        self._processed_filings = set()

    def _process_newly_filed_10ks(self, decision_date: str):
        """
        Step 9 extraction: for each traded equity, once per newly-filed 10-K, run one LLM call on the
        filing's Item 1 / Item 1A text (or the first N chars of the body) plus the fixed closed
        candidate list, then store the validated (source, target, relationship_type, filed_date) edges.
        The point-in-time filter (filed < decision day) ensures a filing only becomes usable the next
        decision day.
        """
        for equity in self.universe["equities"]:
            try:
                filings = get_10k_filing_documents(equity)
            except Exception as e:
                logger.warning("Could not list 10-K filings for %s: %s", equity, e)
                filings = []

            for filing in filings:
                filed_date = filing.get("filed")
                if not filed_date or filed_date > decision_date:
                    continue
                key = (equity, filed_date)
                if key in self._processed_filings:
                    continue
                self._processed_filings.add(key)

                try:
                    filing_text = get_filing_text(filing)
                except Exception as e:
                    logger.warning("Could not fetch 10-K text for %s on %s: %s", equity, filed_date, e)
                    filing_text = ""

                relationships = extract_relationships_from_filing(
                    source_ticker=equity,
                    filed_date=filed_date,
                    filing_text=filing_text,
                    candidate_tickers=self._relationship_candidates,
                    model_id=self.model_id
                )
                self.relationship_graph.register_extraction_result(equity, filed_date, relationships)

    def _run_view_formation_phase(self, start_date: str, end_date: str) -> dict:
        """
        Steps 1-4 (data, memory decay/migration + Step 4 feedback, Step 9 relationship extraction,
        Step 3 view formation), plus Step 7 reflection and the Step 8.6 memory write -- every part of
        the daily loop that does NOT depend on risk_aversion_lambda. Runs once over
        [start_date, end_date] and returns a cache Step 5 (portfolio optimization) can be re-run
        against for as many lambda candidates as needed, without re-invoking any of this.

        This is the only place `self.stores`, `self.feedback_tracker`, and `self.relationship_graph`
        are mutated for a given (orchestrator, window) pair: callers evaluating multiple lambda
        candidates over the same window must call this once and reuse the returned cache via
        `_run_portfolio_phase`, never call this (or `run_daily_loop`) once per candidate -- doing so
        would decay/migrate/reflect/write memory multiple times over the same dates and let each
        later candidate's view formation retrieve memories the earlier candidates' passes already
        wrote, silently changing the (direction, conviction) sequence between candidates.
        """
        # Fetch price history from 2015-01-01 to end_date to ensure 252-day trailing window is populated
        price_pivot = get_pivot_close_prices(self.assets, start_date="2015-01-01", end_date=end_date)
        # Align column order with self.assets so that positional operations (w_t, exposures, .values) line up.
        price_pivot = price_pivot.reindex(columns=self.assets)
        reference_assets = list(self.universe.get("equities", [])) + list(self.universe.get("etfs", []))
        all_dates = derive_decision_calendar(price_pivot, self.assets, reference_assets)
        price_pivot = price_pivot.loc[all_dates].ffill()

        # Target decision dates in window [start_date, end_date]
        target_dates = [d for d in all_dates if start_date <= d <= end_date]
        if not target_dates:
            return {"target_dates": [], "price_pivot": price_pivot, "all_dates": all_dates,
                    "style_stats": {}, "barra_factor_map": {}, "day_cache": {}}

        # Compute Barra factor model for the locked ~300-name estimation universe (a separate,
        # mechanically-selected cross-section, not the traded book): real factor-return series +
        # the daily estimation-universe cross-sectional style stats used to z-score traded-equity
        # exposures.
        estimation_prices = get_pivot_close_prices(
            self.estimation_universe, start_date="2015-01-01", end_date=end_date
        )
        estimation_prices = estimation_prices.reindex(columns=self.estimation_universe)
        barra_factors, style_stats = compute_barra_model(estimation_prices, self.estimation_universe)
        barra_factor_map = {}
        if not barra_factors.empty:
            barra_factor_map = {row["Date"]: row.to_dict() for _, row in barra_factors.iterrows()}

        # Shared per-run/per-day state so each Step 8 sub-step is independently callable.
        self._price_pivot = price_pivot
        self._all_dates = all_dates
        self._style_stats = style_stats
        self._barra_factor_map = barra_factor_map

        day_cache = {}
        for t_idx, d in enumerate(target_dates):
            self._t_idx = t_idx
            self._full_idx = all_dates.index(d)
            self._prev_d = all_dates[self._full_idx - 1] if self._full_idx > 0 else d

            # Step 8.1: memory decay & migration (before that day's feedback), then Step 4 feedback.
            self._decay_and_migrate_memory(d)
            # Step 9: extract company relationships from newly-filed 10-Ks (edges stored with their
            # filing `filed` date become visible starting the NEXT decision day).
            self._process_newly_filed_10ks(d)
            # Step 8.2: retrieve memories & run view formation (Step 3) using info up to (t-1) close.
            views = self._form_views(d)
            # Step 8.5: reflection generation every 5th trading day (Step 7). Depends only on realized
            # price action, never on a day's portfolio weights, so it belongs in this lambda-independent
            # phase, not repeated inside the per-lambda Step 5 phase below.
            self._maybe_run_reflection(d)
            # Step 8.6: write short-layer memory from day t realized price action. Same reasoning.
            self._write_memory(d)

            day_cache[d] = {"t_idx": t_idx, "full_idx": self._full_idx, "prev_d": self._prev_d, "views": views}

        return {
            "target_dates": target_dates,
            "price_pivot": price_pivot,
            "all_dates": all_dates,
            "style_stats": style_stats,
            "barra_factor_map": barra_factor_map,
            "day_cache": day_cache,
        }

    def _run_portfolio_phase(self, cache: dict, risk_aversion_lambda: float) -> pd.DataFrame:
        """
        Step 5 (portfolio optimization) + Step 6 (share/value bookkeeping) + Step 8.4
        (breadth/realized-exposure/peer-concentration reporting) -- the only parts of the daily loop
        that depend on risk_aversion_lambda. Re-run once per candidate lambda against the
        `_run_view_formation_phase` cache; does not touch self.stores, self.feedback_tracker, or
        self.relationship_graph. portfolio_value is reset to 1.0 at the start of every call since it
        evolves path-dependently on the weights chosen under this specific lambda, not shared
        across candidates.
        """
        self.risk_aversion_lambda = risk_aversion_lambda
        self.portfolio_value = 1.0
        self._price_pivot = cache["price_pivot"]
        self._all_dates = cache["all_dates"]
        self._style_stats = cache["style_stats"]

        records = []
        for d in cache["target_dates"]:
            day_info = cache["day_cache"][d]
            self._t_idx = day_info["t_idx"]
            self._full_idx = day_info["full_idx"]
            self._prev_d = day_info["prev_d"]
            views = day_info["views"]

            # Step 8.3: Step 5 portfolio construction + Step 6 share/value bookkeeping.
            w_t, p_ret_today = self._construct_portfolio(d, views)
            # Step 8.4: Step 6 breadth, realized factor exposure, and Step 9 peer concentration.
            breadth, realized_exp, peer_conc = self._report_breadth_exposure(d, w_t)

            records.append({
                "Date": d,
                "Portfolio_Return": p_ret_today,
                "Breadth": breadth,
                "Gross_Exposure": float(np.sum(np.abs(w_t))),
                "Factor_Exposures": realized_exp,
                "Peer_Concentration": peer_conc,
                "Barra_Factor_Returns": cache["barra_factor_map"].get(d, {}),
            })

        return pd.DataFrame(records)

    def run_daily_loop(self, start_date: str, end_date: str, is_warmup: bool = True) -> pd.DataFrame:
        """
        Runs chronological daily loop from start_date to end_date, at the orchestrator's current
        risk_aversion_lambda.

        Composed from `_run_view_formation_phase` (Steps 1-4, 9, and the lambda-independent Step 7/8.6
        memory writes) followed by `_run_portfolio_phase` (Step 5 optimization + Step 6 bookkeeping +
        Step 8.4 reporting) -- the same two phases `run_lambda_grid_search` composes, just invoked
        back-to-back here for a single lambda instead of the phase-1 cache being reused across several
        phase-2 calls.
        """
        cache = self._run_view_formation_phase(start_date, end_date)
        if not cache["target_dates"]:
            return pd.DataFrame()
        return self._run_portfolio_phase(cache, self.risk_aversion_lambda)

    def _decay_and_migrate_memory(self, day: str) -> None:
        """
        Step 8.1: decay & migrate memory for every asset, then apply due feedback (Step 4) for
        decisions 5 days prior. Migration runs before that day's feedback (locked ordering), and
        Step 2's cleanup rule (importance < 5 OR recency < 0.05) removes qualifying memories each
        day so stores do not grow monotonically across a multi-year warmup.
        """
        for a in self.assets:
            self.stores[a].step_daily_decay()
            run_layer_migration(self.stores[a])
            self.stores[a].run_cleanup()

        self.feedback_tracker.process_feedback_for_day(day, self.stores, self._price_pivot)

    def _form_views(self, day: str) -> list[dict]:
        """
        Step 8.2: retrieve memories & run view formation (Step 3) using info up to (t-1) close.
        Also records the day's decision in the feedback tracker (Step 4 write side) so feedback can
        be applied L days later. Returns the list of per-asset views.
        """
        price_pivot = self._price_pivot
        prev_d = self._prev_d

        views = []
        for a in self.assets:
            # 20-day price history through prev_d
            sub_prices = price_pivot.loc[:prev_d, a].tail(20)
            price_hist_20d = []
            for p_date, p_val in sub_prices.items():
                price_hist_20d.append({"Date": p_date, "Close": float(p_val)})

            # Fundamentals through prev_d
            prev_price = float(price_pivot.loc[prev_d, a]) if prev_d in price_pivot.index else None
            fund = get_point_in_time_fundamentals(a, decision_date=prev_d, closing_price=prev_price)

            # Retrieve memories
            query_str = f"{a} price action and fundamentals analysis"
            retrieved = self.stores[a].retrieve_top_k(query_str, k_per_layer=5)
            retrieved_ids = [m["memory_id"] for m in retrieved]

            # Step 9: point-in-time-filtered outgoing relationship summary for this asset
            relationships_summary = self.relationship_graph.summarize_for_view(a, day)

            # Call view formation agent
            v = call_view_formation_agent(
                asset_ticker=a,
                decision_date=day,
                price_history_20d=price_hist_20d,
                fundamentals=fund,
                retrieved_memories=retrieved,
                relationships_summary=relationships_summary,
                model_id=self.model_id
            )
            views.append(v)

            # Record the day's decision in the feedback tracker (written now, applied L days later).
            self.feedback_tracker.record_decision(
                date_str=day,
                asset=a,
                memory_ids=retrieved_ids,
                direction=v["direction"],
                close_price=float(price_pivot.loc[day, a])
            )

        return views

    def _construct_portfolio(self, day: str, views: list[dict]) -> tuple[np.ndarray, float]:
        """
        Step 8.3: Step 5 portfolio construction from the trailing 252-day window, then Step 6
        bookkeeping converting target weights into share counts and evolving the tracked portfolio
        value (close(t) -> close(t+1)). Any uninvested budget is held as cash and carried forward.
        Returns (w_t, p_ret_today) and stores the day's Sigma_shrunk / factor_exposures for reporting.
        """
        price_pivot = self._price_pivot
        full_idx = self._full_idx

        if full_idx >= 253:
            window_prices = price_pivot.iloc[full_idx - 253:full_idx].ffill().bfill()
            window_returns = window_prices.pct_change().dropna().values
            if window_returns.shape[0] >= 2 and window_returns.shape[1] > 0:
                mean_returns = np.mean(window_returns, axis=0)
                Sigma_shrunk, _ = compute_ledoit_wolf_constant_correlation(window_returns)
            else:
                mean_returns = np.zeros(len(self.assets))
                Sigma_shrunk = np.eye(len(self.assets))

            # Step 5: real 14 Barra factor exposures (N x 14) available as of the prior close.
            factor_exposures = compute_asset_factor_exposures(
                price_pivot, self.assets, self.assets, self._prev_d, self._style_stats
            )
            if factor_exposures is None:
                logger.warning("No estimation-universe style stats for %s; using neutral zero exposures.", day)
                factor_exposures = np.zeros((len(self.assets), 14))

            w_t = solve_portfolio_optimization(
                asset_names=self.assets,
                mean_returns=mean_returns,
                Sigma_shrunk=Sigma_shrunk,
                views=views,
                factor_exposures=factor_exposures,
                risk_aversion_lambda=self.risk_aversion_lambda
            )
        else:
            # Less than 252 days of history -> 100% cash fallback
            w_t = np.zeros(len(self.assets))
            Sigma_shrunk = np.eye(len(self.assets))
            factor_exposures = np.zeros((len(self.assets), 14))

        self._Sigma_shrunk = Sigma_shrunk
        self._factor_exposures = factor_exposures

        # Step 6 bookkeeping: convert target weights into share counts at day-t close and track the
        # actual evolving portfolio value (close(t) -> close(t+1)).
        close_today = price_pivot.loc[day].values
        value_before = self.portfolio_value
        shares_held = convert_weights_to_shares(w_t, value_before, close_today)
        valid_today = np.isfinite(close_today) & (close_today > 0)
        invested_at_close_today = float(np.sum(np.where(valid_today, shares_held * close_today, 0.0)))
        cash = value_before - invested_at_close_today
        if full_idx + 1 < len(self._all_dates):
            next_d = self._all_dates[full_idx + 1]
            close_next = price_pivot.loc[next_d].values
            close_next_safe = np.where(np.isfinite(close_next), close_next, close_today)
            # Only count assets with a known price today or carried forward (avoids 0 * NaN).
            valid_next = np.isfinite(close_next_safe)
            invested_next = float(np.sum(np.where(valid_next, shares_held * close_next_safe, 0.0)))
            self.portfolio_value = invested_next + cash
            p_ret_today = self.portfolio_value / value_before - 1.0 if value_before > 0 else 0.0
        else:
            p_ret_today = 0.0

        return w_t, p_ret_today

    def _report_breadth_exposure(self, day: str, w: np.ndarray) -> tuple:
        """
        Step 8.4: compute Step 6 breadth and realized factor exposure, plus Step 9 peer
        concentration, from the day's weights and the already-computed Sigma/factor exposures.
        """
        breadth = compute_portfolio_breadth(w, self._Sigma_shrunk)
        realized_exp = compute_realized_factor_exposures(w, self._factor_exposures)
        visible_edges = self.relationship_graph.visible_edges(day)
        peer_conc = compute_peer_concentration_exposure(
            w, self.assets, self.universe["equities"], visible_edges
        )
        return breadth, realized_exp, peer_conc

    def _maybe_run_reflection(self, day: str) -> None:
        """
        Step 8.5: every 5th trading day, run reflection generation (Step 7) over the trailing
        5-day price move for every asset.
        """
        price_pivot = self._price_pivot
        full_idx = self._full_idx

        if self._t_idx > 0 and self._t_idx % REFLECTION_INTERVAL_DAYS == 0:
            for a in self.assets:
                p_5d_ago = price_pivot.iloc[full_idx - 5][a] if full_idx >= 5 else price_pivot.iloc[0][a]
                p_now = price_pivot.loc[day, a]
                if pd.notna(p_5d_ago) and pd.notna(p_now) and float(p_5d_ago) > 0:
                    five_day_ret = float((p_now - p_5d_ago) / p_5d_ago)
                else:
                    five_day_ret = 0.0
                process_reflection_cycle(self.stores[a], date_str=day, five_day_return=five_day_ret)

    def _write_memory(self, day: str) -> None:
        """
        Step 8.6: write short-layer memory from day t realized price action.
        """
        price_pivot = self._price_pivot
        full_idx = self._full_idx
        prev_d = self._prev_d

        for a in self.assets:
            p_now = price_pivot.loc[day, a]
            p_prev = price_pivot.loc[prev_d, a] if full_idx > 0 else None
            if pd.notna(p_now) and pd.notna(p_prev) and float(p_prev) > 0:
                ret_t = float((p_now - p_prev) / p_prev)
            else:
                ret_t = 0.0
            mem_text = f"Day {day} price closed at {p_now:.2f} ({ret_t:+.2%})."
            self.stores[a].add_memory(mem_text, date_written=day, layer="short")

def run_lambda_grid_search(orchestrator: DailyOrchestrator, start_date: str = WARMUP_START_DATE, end_date: str = WARMUP_END_DATE) -> float:
    """
    Grid searches lambda in {0.5, 1.0, 2.0, 5.0, 10.0} on Warmup Sharpe ratio.

    Steps 1-4 (data, memory decay/migration + Step 4 feedback, Step 9 relationship extraction, Step 3
    view formation) do not depend on lambda at all, so they run exactly once over the warmup window
    (`_run_view_formation_phase`), caching the resulting daily (direction, conviction) view sequence
    per asset; only Step 5's portfolio optimization is then re-run, once per lambda candidate
    (`_run_portfolio_phase`), against that single cached sequence. This never re-invokes memory
    decay/migration/reflection/write or feedback recording per candidate, so `orchestrator.stores` is
    mutated exactly once regardless of how many lambda values are evaluated.

    Returns optimal lambda.
    """
    best_lambda = 1.0
    best_sharpe = -999.0

    from src.evaluation.metrics import calculate_annualized_sharpe

    cache = orchestrator._run_view_formation_phase(start_date, end_date)

    for lam in LAMBDA_GRID:
        df_warmup = orchestrator._run_portfolio_phase(cache, lam)
        if not df_warmup.empty and "Portfolio_Return" in df_warmup.columns:
            sr = calculate_annualized_sharpe(df_warmup["Portfolio_Return"])
            if sr > best_sharpe:
                best_sharpe = sr
                best_lambda = lam

    return best_lambda
