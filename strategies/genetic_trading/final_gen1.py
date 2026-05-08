"""Genetic Trading — final algo: ensemble walk-forward on BTC/USDT 1h.

The configuration retained after systematic experimentation. What it does:

  1. Builds 16 normalized features per bar (returns, MA distances, RSI, ATR,
     volume z-score, channel positions, momentum) — all roughly in [-1, 1].

  2. Walks forward through the data with rolling windows
     (train 26000 bars / val 4000 / test 4000, step 4000) — 6 folds.

  3. For each fold, runs the GA 5 times with different seeds. Each GA run:
     - Evolves a 16-weight vector w over 40 generations of population 80.
     - Fitness = min Sharpe across 3 chronological sub-folds of train, minus
       L1 penalty on weights (forces robustness across train regimes).
     - Champion selected by best Sharpe on val set.

  4. Ensembles the top 3 of those 5 champions (ranked by val Sharpe):
        signal(t) = mean( tanh(X[t] · w_i) for i in top 3 )
     The continuous signal is then converted to discrete exposure via
     hysteresis: levels {-1.0, -0.8, …, 0, …, 0.8, 1.0}.

  5. Backtests the ensemble on the test period with 0.06% per-trade fee.

  6. Aggregates the 6 fold test Sharpes (mean / median / std).

Validation: real WF mean Sharpe ≈ +0.31 vs placebo (shuffled returns) WF
mean ≈ -1.25, gap +1.56 Sharpe (Mann-Whitney p < 0.05 on n=6 folds).

Run:
    python -m strategies.genetic_trading.final_gen1                  # real
    python -m strategies.genetic_trading.final_gen1 --placebo 0      # null check
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utilities.data_manager import ExchangeDataManager


# --------------------------------------------------------------------------- #
# Feature library — 16 indicators all roughly bounded in [-1, 1]
# --------------------------------------------------------------------------- #

def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std().replace(0, np.nan)
    return (s - m) / sd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    rs = avg_up / avg_dn
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """16 features: log-return z-scores, MA distances, RSI, ATR, volume z,
    channel position, momentum, Trix histograms. All squashed to ~[-1, 1]."""
    feats: dict[str, pd.Series] = {}

    # Returns z-score over daily / 3-day / weekly horizons. Dropped h=1, h=6
    # (redundant with mom_12 and ma_dist_5).
    for h in (24, 72, 168):
        r = np.log(df["close"] / df["close"].shift(h))
        feats[f"ret_z_{h}"] = _rolling_zscore(r, 500).clip(-3, 3) / 3

    # MA distance. Dropped p=1 (degenerate: log(close/close)=0 always).
    for p in (5, 20, 50):
        ma = df["close"].rolling(p).mean()
        feats[f"ma_dist_{p}"] = np.tanh(np.log(df["close"] / ma) * 10)

    for p in (14, 50):
        feats[f"rsi_{p}"] = (_rsi(df["close"], p) - 50) / 50

    atr = _atr(df, 14)
    feats["atr_rel"] = np.tanh(atr / df["close"] * 200 - 2)

    feats["vol_z"] = _rolling_zscore(df["volume"], 100).clip(-3, 3) / 3

    return pd.DataFrame(feats, index=df.index)


# --------------------------------------------------------------------------- #
# Signal → discrete exposure with hysteresis
# --------------------------------------------------------------------------- #

MAX_LEVEL = 5      # 5 * 0.2 = 1.0 full exposure
STEP = 0.2


def apply_hysteresis(signal: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Map signal in [-1, 1] to discrete exposure in 0.2 steps with hysteresis.

    At integer level k (exposure = k * 0.2), stay while signal is in the band
    [(k - width)·0.2, (k + width)·0.2]. width=1.0 = original spec (move up
    when signal > (k+1)·0.2). Larger widths = stickier positions (less churn).
    """
    n = signal.shape[0]
    pos = np.empty(n, dtype=np.float64)
    k = 0
    band = width * STEP
    for t in range(n):
        s = signal[t]
        while k < MAX_LEVEL and s > k * STEP + band:
            k += 1
        while k > -MAX_LEVEL and s < k * STEP - band:
            k -= 1
        pos[t] = k * STEP
    return pos


# --------------------------------------------------------------------------- #
# Backtest + Sharpe
# --------------------------------------------------------------------------- #

BARS_PER_YEAR = 24 * 365  # 1h timeframe


def backtest(positions: np.ndarray, prices: np.ndarray, fee: float = 0.0006) -> np.ndarray:
    """Per-bar net simple returns. positions[t] held from bar t to t+1.

    All transition costs are charged:
      - cost[t] = |positions[t] - positions[t-1]| * fee  for t in 0..n-1
        (with positions[-1] virtually 0 → cost[0] is the entry cost).
      - The cost paid at bar t is attributed to the same bar's PnL.
      - For the final bar (n-1) we also charge the close-out
        (|positions[n-1]| * fee), since liquidating the last position is real.
    """
    ret = np.diff(prices) / prices[:-1]                              # length n-1
    pnl_gross = positions[:-1] * ret
    prev = np.concatenate(([0.0], positions[:-1]))
    cost = np.abs(positions - prev) * fee                            # length n
    pnl = pnl_gross - cost[:-1]
    if len(pnl) > 0:
        # Charge cost[n-1] (last transition, paid but never realized further)
        # and the close-out (|pos[n-1]| * fee) on the last PnL bar.
        pnl = pnl.copy()
        pnl[-1] -= cost[-1] + abs(positions[-1]) * fee
    return pnl


def sharpe(pnl: np.ndarray) -> float:
    """Annualized Sharpe ratio (sample std, ddof=1)."""
    if len(pnl) < 2:
        return 0.0
    sd = pnl.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(pnl.mean() / sd * np.sqrt(BARS_PER_YEAR))


def evaluate(
    w: np.ndarray, X: np.ndarray, prices: np.ndarray,
    fee: float, hysteresis_width: float = 1.0, long_only: bool = False,
) -> tuple[float, np.ndarray]:
    signal = np.tanh(X @ w)
    if long_only:
        signal = np.maximum(0.0, signal)
    pos = apply_hysteresis(signal, width=hysteresis_width)
    pnl = backtest(pos, prices, fee)
    return sharpe(pnl), pnl


def fold_sharpes(
    w: np.ndarray, X: np.ndarray, prices: np.ndarray,
    fee: float, n_folds: int, hysteresis_width: float = 1.0, long_only: bool = False,
) -> np.ndarray:
    """Split (X, prices) into n_folds chronological chunks, return one Sharpe per fold.
    Each fold is a standalone backtest (positions reset to 0 at each fold start)."""
    if n_folds <= 1:
        s, _ = evaluate(w, X, prices, fee, hysteresis_width, long_only)
        return np.array([s])
    n = len(prices)
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    out = np.empty(n_folds, dtype=np.float64)
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        s, _ = evaluate(w, X[a:b], prices[a:b], fee, hysteresis_width, long_only)
        out[k] = s
    return out


def _aggregate_folds(sharpes: np.ndarray, mode: str) -> float:
    if mode == "min":
        return float(sharpes.min())
    if mode == "median":
        return float(np.median(sharpes))
    if mode == "mean":
        return float(sharpes.mean())
    raise ValueError(f"unknown fold_aggregator: {mode!r}")


# --------------------------------------------------------------------------- #
# Genetic algorithm
# --------------------------------------------------------------------------- #

@dataclass
class GAConfig:
    pop_size: int = 80
    n_generations: int = 40
    elite: int = 4
    tournament_k: int = 4
    mut_rate: float = 0.15
    mut_sigma: float = 0.25
    crossover_alpha_low: float = -0.25
    crossover_alpha_high: float = 1.25
    l1_penalty: float = 0.02
    seed: int = 42
    n_folds: int = 3                       # train sub-folds for fitness
    fold_aggregator: str = "min"           # aggregator across the n_folds sub-folds
    hysteresis_width: float = 1.0
    long_only: bool = False                # if True, signal is clipped to [0, 1] in training


class GA:
    def __init__(self, n_features: int, cfg: GAConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        scale = 1.0 / np.sqrt(n_features)
        self.population = self.rng.standard_normal((cfg.pop_size, n_features)) * scale

    def _tournament(self, fits: np.ndarray) -> np.ndarray:
        idx = self.rng.choice(len(fits), self.cfg.tournament_k, replace=False)
        return self.population[idx[np.argmax(fits[idx])]]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        a = self.rng.uniform(self.cfg.crossover_alpha_low, self.cfg.crossover_alpha_high)
        return a * p1 + (1 - a) * p2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        mask = self.rng.random(ind.shape) < self.cfg.mut_rate
        noise = self.rng.standard_normal(ind.shape) * self.cfg.mut_sigma
        return ind + mask * noise

    def step(self, fits: np.ndarray) -> None:
        order = np.argsort(-fits)
        new_pop = [self.population[i].copy() for i in order[: self.cfg.elite]]
        while len(new_pop) < self.cfg.pop_size:
            p1 = self._tournament(fits)
            p2 = self._tournament(fits)
            new_pop.append(self._mutate(self._crossover(p1, p2)))
        self.population = np.asarray(new_pop)


# --------------------------------------------------------------------------- #
# Data loading (with optional placebo via return shuffling)
# --------------------------------------------------------------------------- #

def shuffle_ohlcv(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Destroy temporal structure while preserving marginal distributions.

    Shuffles log-returns of close → rebuilds a new close series. Keeps the
    intra-bar shape (open/high/low ratios to close), so candles still look
    realistic, but cross-bar dynamics become random. Volume is shuffled
    independently. Used to build a null-hypothesis dataset for placebo tests.
    """
    rng = np.random.default_rng(seed)
    close = df["close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(close))
    shuffled = rng.permutation(log_ret)
    new_close = np.empty_like(close)
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.exp(np.cumsum(shuffled))
    ratio = new_close / close

    out = df.copy()
    out["close"] = new_close
    out["open"] = df["open"].to_numpy() * ratio
    out["high"] = df["high"].to_numpy() * ratio
    out["low"] = df["low"].to_numpy() * ratio
    out["volume"] = rng.permutation(df["volume"].to_numpy())
    return out


def _load_data(
    pair: str,
    interval: str,
    exchange: str,
    start_date: str,
    end_date: str,
    placebo_seed: int | None = None,
) -> dict:
    """Load OHLCV, optionally shuffle for placebo, build features, return arrays."""
    dm = ExchangeDataManager(exchange, "./database/exchanges")
    df = dm.load_data(pair, interval, start_date=start_date, end_date=end_date)
    df = df[["open", "high", "low", "close", "volume"]].copy()

    if placebo_seed is not None:
        df = shuffle_ohlcv(df, seed=placebo_seed)

    feats = build_features(df)
    full = pd.concat([df[["close"]], feats], axis=1).dropna()
    feat_cols = list(feats.columns)
    X_all = full[feat_cols].to_numpy(dtype=np.float64)
    prices_all = full["close"].to_numpy(dtype=np.float64)
    return {
        "X_all": X_all, "prices_all": prices_all,
        "feat_cols": feat_cols, "index": full.index, "n": len(full),
    }


# --------------------------------------------------------------------------- #
# GA search on a single fold (returns the champion's weights)
# --------------------------------------------------------------------------- #

def _ga_search(fold: dict, cfg: GAConfig, fee: float) -> dict:
    """Run a single GA on a single fold. Returns the champion (best val) weights
    and its train/val/test Sharpes."""
    X_tr, p_tr = fold["X_tr"], fold["p_tr"]
    X_va, p_va = fold["X_va"], fold["p_va"]
    X_te, p_te = fold["X_te"], fold["p_te"]

    def fitness_train(w: np.ndarray) -> float:
        folds = fold_sharpes(w, X_tr, p_tr, fee, cfg.n_folds, cfg.hysteresis_width, cfg.long_only)
        agg = _aggregate_folds(folds, cfg.fold_aggregator)
        return agg - cfg.l1_penalty * float(np.sum(np.abs(w)))

    def sharpe_val(w: np.ndarray) -> float:
        return evaluate(w, X_va, p_va, fee, cfg.hysteresis_width, cfg.long_only)[0]

    ga = GA(n_features=X_tr.shape[1], cfg=cfg)
    best_val, best_w, best_gen = -np.inf, None, -1

    for gen in range(cfg.n_generations):
        fits = np.array([fitness_train(w) for w in ga.population])
        top = int(np.argmax(fits))
        val_s = sharpe_val(ga.population[top])
        if val_s > best_val:
            best_val, best_w, best_gen = val_s, ga.population[top].copy(), gen
        ga.step(fits)

    assert best_w is not None
    tr_s, _ = evaluate(best_w, X_tr, p_tr, fee, cfg.hysteresis_width, cfg.long_only)
    va_s, _ = evaluate(best_w, X_va, p_va, fee, cfg.hysteresis_width, cfg.long_only)
    te_s, _ = evaluate(best_w, X_te, p_te, fee, cfg.hysteresis_width, cfg.long_only)
    return {
        "best_weights": best_w,
        "train_sharpe": tr_s, "val_sharpe": va_s, "test_sharpe": te_s,
        "best_gen": best_gen,
    }


# --------------------------------------------------------------------------- #
# Walk-forward ensemble (the headline routine)
# --------------------------------------------------------------------------- #

def _bh_sharpe(prices: np.ndarray, fee: float) -> float:
    """Sharpe of a constant long position (entry + exit cost only)."""
    pnl = backtest(np.ones_like(prices), prices, fee)
    return sharpe(pnl)


def run_walk_forward_ensemble(
    pair: str = "BTC/USDT",
    interval: str = "1h",
    exchange: str = "binance",
    start_date: str = "2020-01-01 00:00:00",
    end_date: str = "2050",
    train_bars: int = 26000,
    val_bars: int = 4000,
    test_bars: int = 4000,
    step_bars: int = 4000,
    K_seeds: int = 5,
    top_N: int = 3,
    base_cfg: GAConfig | None = None,
    fee: float = 0.0006,
    placebo_seed: int | None = None,
    collect_arrays: bool = False,
    seed_offset: int = 0,
):
    """For each rolling fold, run K_seeds GA searches, ensemble the top_N (by
    val Sharpe) by averaging their continuous signals, then apply hysteresis
    on the averaged signal and backtest on test. Returns a DataFrame with
    one row per fold."""
    base_cfg = base_cfg or GAConfig()
    if not 1 <= top_N <= K_seeds:
        raise ValueError(f"top_N={top_N} must be in [1, K_seeds={K_seeds}]")

    data = _load_data(pair, interval, exchange, start_date, end_date, placebo_seed=placebo_seed)
    X_all, prices_all, index_all, n = data["X_all"], data["prices_all"], data["index"], data["n"]

    fold_size = train_bars + val_bars + test_bars
    if n < fold_size:
        raise ValueError(f"Not enough data ({n} bars) for one fold ({fold_size} bars)")
    starts = list(range(0, n - fold_size + 1, step_bars))

    label = "PLACEBO" if placebo_seed is not None else "REAL"
    print(f"Walk-forward ensemble [{label}]: {pair}, {len(starts)} folds, "
          f"K={K_seeds} seeds/fold, top_N={top_N}")
    print(f"  train={train_bars} val={val_bars} test={test_bars} step={step_bars}")
    print(f"  cfg: pop={base_cfg.pop_size}, gens={base_cfg.n_generations}, "
          f"L1={base_cfg.l1_penalty}, n_folds={base_cfg.n_folds}, "
          f"fold_agg={base_cfg.fold_aggregator}, fee={fee}")
    print("=" * 90)

    rows = []
    arrays = [] if collect_arrays else None
    for fold_idx, start in enumerate(starts):
        tr_end = start + train_bars
        va_end = tr_end + val_bars
        te_end = va_end + test_bars
        fold = {
            "X_tr": X_all[start:tr_end], "p_tr": prices_all[start:tr_end],
            "X_va": X_all[tr_end:va_end], "p_va": prices_all[tr_end:va_end],
            "X_te": X_all[va_end:te_end], "p_te": prices_all[va_end:te_end],
        }

        # Run K_seeds independent GA searches. seed_offset shifts the seeds
        # used (default 0 -> seeds 0..K_seeds-1; offset=5 -> 5..5+K_seeds-1)
        # so we can rerun the whole WF with fresh randomness for multi-seed
        # validation of the headline number.
        seed_results = []
        for seed in range(K_seeds):
            seed_value = seed_offset + seed
            cfg = replace(base_cfg, seed=seed_value)
            res = _ga_search(fold, cfg, fee)
            seed_results.append({
                "seed": seed_value, "w": res["best_weights"],
                "val_sharpe": res["val_sharpe"], "test_sharpe": res["test_sharpe"],
            })

        # Ensemble: average continuous signals of top_N by val Sharpe.
        seed_results.sort(key=lambda r: r["val_sharpe"], reverse=True)
        top_seeds = seed_results[:top_N]

        def _seed_signal(r):
            s = np.tanh(fold["X_te"] @ r["w"])
            return np.maximum(0.0, s) if base_cfg.long_only else s

        avg_signal = np.mean([_seed_signal(r) for r in top_seeds], axis=0)
        pos_ens = apply_hysteresis(avg_signal, width=base_cfg.hysteresis_width)
        pnl_ens = backtest(pos_ens, fold["p_te"], fee)
        ensemble_test = sharpe(pnl_ens)

        # Benchmarks on the SAME test period for honest comparison.
        bh_test = _bh_sharpe(fold["p_te"], fee)
        long_only_signal = np.maximum(0.0, avg_signal)
        pos_long_only = apply_hysteresis(long_only_signal, width=base_cfg.hysteresis_width)
        pnl_long_only = backtest(pos_long_only, fold["p_te"], fee)
        long_only_test = sharpe(pnl_long_only)

        # Activity metrics on the ensemble strategy (diagnostic for "do nothing" bias).
        mean_abs_pos = float(np.mean(np.abs(pos_ens)))
        switches = int(np.sum(np.diff(pos_ens) != 0))
        switches_per_bar = switches / max(len(pos_ens) - 1, 1)

        single_best_test = top_seeds[0]["test_sharpe"]   # the val-best seed alone
        seeds_test = np.array([r["test_sharpe"] for r in seed_results])
        row = {
            "fold": fold_idx,
            "test_start": str(index_all[va_end])[:10],
            "test_end": str(index_all[te_end - 1])[:10],
            "ensemble_test": ensemble_test,
            "single_best_test": single_best_test,
            "buy_hold_test": bh_test,
            "long_only_test": long_only_test,
            "mean_abs_pos": mean_abs_pos,
            "switches_per_bar": switches_per_bar,
            "seed_median_test": float(np.median(seeds_test)),
            "seed_mean_test": float(np.mean(seeds_test)),
            "seed_std_test": float(np.std(seeds_test)),
        }
        print(
            f"fold {fold_idx:>2}: ens={ensemble_test:+.2f}  "
            f"long_only={long_only_test:+.2f}  bh={bh_test:+.2f}  "
            f"single={single_best_test:+.2f}  "
            f"|pos|={mean_abs_pos:.2f}  sw/bar={switches_per_bar:.4f}  "
            f"[{row['test_start']} -> {row['test_end']}]"
        )
        rows.append(row)

        if arrays is not None:
            arrays.append({
                "fold": fold_idx,
                "test_index": index_all[va_end:te_end],
                "train_start_date": index_all[start],
                "train_end_date": index_all[tr_end],
                "val_end_date": index_all[va_end],
                "test_end_date": index_all[te_end - 1],
                "prices": fold["p_te"].copy(),
                "pos_ens": pos_ens.copy(),
                "pnl_ens": pnl_ens.copy(),
                "pos_long_only": pos_long_only.copy(),
                "pnl_long_only": pnl_long_only.copy(),
                "avg_signal": avg_signal.copy(),
                # All 5 seeds' weights + diagnostics (the GA-found "parameters")
                "all_seeds": [{
                    "seed": r["seed"],
                    "weights": r["w"].copy(),
                    "val_sharpe": float(r["val_sharpe"]),
                    "test_sharpe": float(r["test_sharpe"]),
                } for r in seed_results],
                # Top-3 used in the ensemble (by val Sharpe).
                "top_seeds_idx": [seed_results.index(r) for r in top_seeds],
                "top_weights": [r["w"].copy() for r in top_seeds],
                "top_val_sharpes": [float(r["val_sharpe"]) for r in top_seeds],
            })

    df_out = pd.DataFrame(rows)
    print("-" * 90)

    def _summary(name: str, col: str) -> None:
        s = df_out[col]
        print(
            f"{name:<14} mean={s.mean():+.3f} std={s.std():.3f} "
            f"median={s.median():+.3f} min={s.min():+.3f} max={s.max():+.3f}"
        )

    _summary("ENSEMBLE", "ensemble_test")
    _summary("LONG-ONLY", "long_only_test")
    _summary("BUY & HOLD", "buy_hold_test")
    _summary("SINGLE-BEST", "single_best_test")
    _summary("SEED-MEDIAN", "seed_median_test")
    print(
        f"|pos| mean = {df_out['mean_abs_pos'].mean():.3f}  "
        f"switches/bar mean = {df_out['switches_per_bar'].mean():.4f}"
    )
    print(
        f"ENSEMBLE − BUY&HOLD: mean={df_out['ensemble_test'].mean() - df_out['buy_hold_test'].mean():+.3f}"
    )
    if arrays is not None:
        return df_out, arrays, data["feat_cols"]
    return df_out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Genetic Trading — final algo (walk-forward ensemble)."
    )
    parser.add_argument(
        "--placebo",
        type=int,
        default=None,
        metavar="SEED",
        help="Run on shuffled returns (null-hypothesis test) with given shuffle seed.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01 00:00:00",
        help="Earliest training data. Use '2017-08-17 00:00:00' for full history.",
    )
    parser.add_argument(
        "--step-bars",
        type=int,
        default=4000,
        help="Walk-forward step size in bars (smaller = more folds).",
    )
    parser.add_argument(
        "--rigorous",
        action="store_true",
        help="Shortcut: --start-date='2017-08-17 00:00:00' --step-bars=2000.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="BTC/USDT",
        help="Trading pair (e.g. 'BTC/USDT', 'ETH/USDT').",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Train and evaluate the GA with the constraint signal >= 0 (no shorts).",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        metavar="PATH",
        help="Pickle per-fold positions, PnL and prices to PATH for offline analysis.",
    )
    parser.add_argument(
        "--hysteresis-width",
        type=float,
        default=1.0,
        help="Hysteresis bandwidth (default 1.0). Lower = more aggressive (e.g. 0.5).",
    )
    parser.add_argument(
        "--l1-penalty",
        type=float,
        default=0.02,
        help="L1 penalty on weights in fitness (default 0.02). Lower = bigger weights.",
    )
    parser.add_argument(
        "--fold-aggregator",
        choices=["min", "median", "mean"],
        default="min",
        help="How to aggregate Sharpe across the 3 train sub-folds (default 'min').",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Shift the K GA seeds used per fold (default 0 -> seeds 0..K-1). "
             "Use different offsets to do multi-seed validation of the headline.",
    )
    args = parser.parse_args()

    if args.rigorous:
        args.start_date = "2017-08-17 00:00:00"
        args.step_bars = 2000

    base_cfg = GAConfig(
        long_only=args.long_only,
        hysteresis_width=args.hysteresis_width,
        l1_penalty=args.l1_penalty,
        fold_aggregator=args.fold_aggregator,
    )
    out = run_walk_forward_ensemble(
        pair=args.pair,
        start_date=args.start_date,
        step_bars=args.step_bars,
        base_cfg=base_cfg,
        placebo_seed=args.placebo,
        collect_arrays=args.save_results is not None,
        seed_offset=args.seed_offset,
    )
    if args.save_results is not None:
        import pickle
        df, arrays, feat_cols = out
        payload = {
            "df": df,
            "arrays": arrays,
            "feat_cols": feat_cols,
            "pair": args.pair,
            "long_only": args.long_only,
            "start_date": args.start_date,
            "step_bars": args.step_bars,
            "hysteresis_width": args.hysteresis_width,
            "l1_penalty": args.l1_penalty,
            "fold_aggregator": args.fold_aggregator,
            "seed_offset": args.seed_offset,
            "fee": 0.0006,
        }
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"\nSaved per-fold arrays + dataframe to {save_path}")
    else:
        df = out
    print()
    print(df.to_string(index=False))
