"""Live GA training visualization for the genetic trading algo (ETH default L/S).

Reproduces the configuration that came out of the systematic review as the only
robust edge: ETH/USDT, default L/S (no --long-only), rigorous walk-forward
(history from 2017-08-17, step_bars=2000). Picks ONE fold (the latest by
default) and runs the K=5 GA seeds in *interleaved* fashion so you can watch
them race in real time.

Layout (3 rows, dark theme, 20×11):
  - Top         : THE RACE. Train fitness per generation, top + mean of pop.
  - Mid (split) : CONSENSUS HEATMAP (left)  +  POPULATION SWARM (right).
  - Bottom      : status banner + final test-set reveal.

After 40 generations: top-3 seeds by val Sharpe stay bright, others fade out,
ensemble's test-set equity curve appears next to the race plot.

Run:
    uv run python -m strategies.genetic_trading.viz_train
    uv run python -m strategies.genetic_trading.viz_train --fold 10
    uv run python -m strategies.genetic_trading.viz_train --frame-pause 0.2
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategies.genetic_trading.final_gen1 import (
    GA,
    GAConfig,
    _aggregate_folds,
    _load_data,
    apply_hysteresis,
    backtest,
    evaluate,
    fold_sharpes,
    sharpe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="ETH/USDT")
    parser.add_argument("--start-date", default="2017-08-17 00:00:00")
    parser.add_argument("--step-bars", type=int, default=2000)
    parser.add_argument("--fold", type=int, default=-1,
                        help="Fold index (-1 = last available, the most recent).")
    parser.add_argument("--frame-pause", type=float, default=0.05,
                        help="Extra plt.pause() per generation. Increase to slow down.")
    parser.add_argument("--save-final", default=None,
                        help="If set, save the final figure to this PNG path.")
    args = parser.parse_args()

    cfg = GAConfig()  # default L/S, same as the winning ETH run
    K_SEEDS = 5
    TOP_N = 3
    FEE = 0.0006
    TRAIN_BARS, VAL_BARS, TEST_BARS = 26000, 4000, 4000

    # ----- Load data and slice the chosen fold -----
    print(f"Loading {args.pair} ...")
    data = _load_data(args.pair, "1h", "binance", args.start_date, "2050")
    X_all, prices_all, idx_all, n = (
        data["X_all"], data["prices_all"], data["index"], data["n"],
    )
    fold_size = TRAIN_BARS + VAL_BARS + TEST_BARS
    starts = list(range(0, n - fold_size + 1, args.step_bars))
    if not starts:
        raise SystemExit(f"Not enough data: {n} bars vs needed {fold_size}.")
    fold_idx = args.fold if args.fold >= 0 else len(starts) - 1
    fold_idx = max(0, min(fold_idx, len(starts) - 1))
    start = starts[fold_idx]
    tr_end = start + TRAIN_BARS
    va_end = tr_end + VAL_BARS
    te_end = va_end + TEST_BARS
    X_tr, p_tr = X_all[start:tr_end], prices_all[start:tr_end]
    X_va, p_va = X_all[tr_end:va_end], prices_all[tr_end:va_end]
    X_te, p_te = X_all[va_end:te_end], prices_all[va_end:te_end]
    n_features = X_tr.shape[1]
    print(f"Fold {fold_idx}/{len(starts)-1}: train {idx_all[start].date()} -> "
          f"{idx_all[tr_end].date()}, val -> {idx_all[va_end].date()}, "
          f"test -> {idx_all[te_end-1].date()}")

    # ----- Initialize K independent GA populations -----
    seeds = list(range(K_SEEDS))
    gas = [GA(n_features, replace(cfg, seed=s)) for s in seeds]
    best = [{"val": -np.inf, "w": None, "gen": -1} for _ in seeds]
    # Per-gen train fitness (what the GA actually optimizes — climbs with evolution).
    train_top_history: list[list[float]] = [[] for _ in seeds]   # max fitness in pop
    train_mean_history: list[list[float]] = [[] for _ in seeds]  # mean fitness in pop

    # ----- Matplotlib setup -----
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 11), facecolor="#0d1117", constrained_layout=False)
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[3.0, 2.8, 0.4],
        width_ratios=[1.05, 1.0],
        hspace=0.55, wspace=0.18,
        left=0.05, right=0.97, top=0.94, bottom=0.06,
    )

    # The race  (full width)
    ax_race = fig.add_subplot(gs[0, :])
    ax_race.set_title(
        f"THE RACE — train fitness per generation   |   {args.pair}, fold {fold_idx}",
        fontsize=18, fontweight="bold", color="white", pad=12,
    )
    ax_race.set_xlabel("Generation", color="white", fontsize=13)
    ax_race.set_ylabel("Train fitness", color="white", fontsize=13)
    ax_race.set_xlim(0, cfg.n_generations + 0.5)
    ax_race.set_ylim(-2.0, 2.5)  # autoscaled later
    ax_race.tick_params(labelsize=11)
    ax_race.grid(alpha=0.2)
    ax_race.axhline(0, color="#888", lw=0.8, ls="--", alpha=0.6)

    seed_colors = ["#ff4d6d", "#4cc9f0", "#ffd166", "#06d6a0", "#c77dff"]
    race_lines = []     # bold: top fitness in current gen (the champion)
    mean_lines = []     # thin dashed: mean fitness of population (the pack)
    for i, c in enumerate(seed_colors):
        line, = ax_race.plot([], [], lw=3.5, color=c, label=f"seed {i}",
                             marker="o", markersize=6, markerfacecolor=c,
                             markeredgecolor="white", markeredgewidth=0.6)
        race_lines.append(line)
        mean_line, = ax_race.plot([], [], lw=1.4, color=c, ls="--", alpha=0.55)
        mean_lines.append(mean_line)
    ax_race.legend(loc="lower right", fontsize=12, framealpha=0.6,
                   facecolor="#1a1a1a", edgecolor="#444", ncol=5,
                   columnspacing=1.0, handletextpad=0.4)

    # === CONSENSUS HEATMAP : rows = seeds, cols = features (left of mid row) ===
    # Each cell shows the *mean weight* of that seed's population for that
    # feature. Watch rows settle into stable color patterns and similarities
    # across rows = cross-seed agreement on the signal.
    ax_heatmap = fig.add_subplot(gs[1, 0])
    mean_weights = np.zeros((K_SEEDS, n_features), dtype=np.float64)
    for i in range(K_SEEDS):
        mean_weights[i] = gas[i].population.mean(axis=0)
    heatmap_im = ax_heatmap.imshow(
        mean_weights, aspect="auto", cmap="RdBu_r",
        vmin=-0.6, vmax=0.6, interpolation="nearest",
    )
    feat_labels = data["feat_cols"]
    ax_heatmap.set_xticks(np.arange(n_features))
    ax_heatmap.set_xticklabels(feat_labels, rotation=35, ha="right",
                               fontsize=11, color="white")
    ax_heatmap.set_yticks(np.arange(K_SEEDS))
    ax_heatmap.set_yticklabels([f"seed {i}" for i in range(K_SEEDS)],
                               fontsize=13, fontweight="bold")
    for tick_lbl, c in zip(ax_heatmap.get_yticklabels(), seed_colors):
        tick_lbl.set_color(c)
    ax_heatmap.set_title(
        "CONSENSUS WEIGHTS  (red = long, blue = short)",
        fontsize=14, color="white", pad=10,
    )
    cbar = fig.colorbar(heatmap_im, ax=ax_heatmap, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelcolor="white", labelsize=10)
    cbar.outline.set_edgecolor("#444")

    # === POPULATION SWARM : 80 individuals × 5 seeds in 2D PCA (right of mid row) ===
    # Watch each colored cloud migrate and shrink toward an attractor as the
    # GA evolves. Star = current top-fitness individual of each seed.
    ax_swarm = fig.add_subplot(gs[1, 1])
    ax_swarm.set_title(
        "POPULATION SWARM  (PCA 2D — stars = current top of each seed)",
        fontsize=14, color="white", pad=10,
    )
    ax_swarm.set_xlabel("PC1", color="#aaa", fontsize=11)
    ax_swarm.set_ylabel("PC2", color="#aaa", fontsize=11)
    ax_swarm.tick_params(colors="#888", labelsize=10)
    ax_swarm.grid(alpha=0.15)
    ax_swarm.set_facecolor("#06080d")

    # Fit PCA basis on the concatenated initial populations (5 * 80 = 400 vectors).
    all_initial = np.concatenate([ga.population for ga in gas], axis=0)
    pca_mean = all_initial.mean(axis=0)
    centered = all_initial - pca_mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = Vt[:2]                                  # shape (2, n_features)

    def project(W: np.ndarray) -> np.ndarray:
        return (W - pca_mean) @ pca_components.T             # (n, 2)

    swarm_scatters = []
    swarm_best_markers = []
    for i, c in enumerate(seed_colors):
        proj = project(gas[i].population)
        scat = ax_swarm.scatter(
            proj[:, 0], proj[:, 1],
            s=32, c=c, alpha=0.5, edgecolors="none",
            label=f"seed {i}",
        )
        best_marker, = ax_swarm.plot(
            [], [], marker="*", linestyle="",
            color=c, markersize=28,
            markeredgecolor="white", markeredgewidth=1.6,
        )
        swarm_scatters.append(scat)
        swarm_best_markers.append(best_marker)

    # Helper to auto-rescale the swarm axes around all current swarms.
    def rescale_swarm() -> None:
        all_pcs = np.concatenate([project(ga.population) for ga in gas], axis=0)
        xs, ys = all_pcs[:, 0], all_pcs[:, 1]
        xc, yc = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        xw = max(xs.max() - xs.min(), 0.4)
        yw = max(ys.max() - ys.min(), 0.4)
        pad = 0.18
        ax_swarm.set_xlim(xc - xw / 2 - pad, xc + xw / 2 + pad)
        ax_swarm.set_ylim(yc - yw / 2 - pad, yc + yw / 2 + pad)

    rescale_swarm()

    # Status banner (full width, bottom)
    ax_status = fig.add_subplot(gs[2, :])
    ax_status.axis("off")
    status_text = ax_status.text(
        0.5, 0.65, f"Generation 0 / {cfg.n_generations}",
        ha="center", va="center", fontsize=24, fontweight="bold",
        color="#39ff14", transform=ax_status.transAxes,
    )
    sub_text = ax_status.text(
        0.5, 0.10, "training in progress...",
        ha="center", va="center", fontsize=12, color="#bbb",
        transform=ax_status.transAxes, style="italic",
    )

    plt.show(block=False)
    plt.pause(0.5)

    # ----- Interleaved generations: gen 0 of all seeds, then gen 1 of all, ... -----
    for gen in range(cfg.n_generations):
        for i, ga in enumerate(gas):
            scfg = ga.cfg
            fits = np.empty(scfg.pop_size)
            for j, w in enumerate(ga.population):
                folds_s = fold_sharpes(
                    w, X_tr, p_tr, FEE, scfg.n_folds,
                    scfg.hysteresis_width, scfg.long_only,
                )
                fits[j] = (_aggregate_folds(folds_s, scfg.fold_aggregator)
                           - scfg.l1_penalty * float(np.sum(np.abs(w))))
            top = int(np.argmax(fits))
            top_w = ga.population[top]
            val_s, _ = evaluate(top_w, X_va, p_va, FEE,
                                scfg.hysteresis_width, scfg.long_only)
            if val_s > best[i]["val"]:
                best[i] = {"val": float(val_s), "w": top_w.copy(), "gen": gen}

            # Track train fitness top + mean per gen (these climb monotonically with elitism).
            train_top_history[i].append(float(fits.max()))
            train_mean_history[i].append(float(fits.mean()))

            # Update visuals for this seed.
            xs = np.arange(1, len(train_top_history[i]) + 1)
            race_lines[i].set_data(xs, train_top_history[i])
            mean_lines[i].set_data(xs, train_mean_history[i])
            mean_weights[i] = ga.population.mean(axis=0)
            heatmap_im.set_data(mean_weights)

            # SWARM update: project the 80 individuals into PCA space.
            # The star marker tracks the current top-fitness individual.
            proj = project(ga.population)
            swarm_scatters[i].set_offsets(proj)
            top_proj = proj[top]
            swarm_best_markers[i].set_data([top_proj[0]], [top_proj[1]])

            # Step the GA only after the visual update (so the heatmap shows the
            # population that was *just* evaluated, not the next gen's draft).
            ga.step(fits)

        # Rescale the swarm axes once per generation (all 5 seeds together).
        rescale_swarm()

        # Auto-rescale Y on the union of top + mean fitness values seen
        all_vals = [v for h in train_top_history for v in h] + \
                   [v for h in train_mean_history for v in h]
        if all_vals:
            lo = min(all_vals) - 0.2
            hi = max(all_vals) + 0.3
            ax_race.set_ylim(lo, hi)

        status_text.set_text(f"Generation {gen+1} / {cfg.n_generations}")
        leader = max(range(K_SEEDS), key=lambda i: best[i]["val"])
        sub_text.set_text(
            f"leader: seed {leader} (val Sharpe {best[leader]['val']:+.2f})"
        )

        plt.pause(args.frame_pause)

    # ----- Final reveal: top-3 ensemble on test set -----
    ranked = sorted(range(K_SEEDS), key=lambda i: best[i]["val"], reverse=True)
    top_idx = ranked[:TOP_N]
    losers = ranked[TOP_N:]

    y_lbls = ax_heatmap.get_yticklabels()
    for i in losers:
        race_lines[i].set_alpha(0.15)
        race_lines[i].set_lw(1.2)
        mean_lines[i].set_alpha(0.10)
        # Strikethrough loser rows in the consolidated heatmap.
        y_lbls[i].set_color("#555")
        y_lbls[i].set_text(f"seed {i}  (out)")
        # Fade loser swarms.
        swarm_scatters[i].set_alpha(0.10)
        swarm_best_markers[i].set_alpha(0.15)
    for i in top_idx:
        race_lines[i].set_lw(5.0)
        mean_lines[i].set_lw(2.0)
        y_lbls[i].set_text(f"seed {i}  *TOP-3*")
        y_lbls[i].set_fontweight("bold")
        # Boost winner swarms: bigger, brighter dots and stars.
        swarm_scatters[i].set_alpha(0.75)
        swarm_scatters[i].set_sizes(np.full(swarm_scatters[i].get_offsets().shape[0], 35))
        swarm_best_markers[i].set_markersize(28)
        swarm_best_markers[i].set_markeredgewidth(2.0)

    sigs = []
    for i in top_idx:
        s = np.tanh(X_te @ best[i]["w"])
        if cfg.long_only:
            s = np.maximum(0.0, s)
        sigs.append(s)
    avg_sig = np.mean(sigs, axis=0)
    pos_ens = apply_hysteresis(avg_sig, width=cfg.hysteresis_width)
    pnl_ens = backtest(pos_ens, p_te, FEE)
    pnl_bh = backtest(np.ones_like(p_te), p_te, FEE)

    eq_ens = np.cumprod(1.0 + pnl_ens)
    eq_bh = np.cumprod(1.0 + pnl_bh)
    sh_ens = sharpe(pnl_ens)
    sh_bh = sharpe(pnl_bh)

    # Plot test equity curves on a twinx of the race panel, to the right of training.
    ax_test = ax_race.twinx()
    ax_test.set_ylabel("test equity (start = 1.0)", color="#39ff14", fontsize=12)
    ax_test.tick_params(axis="y", labelcolor="#39ff14")
    test_x = np.linspace(cfg.n_generations + 1, cfg.n_generations + 12, len(eq_ens))
    ax_test.plot(test_x, eq_ens, color="#39ff14", lw=4, label=f"ENSEMBLE  Sharpe {sh_ens:+.2f}")
    ax_test.plot(test_x, eq_bh, color="#ff9500", lw=2.2, alpha=0.85,
                 label=f"BUY & HOLD  Sharpe {sh_bh:+.2f}")
    ax_test.legend(loc="upper left", fontsize=11, framealpha=0.6,
                   facecolor="#1a1a1a", edgecolor="#444")

    ax_race.set_xlim(0, cfg.n_generations + 13)
    ax_race.axvline(cfg.n_generations + 0.5, color="white", lw=1.2, ls="--", alpha=0.5)
    ymax = ax_race.get_ylim()[1]
    ax_race.text(cfg.n_generations + 1.0, ymax * 0.92,
                 "TRAINING DONE  ->  TEST SET",
                 color="white", fontsize=12, fontweight="bold")

    status_text.set_text(
        f"TOP-{TOP_N} ENSEMBLE on TEST  ->  Sharpe {sh_ens:+.2f}  vs  B&H {sh_bh:+.2f}"
        f"   |   equity x{eq_ens[-1]:.2f}  vs  x{eq_bh[-1]:.2f}"
    )
    sub_text.set_text(
        "top-3 seeds: " + ", ".join(f"s{i}" for i in top_idx)
    )

    if args.save_final:
        out = Path(args.save_final)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
        print(f"Saved final figure to {out}")

    print(f"\nFINAL — Ensemble Sharpe {sh_ens:+.2f}, equity x{eq_ens[-1]:.2f}  "
          f"vs B&H Sharpe {sh_bh:+.2f}, equity x{eq_bh[-1]:.2f}")
    plt.show(block=True)


if __name__ == "__main__":
    main()
