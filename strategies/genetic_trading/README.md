# Genetic Trading

Stratégie de trading sur BTC/ETH (1h) où les poids d'un signal linéaire sur 16 features techniques sont **appris par algorithme génétique**, avec validation walk-forward et ensemblage multi-seed.

## Fichiers

- `final_gen1.py` — moteur complet : features, GA, walk-forward, ensemble, backtest, CLI.
- `viz_train.py` — visualisation live (matplotlib) de l'entraînement GA sur un seul fold.

## Pipeline (résumé)

1. **16 features** normalisées dans ~[-1, 1] : returns z-score (24h/72h/168h), distance aux MA (5/20/50), RSI (14/50), ATR relatif, volume z-score, etc.
2. **Walk-forward** glissant : train 26000 bars / val 4000 / test 4000, step 4000 → ~6 folds.
3. Pour chaque fold, **5 GAs** indépendants (seeds différents). Chaque GA = pop 80, 40 générations, fitness = `min(Sharpe sur 3 sous-folds train) − L1·||w||₁`.
4. Le **champion** de chaque seed est sélectionné sur le Sharpe de validation.
5. **Ensemble** : moyenne des signaux continus du top-3 (par Sharpe val), puis hystérésis → exposition discrète par paliers de 0.2 entre -1 et +1.
6. Backtest test set, fee 0.06% par trade.

## Utilisation

Toujours lancer **depuis la racine du repo** (les imports utilisent `strategies.genetic_trading...`).

### `final_gen1.py` — run principal

```bash
# Run réel (BTC/USDT par défaut, depuis 2020)
python -m strategies.genetic_trading.final_gen1

# Test placebo (returns shufflés, null hypothesis)
python -m strategies.genetic_trading.final_gen1 --placebo 0

# Mode rigoureux : historique complet (2017-08-17) + step_bars=2000 (plus de folds)
python -m strategies.genetic_trading.final_gen1 --rigorous

# Sur ETH, long-only, sauvegarde des arrays par fold pour analyse offline
python -m strategies.genetic_trading.final_gen1 \
    --pair ETH/USDT --long-only \
    --save-results results/eth_long_only.pkl
```

**Flags utiles :**

| Flag | Effet |
|---|---|
| `--pair` | Paire (`BTC/USDT`, `ETH/USDT`, …). |
| `--start-date` | Début des données (défaut `2020-01-01`). |
| `--step-bars` | Pas du walk-forward (plus petit = plus de folds). |
| `--rigorous` | Raccourci : `--start-date=2017-08-17 --step-bars=2000`. |
| `--placebo SEED` | Shuffle les log-returns (test de nullité). |
| `--long-only` | Contraint le signal ≥ 0 (pas de short). |
| `--hysteresis-width` | Largeur de l'hystérésis (défaut 1.0, plus bas = plus agressif). |
| `--l1-penalty` | Pénalité L1 sur les poids (défaut 0.02). |
| `--fold-aggregator` | `min` / `median` / `mean` sur les 3 sous-folds train. |
| `--seed-offset` | Décale les K seeds GA (multi-seed validation). |
| `--save-results PATH` | Pickle DataFrame + arrays par fold (positions, PnL, prix, poids des 5 seeds). |

**Sortie :** par fold, ligne avec `ensemble_test`, `long_only_test`, `buy_hold_test`, `single_best_test`, `|pos|`, `switches/bar`. Puis stats agrégées sur l'ensemble des folds.

### `viz_train.py` — visualisation live

Ouvre une fenêtre matplotlib (thème sombre, 20×11) qui montre les 5 GAs en course **en temps réel** sur un seul fold :

- **Top** : courbes de fitness train par génération (top + moyenne population) pour chaque seed.
- **Mid gauche** : heatmap consensus — poids moyens par seed × feature (rouge = long, bleu = short).
- **Mid droite** : population swarm en PCA 2D, étoile = top de chaque seed.
- **Final** : top-3 conservés (lw boostée), losers fanés, courbe d'equity de l'ensemble sur le test set en twinx.

```bash
# Configuration par défaut (ETH/USDT, rigorous, dernier fold disponible)
uv run python -m strategies.genetic_trading.viz_train

# Choisir un fold précis
uv run python -m strategies.genetic_trading.viz_train --fold 10

# Ralentir l'animation
uv run python -m strategies.genetic_trading.viz_train --frame-pause 0.2

# Sauvegarder la figure finale
uv run python -m strategies.genetic_trading.viz_train --save-final viz.png
```

## Prérequis données

Les deux scripts utilisent `utilities.data_manager.ExchangeDataManager` qui lit `./database/exchanges`. La paire / l'intervalle / l'échange demandés doivent y exister (`binance` 1h par défaut).
