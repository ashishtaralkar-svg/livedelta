# Delta Exchange Intraday Trend Bot

Production-grade Python 3.12 bot that trades a Delta Exchange BTC perpetual on a
**selective intraday trend** strategy (a faithful port of `ashish.pine`). On each
closed candle it enters **only** when price clears *all* of: previous-day open,
previous-day close, 50-EMA(high), 50-EMA(low), **and** the Supertrend(ATR=10,
multiplier=3) is aligned. It may sit **flat**, holds one position at a time, and
exits on either a **previous-candle stop-loss** or a **forced square-off** at the
configured cut-off time (default 17:30 IST). The previous-day levels use a custom
day boundary (default 05:30 IST). One position at a time, 0.001 BTC = 1 contract,
10x leverage. Includes a backtester and Telegram notifications, and runs 24/7 in
Docker on AWS EC2.

> **Symbol note:** On **Delta India** the BTC perpetual is **`BTCUSD`** (product_id 27).
> `BTCUSDT` (product_id 139) exists only on **Delta Global**. Set `DELTA_REGION` and
> `DELTA_SYMBOL` accordingly; the numeric `product_id` is resolved automatically at startup.

## Architecture

```
WS candlestick_1m ─▶ CandleAggregator ─(closed bar)─▶ PineStrategy
        │                                                     │
        ▼                                      StrategyDecision{exits, entries}
  heartbeat watchdog                                          │
  + auto-reconnect ─▶ reconcile (REST = truth)                ▼
                                              plan_actions (close, then open)
                                                          │
                                              OrderEngine (close→open, lock,
                                              retry, fill-confirm via REST)
                                                          │
                                              TradeLedger + Telegram notify
```

Key modules (`src/deltabot/`):

| Module | Responsibility |
| --- | --- |
| `strategy/pine_strategy.py` | Pine port: prev-day O/C + Supertrend + EMA(H/L) entries, SL & square-off exits. **Shared by live + backtest.** |
| `strategy/supertrend.py` | Repaint-safe incremental Supertrend (entry filter). |
| `strategy/indicators.py` | Incremental `ta.ema`-equivalent EMA. |
| `core/candle_aggregator.py` | Emits a candle only when it has closed (start-time rollover). |
| `core/state_machine.py` | Builds ordered CLOSE/OPEN action plans from a strategy decision. |
| `core/order_engine.py` | Market orders, `asyncio.Lock`, retry/backoff, fill confirmation. |
| `core/reconciler.py` | Aligns bot state to the exchange (positions = source of truth). |
| `core/trader.py` | Orchestrates the live event loop, warmup, gap re-seed, notifications. |
| `exchange/signer.py` | HMAC-SHA256 REST + WS signing. |
| `exchange/rest_client.py` | Resilient REST client (orders, positions, leverage, candles). |
| `exchange/ws_manager.py` | Connect/auth/subscribe/heartbeat/auto-reconnect. |
| `notify/telegram.py` | Async, non-blocking Telegram alerts. |
| `backtest/` | Historical download + simulation (same strategy) + metrics. |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                               # then fill in keys
```

## Run tests

```bash
pytest
```

## Backtest

Downloads (and caches) historical 1m candles, runs the identical strategy logic
offline, and prints metrics (total/win/lose trades, win rate, net profit, profit
factor, max drawdown, monthly returns).

```bash
python -m deltabot.cli backtest --start 2025-01-01 --end 2025-03-01 \
    --symbol BTCUSD --equity-csv equity.csv
```

## Live trading

```bash
# Edit .env: keys, DELTA_TESTNET=true to start on testnet.
python -m deltabot.cli live
```

The engine: sets leverage → warms up the indicators and prev-day levels from REST
history (paged to cover `DELTA_WARMUP_DAYS` days) → reconciles with the exchange →
streams 1m candles → enters/exits per the strategy on each closed candle.

## Deploy on AWS EC2 (Ubuntu 24.04, Docker)

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker            # auto-start Docker on reboot

git clone <repo> /opt/deltabot && cd /opt/deltabot
cp .env.example .env                          # fill in keys
docker compose up -d --build                  # restart: always handles crashes + reboots
docker compose logs -f                        # tail JSON logs
```

- **Auto-start on reboot / auto-restart on crash:** `restart: always` in
  `docker-compose.yml` plus an enabled Docker service. An optional
  `deploy/deltabot.service` systemd unit is provided.
- **CloudWatch logs:** logs are structured JSON on stdout — ship them with the
  CloudWatch agent or swap the compose logging driver to `awslogs`.
- **Secrets:** provided via `.env` (gitignored); never baked into the image.
- **Shutdown:** on SIGTERM the bot stops the stream and drains the Telegram queue.
  With `DELTA_CLOSE_ON_SHUTDOWN=false` (default) it keeps the open position and
  reconciles from the exchange on restart.

## Configuration

All settings are environment variables with the `DELTA_` prefix — see
[`.env.example`](.env.example) for the full list.

## Safety notes

- Default is **testnet**. Validate with a multi-hour testnet soak before setting
  `DELTA_TESTNET=false`.
- The backtest PnL uses the linear (USD/USDT) convention to evaluate the
  directional edge; exact inverse-contract settlement on BTCUSD differs but trade
  ranking, win rate, profit factor and drawdown are governed by the captured move.
- This software is provided as-is; trading leveraged derivatives carries risk of
  loss. Use at your own risk.
```
