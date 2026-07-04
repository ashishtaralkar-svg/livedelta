"""Configuration via environment variables (``DELTA_`` prefix) and ``.env``."""

from __future__ import annotations

from typing import Literal

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base URLs keyed by (region, testnet).
_REST_URLS: dict[tuple[str, bool], str] = {
    ("india", False): "https://api.india.delta.exchange",
    ("india", True): "https://cdn-ind.testnet.deltaex.org",
    ("global", False): "https://api.delta.exchange",
    ("global", True): "https://testnet-api.delta.exchange",
}

_WS_URLS: dict[tuple[str, bool], str] = {
    ("india", False): "wss://socket.india.delta.exchange",
    ("india", True): "wss://socket-ind.testnet.deltaex.org",
    ("global", False): "wss://socket.delta.exchange",
    ("global", True): "wss://testnet-api.delta.exchange",
}


class Settings(BaseSettings):
    """Runtime configuration. All fields overridable via ``DELTA_*`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="DELTA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials ---
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")

    # --- Exchange / market ---
    testnet: bool = True
    region: Literal["india", "global"] = "india"
    symbol: str = "BTCUSD"
    product_id: int | None = None  # resolved at startup if unset

    # --- Strategy / sizing ---
    contracts: int = 1  # 0.001 BTC == 1 contract (contract_value 0.001)
    leverage: int = 10
    atr_period: int = 10
    st_multiplier: float = 3.0
    ema_length: int = 50  # 50-EMA of high and of low (entry filters)
    use_close: bool = True  # compare bar close (False = high/low) against the levels
    resolution: str = "1m"
    warmup_candles: int = 200
    warmup_days: int = 3  # also pull at least this many days so prev-day levels exist

    # --- Custom day boundary (previous-day open/close) & forced square-off ---
    day_tz: str = "Asia/Kolkata"
    day_start_hour: int = 5  # a new custom "day" begins at 05:30 in day_tz
    day_start_minute: int = 30
    square_off_hour: int = 17  # force-exit any open position at 17:25 in day_tz
    square_off_minute: int = 25
    # Entries resume at this IST time (after the 17:30 options settlement). Between
    # the square-off and this time the bot stays flat (the settlement window).
    entry_resume_hour: int = 17
    entry_resume_minute: int = 30
    # Day-of-week entry filter: comma-separated IST day names whose NEW entries are
    # blocked (exits still run). e.g. "Sat" or "Sat,Sun". Empty = trade every day.
    skip_weekdays: str = ""

    # --- Notifications ---
    telegram_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    # --- Options execution mode ---
    options_mode: bool = False  # True = trade C/P (call/put) options instead of futures
    option_offset: int = 400  # points from current price for strike (e.g. 400 ITM)
    option_contracts: int = 1  # number of option lots to sell
    option_strike_interval: int = 200  # fallback rounding only; primary path snaps to nearest listed strike
    option_expiry_cutoff_hour: int = 17  # IST hour; if past this, use next-day expiry
    option_min_available_balance: float = 0.0  # skip selling if available balance below this (0 = no check)
    option_margin_asset: str | None = None  # wallet asset to check balance for (None = max across wallets)

    # --- Strategy selector ---
    # "pine"     = PineStrategy (EMA/Supertrend, fixed ITM offset)
    # "revbreak" = RevBreakStrategy (prev-day-zone breakout, premium-targeted strike, option TP)
    strategy: str = "pine"

    # RevBreak-specific settings (ignored when strategy="pine")
    target_premium: float = 900.0       # target option mark price at entry
    take_profit_pct: float = 70.0       # option TP: exit when premium falls by this %
    revbreak_gate: str = "open"         # "open" = vs today's 05:30 open; "zone" = prev-day O/C zone
    revbreak_st_filter: bool = True     # require Supertrend aligned to enter
    revbreak_reentry_block: bool = True  # block same-dir re-entry after a TP until an ST flip
    revbreak_tp_poll_seconds: float = 15.0  # how often to poll the option mark for the TP (0 = only at 5m close)
    revbreak_max_sl_distance: float = 0.0  # skip trades where BTC SL is > this many points from entry (0 = no limit)
    # Percentage-of-price SL band for REAL execution (adapts to BTC price). When
    # max_sl_pct > 0 this band takes precedence over the fixed max_sl_distance:
    # a trade whose BTC SL distance is < min_sl_pct or > max_sl_pct of the entry
    # price is out-of-band (paper-traded if enabled, else skipped). E.g. 0.25 / 0.75.
    revbreak_min_sl_pct: float = 0.0  # floor: skip/paper trades with SL closer than this % of price
    revbreak_max_sl_pct: float = 0.0  # ceiling: skip/paper trades with SL wider than this % of price
    revbreak_paper_trade_wide_sl: bool = False  # paper-trade out-of-band SL trades instead of real; monitor only

    # State file for position ownership (prevents reconcile conflict when two bots share one account).
    # Each bot should have a DIFFERENT path. Empty = no state persistence (adopt any short on restart).
    state_file: str = ""

    # --- Operational ---
    close_on_shutdown: bool = False
    daily_summary_hour_utc: int = 0
    heartbeat_timeout_s: float = 35.0
    fill_confirm_timeout_s: float = 10.0
    log_level: str = "INFO"

    @property
    def skip_weekday_ints(self) -> frozenset[int]:
        """``skip_weekdays`` parsed to weekday ints (Mon=0 .. Sun=6)."""
        idx = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        out = {idx[t.strip().lower()[:3]] for t in self.skip_weekdays.replace(";", ",").split(",")
               if t.strip().lower()[:3] in idx}
        return frozenset(out)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rest_base_url(self) -> str:
        return _REST_URLS[(self.region, self.testnet)]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ws_url(self) -> str:
        return _WS_URLS[(self.region, self.testnet)]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)


def load_settings() -> Settings:
    """Load settings from environment / .env file."""
    return Settings()
