import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from almanak.framework.intents import Intent
from almanak.framework.market import (
    BalanceUnavailableError,
    IndicatorUnavailableError,
    MarketSnapshot,
    PriceUnavailableError,
    RSIUnavailableError,
)
from almanak.framework.strategies import IntentStrategy, almanak_strategy

logger = logging.getLogger(__name__)


@almanak_strategy(
    name="bearish_ta_usdc_weth_base",
    description="Bearish-biased TA swap strategy for USDC/WETH on Base",
    version="1.0.0",
    author="Almanak",
    tags=["ta", "bearish", "swap", "base"],
    supported_chains=["base"],
    supported_protocols=["uniswap_v3"],
    intent_types=["SWAP", "HOLD"],
    default_chain="base",
)
class BearishTaUsdcWethBaseStrategy(IntentStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.protocol = str(self.get_config("protocol", "uniswap_v3"))
        self.base_token = str(self.get_config("base_token", "WETH"))
        self.quote_token = str(self.get_config("quote_token", "USDC"))

        self.indicator_timeframe = str(self.get_config("indicator_timeframe", "1h"))

        self.trade_size_usd = Decimal(str(self.get_config("trade_size_usd", "300")))
        self.max_slippage_bps = int(self.get_config("max_slippage_bps", 50))
        self.max_price_impact = Decimal(str(self.get_config("max_price_impact", "0.20")))

        self.min_trade_value_usd = Decimal(str(self.get_config("min_trade_value_usd", "25")))
        self.max_gas_ratio = Decimal(str(self.get_config("max_gas_ratio", "0.05")))

        self.rsi_period = int(self.get_config("rsi_period", 14))
        self.rsi_entry = Decimal(str(self.get_config("rsi_entry", "24")))
        self.rsi_exit = Decimal(str(self.get_config("rsi_exit", "55")))

        self.macd_fast_period = int(self.get_config("macd_fast_period", 12))
        self.macd_slow_period = int(self.get_config("macd_slow_period", 26))
        self.macd_signal_period = int(self.get_config("macd_signal_period", 9))
        self.macd_bullish_hist_threshold = Decimal(str(self.get_config("macd_bullish_hist_threshold", "0")))
        self.macd_bearish_hist_threshold = Decimal(str(self.get_config("macd_bearish_hist_threshold", "0")))

        self.use_bollinger_filter = bool(self.get_config("use_bollinger_filter", True))
        self.bb_period = int(self.get_config("bb_period", 20))
        self.bb_std_dev = float(self.get_config("bb_std_dev", 2.0))
        self.bb_entry_percent_b = Decimal(str(self.get_config("bb_entry_percent_b", "0.15")))

        self.take_profit_pct = Decimal(str(self.get_config("take_profit_pct", "0.03")))
        self.stop_loss_pct = Decimal(str(self.get_config("stop_loss_pct", "0.015")))
        self.max_holding_hours = Decimal(str(self.get_config("max_holding_hours", "12")))
        self.min_base_position_usd = Decimal(str(self.get_config("min_base_position_usd", "30")))

        self.force_action = str(self.get_config("force_action", "") or "").lower().strip()

        self._holding_base = False
        self._entry_price: Decimal | None = None
        self._entry_timestamp: datetime | None = None
        self._last_signal = "INIT"

    def decide(self, market: MarketSnapshot) -> Intent:
        if self.force_action:
            return self._forced_intent()

        try:
            base_price = market.price(self.base_token)
        except (PriceUnavailableError, ValueError) as exc:
            return Intent.hold(reason=f"Price unavailable: {exc}")

        try:
            quote_balance = market.balance(self.quote_token)
            base_balance = market.balance(self.base_token)
        except (BalanceUnavailableError, ValueError) as exc:
            return Intent.hold(reason=f"Balance unavailable: {exc}")

        try:
            rsi = market.rsi(
                self.base_token,
                period=self.rsi_period,
                timeframe=self.indicator_timeframe,
            )
        except (RSIUnavailableError, IndicatorUnavailableError, ValueError) as exc:
            return Intent.hold(reason=f"RSI unavailable: {exc}")

        try:
            macd = market.macd(
                self.base_token,
                fast_period=self.macd_fast_period,
                slow_period=self.macd_slow_period,
                signal_period=self.macd_signal_period,
                timeframe=self.indicator_timeframe,
            )
        except (IndicatorUnavailableError, ValueError) as exc:
            return Intent.hold(reason=f"MACD unavailable: {exc}")

        bb_data = None
        if self.use_bollinger_filter:
            try:
                bb_data = market.bollinger_bands(
                    self.base_token,
                    period=self.bb_period,
                    std_dev=self.bb_std_dev,
                    timeframe=self.indicator_timeframe,
                )
            except (IndicatorUnavailableError, ValueError) as exc:
                return Intent.hold(reason=f"Bollinger data unavailable: {exc}")

        has_base_position = base_balance.balance_usd >= self.min_base_position_usd
        self._holding_base = has_base_position

        if has_base_position and self._entry_timestamp is None:
            self._entry_timestamp = market.timestamp
        if has_base_position and self._entry_price is None:
            self._entry_price = base_price

        if has_base_position:
            return self._maybe_exit_position(
                market=market,
                base_price=base_price,
                rsi_value=Decimal(rsi.value),
                macd_hist=Decimal(macd.histogram),
            )

        entry_signal = Decimal(rsi.value) <= self.rsi_entry and Decimal(macd.histogram) >= self.macd_bullish_hist_threshold
        if bb_data is not None:
            entry_signal = entry_signal and Decimal(bb_data.percent_b) <= self.bb_entry_percent_b

        if not entry_signal:
            self._last_signal = "NO_ENTRY"
            return Intent.hold(reason="No bearish-bias entry signal")

        if quote_balance.balance_usd < self.trade_size_usd:
            self._last_signal = "ENTRY_BLOCKED_NO_QUOTE"
            return Intent.hold(
                reason=f"Insufficient {self.quote_token}: ${quote_balance.balance_usd} < ${self.trade_size_usd}"
            )

        if not self._is_trade_worthwhile(market):
            self._last_signal = "ENTRY_BLOCKED_GAS"
            return Intent.hold(reason="Trade not worthwhile vs gas")

        self._last_signal = "BUY"
        return self._build_buy_intent()

    def _forced_intent(self) -> Intent:
        if self.force_action == "buy":
            return self._build_buy_intent()
        if self.force_action == "sell":
            return self._build_sell_intent(all_amount=True)
        raise ValueError(f"Unknown force_action: {self.force_action!r}")

    def _maybe_exit_position(
        self,
        *,
        market: MarketSnapshot,
        base_price: Decimal,
        rsi_value: Decimal,
        macd_hist: Decimal,
    ) -> Intent:
        exit_reasons: list[str] = []

        if self._entry_price is not None and self._entry_price > 0:
            if base_price <= self._entry_price * (Decimal("1") - self.stop_loss_pct):
                exit_reasons.append("stop_loss")
            if base_price >= self._entry_price * (Decimal("1") + self.take_profit_pct):
                exit_reasons.append("take_profit")

        if self._entry_timestamp is not None:
            elapsed_seconds = (market.timestamp - self._entry_timestamp).total_seconds()
            elapsed_hours = Decimal(str(elapsed_seconds)) / Decimal("3600")
            if elapsed_hours >= self.max_holding_hours:
                exit_reasons.append("max_holding_time")

        if rsi_value >= self.rsi_exit:
            exit_reasons.append("rsi_exit")
        if macd_hist <= self.macd_bearish_hist_threshold:
            exit_reasons.append("macd_bearish")

        if not exit_reasons:
            self._last_signal = "HOLD_BASE"
            return Intent.hold(reason="Holding base; exit conditions not met")

        if not self._is_trade_worthwhile(market):
            self._last_signal = "EXIT_BLOCKED_GAS"
            return Intent.hold(reason="Exit signal present but trade not worthwhile vs gas")

        self._last_signal = f"SELL:{','.join(exit_reasons)}"
        return self._build_sell_intent(all_amount=True)

    def _build_buy_intent(self) -> Intent:
        return Intent.swap(
            from_token=self.quote_token,
            to_token=self.base_token,
            amount_usd=self.trade_size_usd,
            max_slippage=self._max_slippage(),
            max_price_impact=self.max_price_impact,
            protocol=self.protocol,
            chain=self.chain,
        )

    def _build_sell_intent(self, *, all_amount: bool) -> Intent:
        sell_kwargs: dict[str, Any] = {
            "from_token": self.base_token,
            "to_token": self.quote_token,
            "max_slippage": self._max_slippage(),
            "max_price_impact": self.max_price_impact,
            "protocol": self.protocol,
            "chain": self.chain,
        }
        if all_amount:
            sell_kwargs["amount"] = "all"
        else:
            sell_kwargs["amount_usd"] = self.trade_size_usd
        return Intent.swap(**sell_kwargs)

    def _is_trade_worthwhile(self, market: MarketSnapshot) -> bool:
        if self.trade_size_usd < self.min_trade_value_usd:
            return False
        return market.is_trade_worthwhile(
            amount_usd=self.trade_size_usd,
            chain=self.chain,
            max_gas_ratio=self.max_gas_ratio,
        )

    def _max_slippage(self) -> Decimal:
        return Decimal(str(self.max_slippage_bps)) / Decimal("10000")

    def supports_teardown(self) -> bool:
        return True

    def get_open_positions(self):
        from almanak.framework.teardown import PositionInfo, PositionType, TeardownPositionSummary

        positions: list[PositionInfo] = []
        try:
            market = self.create_market_snapshot()
            base_balance = market.balance(self.base_token)
            if base_balance.balance > 0:
                positions.append(
                    PositionInfo(
                        position_type=PositionType.TOKEN,
                        position_id=f"{self.STRATEGY_NAME}_base",
                        chain=self.chain,
                        protocol=self.protocol,
                        value_usd=base_balance.balance_usd,
                        details={
                            "asset": self.base_token,
                            "quote": self.quote_token,
                            "balance": str(base_balance.balance),
                        },
                    )
                )
        except (BalanceUnavailableError, ValueError):
            positions = []

        return TeardownPositionSummary(
            deployment_id=getattr(self, "deployment_id", self.STRATEGY_NAME),
            timestamp=datetime.now(UTC),
            positions=positions,
        )

    def generate_teardown_intents(self, mode=None, market=None) -> list[Intent]:
        from almanak.framework.teardown import TeardownMode

        has_position = self._holding_base
        if market is not None:
            try:
                has_position = market.balance(self.base_token).balance > 0
            except (BalanceUnavailableError, ValueError):
                has_position = self._holding_base

        if not has_position:
            return []

        max_slippage = Decimal("0.03") if mode == TeardownMode.HARD else self._max_slippage()
        return [
            Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount="all",
                max_slippage=max_slippage,
                max_price_impact=self.max_price_impact,
                protocol=self.protocol,
                chain=self.chain,
            )
        ]

    def on_intent_executed(self, intent, success: bool, result) -> None:
        if not success:
            return
        intent_type = getattr(getattr(intent, "intent_type", None), "value", "")
        if intent_type != "SWAP":
            return

        from_token = getattr(intent, "from_token", "")
        to_token = getattr(intent, "to_token", "")
        if from_token == self.quote_token and to_token == self.base_token:
            self._holding_base = True
            self._entry_timestamp = datetime.now(UTC)
            self._entry_price = None
        elif from_token == self.base_token and to_token == self.quote_token:
            self._holding_base = False
            self._entry_timestamp = None
            self._entry_price = None

    def get_persistent_state(self):
        return {
            "holding_base": self._holding_base,
            "entry_price": str(self._entry_price) if self._entry_price is not None else None,
            "entry_timestamp": self._entry_timestamp.isoformat() if self._entry_timestamp else None,
            "last_signal": self._last_signal,
        }

    def load_persistent_state(self, state):
        if not state:
            return
        self._holding_base = bool(state.get("holding_base", False))
        entry_price_raw = state.get("entry_price")
        self._entry_price = Decimal(str(entry_price_raw)) if entry_price_raw is not None else None
        entry_ts_raw = state.get("entry_timestamp")
        self._entry_timestamp = datetime.fromisoformat(entry_ts_raw) if entry_ts_raw else None
        self._last_signal = str(state.get("last_signal", "INIT"))

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": self.STRATEGY_NAME,
            "chain": self.chain,
            "pair": f"{self.base_token}/{self.quote_token}",
            "holding_base": self._holding_base,
            "entry_price": str(self._entry_price) if self._entry_price is not None else None,
            "entry_timestamp": self._entry_timestamp.isoformat() if self._entry_timestamp else None,
            "last_signal": self._last_signal,
            "force_action": self.force_action,
        }
