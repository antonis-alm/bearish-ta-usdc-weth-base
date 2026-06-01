from datetime import UTC, datetime, timedelta
from decimal import Decimal

from almanak.framework.market import (
    MarketSnapshot,
    RSIData,
    TokenBalance,
    MACDData,
    BollingerBandsData,
    RSIUnavailableError,
)
from almanak.framework.teardown import TeardownMode

from strategy import BearishTaUsdcWethBaseStrategy


def _base_config() -> dict:
    return {
        "chain": "base",
        "protocol": "uniswap_v3",
        "base_token": "WETH",
        "quote_token": "USDC",
        "indicator_timeframe": "1h",
        "trade_size_usd": 300,
        "max_slippage_bps": 50,
        "max_price_impact": "0.20",
        "min_trade_value_usd": "25",
        "max_gas_ratio": "0.05",
        "rsi_period": 14,
        "rsi_entry": 24,
        "rsi_exit": 55,
        "macd_fast_period": 12,
        "macd_slow_period": 26,
        "macd_signal_period": 9,
        "macd_bullish_hist_threshold": "0",
        "macd_bearish_hist_threshold": "0",
        "use_bollinger_filter": True,
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "bb_entry_percent_b": "0.15",
        "take_profit_pct": "0.03",
        "stop_loss_pct": "0.015",
        "max_holding_hours": "12",
        "min_base_position_usd": "30",
        "force_action": "",
    }


def _market(
    *,
    weth_price: str = "2500",
    usdc_balance: str = "1000",
    usdc_balance_usd: str = "1000",
    weth_balance: str = "0",
    weth_balance_usd: str = "0",
    rsi_value: str = "50",
    macd_hist: str = "0",
    bb_percent_b: str = "0.10",
) -> MarketSnapshot:
    market = MarketSnapshot(chain="base", wallet_address="0x" + "1" * 40, timestamp=datetime.now(UTC))
    market.set_price("WETH", Decimal(weth_price))
    market.set_price("USDC", Decimal("1"))
    market.set_balance("USDC", TokenBalance(symbol="USDC", balance=Decimal(usdc_balance), balance_usd=Decimal(usdc_balance_usd)))
    market.set_balance("WETH", TokenBalance(symbol="WETH", balance=Decimal(weth_balance), balance_usd=Decimal(weth_balance_usd)))
    market.set_rsi("WETH", RSIData(value=Decimal(rsi_value), period=14))
    market.set_macd(
        "WETH",
        MACDData(
            macd_line=Decimal(macd_hist),
            signal_line=Decimal("0"),
            histogram=Decimal(macd_hist),
        ),
    )
    market.set_bollinger_bands(
        "WETH",
        BollingerBandsData(
            upper_band=Decimal("2600"),
            middle_band=Decimal("2500"),
            lower_band=Decimal("2400"),
            bandwidth=Decimal("0.05"),
            percent_b=Decimal(bb_percent_b),
        ),
    )
    return market


def _strategy(config_overrides: dict | None = None) -> BearishTaUsdcWethBaseStrategy:
    config = _base_config()
    if config_overrides:
        config.update(config_overrides)
    return BearishTaUsdcWethBaseStrategy(
        config=config,
        chain="base",
        wallet_address="0x" + "1" * 40,
    )


def test_buy_signal_emits_swap():
    strategy = _strategy()
    market = _market(rsi_value="20", macd_hist="0.10", bb_percent_b="0.05")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "USDC"
    assert intent.to_token == "WETH"
    assert intent.amount_usd == Decimal("300")


def test_entry_blocked_when_bollinger_filter_fails():
    strategy = _strategy()
    market = _market(rsi_value="20", macd_hist="0.10", bb_percent_b="0.40")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"


def test_entry_blocked_when_insufficient_quote_balance():
    strategy = _strategy()
    market = _market(rsi_value="20", macd_hist="0.10", bb_percent_b="0.05", usdc_balance_usd="50")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"
    assert "Insufficient" in (intent.reason or "")


def test_exit_on_stop_loss_emits_sell_all():
    strategy = _strategy()
    strategy._holding_base = True
    strategy._entry_price = Decimal("2500")
    strategy._entry_timestamp = datetime.now(UTC) - timedelta(hours=1)
    market = _market(weth_price="2400", weth_balance="0.2", weth_balance_usd="480", rsi_value="40", macd_hist="0.1")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "WETH"
    assert intent.to_token == "USDC"
    assert intent.amount == "all"


def test_exit_on_take_profit_emits_sell_all():
    strategy = _strategy()
    strategy._holding_base = True
    strategy._entry_price = Decimal("2500")
    strategy._entry_timestamp = datetime.now(UTC) - timedelta(hours=1)
    market = _market(weth_price="2600", weth_balance="0.2", weth_balance_usd="520", rsi_value="40", macd_hist="0.1")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.amount == "all"


def test_exit_on_max_holding_time_emits_sell_all():
    strategy = _strategy()
    strategy._holding_base = True
    strategy._entry_price = Decimal("2500")
    strategy._entry_timestamp = datetime.now(UTC) - timedelta(hours=13)
    market = _market(weth_price="2500", weth_balance="0.2", weth_balance_usd="500", rsi_value="40", macd_hist="0.1")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.amount == "all"


def test_exit_on_bearish_signal_emits_sell_all():
    strategy = _strategy()
    strategy._holding_base = True
    strategy._entry_price = Decimal("2500")
    strategy._entry_timestamp = datetime.now(UTC) - timedelta(hours=1)
    market = _market(weth_price="2500", weth_balance="0.2", weth_balance_usd="500", rsi_value="60", macd_hist="-0.2")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.amount == "all"


def test_gas_gate_blocks_trade(monkeypatch):
    strategy = _strategy()
    market = _market(rsi_value="20", macd_hist="0.1", bb_percent_b="0.05")
    monkeypatch.setattr(market, "is_trade_worthwhile", lambda **kwargs: False)

    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"
    assert "worthwhile" in (intent.reason or "")


def test_force_action_buy_bypasses_signals():
    strategy = _strategy({"force_action": "buy"})
    market = _market(rsi_value="70", macd_hist="-0.2", bb_percent_b="0.9")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "USDC"
    assert intent.to_token == "WETH"


def test_force_action_sell_bypasses_signals():
    strategy = _strategy({"force_action": "sell"})
    market = _market(rsi_value="20", macd_hist="0.2")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "WETH"
    assert intent.to_token == "USDC"
    assert intent.amount == "all"


def test_rsi_unavailable_returns_hold(monkeypatch):
    strategy = _strategy()
    market = _market()

    def _raise(*args, **kwargs):
        raise RSIUnavailableError(reason="missing")

    monkeypatch.setattr(market, "rsi", _raise)
    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"
    assert "RSI unavailable" in (intent.reason or "")


def test_teardown_soft_and_hard_slippage():
    strategy = _strategy()
    strategy._holding_base = True
    market = _market(weth_balance="0.2", weth_balance_usd="500")

    soft = strategy.generate_teardown_intents(mode=TeardownMode.SOFT, market=market)
    hard = strategy.generate_teardown_intents(mode=TeardownMode.HARD, market=market)

    assert len(soft) == 1
    assert len(hard) == 1
    assert soft[0].from_token == "WETH"
    assert hard[0].to_token == "USDC"
    assert hard[0].max_slippage >= soft[0].max_slippage


def test_persistent_state_round_trip():
    strategy = _strategy()
    strategy._holding_base = True
    strategy._entry_price = Decimal("2500")
    strategy._entry_timestamp = datetime.now(UTC)
    strategy._last_signal = "BUY"

    state = strategy.get_persistent_state()

    fresh = _strategy()
    fresh.load_persistent_state(state)

    assert fresh._holding_base is True
    assert fresh._entry_price == Decimal("2500")
    assert fresh._entry_timestamp is not None
    assert fresh._last_signal == "BUY"
