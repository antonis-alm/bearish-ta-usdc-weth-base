from types import SimpleNamespace
from unittest.mock import patch

from dashboard.ui import render_custom_dashboard


def test_dashboard_imports_and_exposes_renderer():
    assert callable(render_custom_dashboard)


def test_render_custom_dashboard_builds_rsi_template_config():
    strategy_config = {
        "rsi_period": 21,
        "rsi_entry": 22,
        "rsi_exit": 58,
        "base_token": "WETH",
        "quote_token": "USDC",
        "chain": "base",
        "protocol": "uniswap_v3",
    }
    initial_state = {"existing": "value"}
    enriched_state = {"price_history": [], "existing": "value"}
    config = SimpleNamespace(
        base_token="ETH",
        quote_token="USDC",
        chain="arbitrum",
        protocol="uniswap_v3",
    )

    api_client = object()

    with (
        patch("dashboard.ui.st.title") as mock_title,
        patch("dashboard.ui.get_rsi_config", return_value=config) as mock_get_rsi_config,
        patch("dashboard.ui.prepare_ta_session_state", return_value=enriched_state) as mock_prepare,
        patch("dashboard.ui.render_ta_dashboard") as mock_render,
    ):
        render_custom_dashboard("dep-1", strategy_config, api_client=api_client, session_state=initial_state)

    mock_title.assert_called_once_with("Bearish TA USDC/WETH (Base)")
    mock_get_rsi_config.assert_called_once_with(period=21, overbought=58.0, oversold=22.0)
    mock_prepare.assert_called_once_with(
        api_client,
        session_state=initial_state,
        config=config,
    )
    mock_render.assert_called_once_with("dep-1", strategy_config, enriched_state, config)

    assert config.base_token == "WETH"
    assert config.quote_token == "USDC"
    assert config.chain == "base"
    assert config.protocol == "uniswap_v3"
