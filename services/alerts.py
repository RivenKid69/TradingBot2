"""
Alert delivery system for CustodiaCloud platform.

Supports multiple delivery channels:
- noop: Silent (for testing)
- telegram: Telegram bot API
- http: Generic HTTP POST webhook
- webhook: Alias for http

Configuration via settings dict or environment variables.
See AlertManager docstring for details.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional

import requests


logger = logging.getLogger(__name__)


def _get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from ``cfg`` supporting both mappings and objects."""

    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def send_http_webhook(text: str, config: Any | None = None) -> bool:
    """Send an alert via generic HTTP POST webhook.

    Configuration via config dict or environment:
        url / ALERT_WEBHOOK_URL: Webhook endpoint URL (required)
        method: HTTP method (default: POST)
        headers: Additional headers dict
        payload_template: JSON payload template with {text} placeholder
        timeout_sec: Request timeout (default: 10)

    Example config:
        {
            "url": "https://hooks.example.com/alert",
            "headers": {"Authorization": "Bearer token"},
            "payload_template": {"message": "{text}", "source": "custodia"}
        }
    """
    url = _get_cfg_value(config, "url") or os.getenv("ALERT_WEBHOOK_URL")
    if not url:
        logger.warning("HTTP webhook URL not configured (set url or ALERT_WEBHOOK_URL)")
        return False

    method = str(_get_cfg_value(config, "method", "POST") or "POST").upper()
    timeout = float(_get_cfg_value(config, "timeout_sec", 10.0) or 10.0)
    headers = _get_cfg_value(config, "headers") or {}

    # Build payload
    payload_template = _get_cfg_value(config, "payload_template")
    if payload_template and isinstance(payload_template, Mapping):
        # Replace {text} placeholders in template
        payload = {}
        for k, v in payload_template.items():
            if isinstance(v, str):
                payload[k] = v.replace("{text}", text)
            else:
                payload[k] = v
    else:
        # Default payload
        payload = {"text": text, "timestamp": time.time()}

    try:
        if method == "GET":
            response = requests.get(url, params=payload, headers=headers, timeout=timeout)
        else:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)

        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as exc:
        logger.warning("failed to send HTTP webhook alert: %s", exc, exc_info=exc)
        return False


def send_telegram(text: str, config: Any | None = None) -> bool:
    """Send a Telegram message using bot credentials from environment.

    Environment variables:
        TELEGRAM_BOT_TOKEN: Bot token.
        TELEGRAM_CHAT_ID: Chat ID of the recipient.
    """
    token_env = _get_cfg_value(config, "bot_token_env", "TELEGRAM_BOT_TOKEN")
    chat_id_env = _get_cfg_value(config, "chat_id_env", "TELEGRAM_CHAT_ID")
    token = _get_cfg_value(config, "bot_token") or (os.getenv(token_env) if token_env else None)
    chat_id = _get_cfg_value(config, "chat_id") or (os.getenv(chat_id_env) if chat_id_env else None)
    if not token or not chat_id:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    api_base = _get_cfg_value(config, "api_base_url", "https://api.telegram.org")
    timeout = float(_get_cfg_value(config, "timeout_sec", 10.0) or 10.0)
    payload = {"chat_id": chat_id, "text": text}

    extra_payload = _get_cfg_value(config, "extra_payload")
    if isinstance(extra_payload, Mapping):
        payload.update(extra_payload)

    url = f"{api_base.rstrip('/')}/bot{token}/sendMessage"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("failed to send telegram alert: %s", exc, exc_info=exc)
        return False

    return True


class AlertManager:
    """Manage alert notifications with cooldown control.

    Supported channels:
    - noop: Silent/no-op (for testing)
    - telegram: Telegram bot API (requires TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    - http: Generic HTTP POST webhook (requires url or ALERT_WEBHOOK_URL)
    - webhook: Alias for http

    Configuration example:
        settings = {
            "channel": "telegram",
            "cooldown_sec": 60.0,
            "telegram": {
                "bot_token": "...",
                "chat_id": "..."
            }
        }
        manager = AlertManager(settings)
        manager.notify("price_alert", "BTC crossed $50k")
    """

    def __init__(self, settings: Any | None = None) -> None:
        channel = _get_cfg_value(settings, "channel", "noop") or "noop"
        cooldown = _get_cfg_value(settings, "cooldown_sec", 0.0)
        telegram_cfg = _get_cfg_value(settings, "telegram")
        http_cfg = _get_cfg_value(settings, "http") or _get_cfg_value(settings, "webhook")

        self.cooldown_sec = float(cooldown or 0.0)
        self._last_sent: Dict[str, float] = {}
        self._channels: Dict[str, Callable[[str], bool]] = {
            "noop": lambda text: True,
            "telegram": lambda text, cfg=telegram_cfg: send_telegram(text, cfg),
            "http": lambda text, cfg=http_cfg: send_http_webhook(text, cfg),
            "webhook": lambda text, cfg=http_cfg: send_http_webhook(text, cfg),
        }

        if channel not in self._channels:
            logger.warning("unknown alert channel '%s', falling back to noop", channel)
            channel = "noop"

        self._channel = channel
        self._send = self._channels[channel]

    def notify(self, key: str, text: str) -> None:
        """Send `text` if `cooldown_sec` has passed for `key`."""
        now = time.time()
        if self.cooldown_sec > 0:
            last_time = self._last_sent.get(key)
            if last_time is not None and now - last_time < self.cooldown_sec:
                return

        try:
            result = self._send(text)
        except Exception:
            logger.exception("failed to send alert '%s' via %s", key, self._channel)
            return

        if result is False:
            logger.debug(
                "alert '%s' via %s reported failure, skipping cooldown", key, self._channel
            )
            return

        if self.cooldown_sec > 0:
            self._last_sent[key] = now
