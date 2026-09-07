"""Comprehensive tests for services.alerts module."""

import os
import time
from unittest.mock import patch, MagicMock

import pytest

from services.alerts import send_telegram, AlertManager, _get_cfg_value


class TestGetCfgValue:
    """Tests for _get_cfg_value helper."""

    def test_get_from_dict(self):
        """Test getting value from dictionary."""
        cfg = {"key": "value"}
        assert _get_cfg_value(cfg, "key") == "value"

    def test_get_from_object(self):
        """Test getting value from object."""

        class Cfg:
            key = "value"

        assert _get_cfg_value(Cfg(), "key") == "value"

    def test_get_with_default(self):
        """Test getting with default value."""
        cfg = {}
        assert _get_cfg_value(cfg, "missing", "default") == "default"

    def test_get_from_none(self):
        """Test getting from None returns default."""
        assert _get_cfg_value(None, "key", "default") == "default"


class TestSendTelegram:
    """Tests for send_telegram function."""

    @patch("requests.post")
    def test_send_telegram_success(self, mock_post):
        """Test successful telegram message send."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = {
            "bot_token": "test_token",
            "chat_id": "12345",
        }

        result = send_telegram("Test message", config)

        assert result is True
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_send_telegram_uses_env_vars(self, mock_post):
        """Test send_telegram uses environment variables."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        with patch.dict(
            os.environ, {"TELEGRAM_BOT_TOKEN": "env_token", "TELEGRAM_CHAT_ID": "env_chat_id"}
        ):
            result = send_telegram("Test message", None)

            assert result is True
            # Check that URL was constructed with env token
            call_args = mock_post.call_args
            assert "env_token" in call_args[0][0]

    def test_send_telegram_missing_credentials(self):
        """Test send_telegram raises when credentials missing."""
        with pytest.raises(EnvironmentError, match="must be set"):
            send_telegram("Test message", None)

    @patch("requests.post")
    def test_send_telegram_request_failure(self, mock_post):
        """Test send_telegram handles request failure."""
        import requests

        mock_post.side_effect = requests.exceptions.RequestException("Connection error")

        config = {
            "bot_token": "test_token",
            "chat_id": "12345",
        }

        result = send_telegram("Test message", config)

        assert result is False

    @patch("requests.post")
    def test_send_telegram_http_error(self, mock_post):
        """Test send_telegram handles HTTP error."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_post.return_value = mock_response

        config = {
            "bot_token": "test_token",
            "chat_id": "12345",
        }

        result = send_telegram("Test message", config)

        assert result is False

    @patch("requests.post")
    def test_send_telegram_custom_api_base(self, mock_post):
        """Test send_telegram with custom API base URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = {
            "bot_token": "test_token",
            "chat_id": "12345",
            "api_base_url": "https://custom.api.com",
        }

        send_telegram("Test message", config)

        call_args = mock_post.call_args
        assert "custom.api.com" in call_args[0][0]

    @patch("requests.post")
    def test_send_telegram_custom_timeout(self, mock_post):
        """Test send_telegram with custom timeout."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = {
            "bot_token": "test_token",
            "chat_id": "12345",
            "timeout_sec": 5.0,
        }

        send_telegram("Test message", config)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["timeout"] == 5.0

    @patch("requests.post")
    def test_send_telegram_extra_payload(self, mock_post):
        """Test send_telegram with extra payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = {
            "bot_token": "test_token",
            "chat_id": "12345",
            "extra_payload": {
                "parse_mode": "Markdown",
                "disable_notification": True,
            },
        }

        send_telegram("Test message", config)

        call_kwargs = mock_post.call_args[1]
        json_data = call_kwargs["json"]
        assert json_data["parse_mode"] == "Markdown"
        assert json_data["disable_notification"] is True


class TestAlertManagerInit:
    """Tests for AlertManager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default settings."""
        manager = AlertManager(None)

        assert manager.cooldown_sec == 0.0
        assert manager._channel == "noop"

    def test_init_with_telegram_channel(self):
        """Test initialization with telegram channel."""
        settings = {
            "channel": "telegram",
            "cooldown_sec": 60.0,
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        assert manager._channel == "telegram"
        assert manager.cooldown_sec == 60.0

    def test_init_with_invalid_channel(self):
        """Test initialization with invalid channel falls back to noop."""
        settings = {"channel": "invalid_channel"}
        manager = AlertManager(settings)

        assert manager._channel == "noop"

    def test_init_with_cooldown(self):
        """Test initialization with cooldown."""
        settings = {"cooldown_sec": 120.0}
        manager = AlertManager(settings)

        assert manager.cooldown_sec == 120.0


class TestAlertManagerNotify:
    """Tests for AlertManager.notify method."""

    def test_notify_noop_channel(self):
        """Test notify with noop channel."""
        manager = AlertManager({"channel": "noop"})

        # Should not raise
        manager.notify("test_key", "Test message")

    @patch("services.alerts.send_telegram")
    def test_notify_telegram_channel(self, mock_send):
        """Test notify with telegram channel."""
        mock_send.return_value = True

        settings = {
            "channel": "telegram",
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        manager.notify("test_key", "Test message")

        mock_send.assert_called_once_with("Test message", settings["telegram"])

    @patch("services.alerts.send_telegram")
    def test_notify_respects_cooldown(self, mock_send):
        """Test notify respects cooldown period."""
        mock_send.return_value = True

        settings = {
            "channel": "telegram",
            "cooldown_sec": 0.1,
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        # First call should succeed
        manager.notify("test_key", "Message 1")
        assert mock_send.call_count == 1

        # Second call within cooldown should be ignored
        manager.notify("test_key", "Message 2")
        assert mock_send.call_count == 1

        # After cooldown, should succeed
        time.sleep(0.15)
        manager.notify("test_key", "Message 3")
        assert mock_send.call_count == 2

    @patch("services.alerts.send_telegram")
    def test_notify_different_keys_independent_cooldown(self, mock_send):
        """Test different keys have independent cooldowns."""
        mock_send.return_value = True

        settings = {
            "channel": "telegram",
            "cooldown_sec": 1.0,
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        manager.notify("key1", "Message 1")
        manager.notify("key2", "Message 2")

        # Both should be sent (different keys)
        assert mock_send.call_count == 2

    @patch("services.alerts.send_telegram")
    def test_notify_failure_does_not_update_cooldown(self, mock_send):
        """Test failed send doesn't update cooldown timer."""
        mock_send.return_value = False

        settings = {
            "channel": "telegram",
            "cooldown_sec": 0.1,
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        manager.notify("test_key", "Message 1")
        manager.notify("test_key", "Message 2")

        # Both should be attempted (no cooldown on failure)
        assert mock_send.call_count == 2

    @patch("services.alerts.send_telegram")
    def test_notify_exception_handling(self, mock_send):
        """Test notify handles exceptions gracefully."""
        mock_send.side_effect = Exception("Test error")

        settings = {
            "channel": "telegram",
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        # Should not raise
        manager.notify("test_key", "Test message")

    def test_notify_unsupported_channel(self):
        """Test notify with unsupported channel logs but doesn't fail."""
        settings = {"channel": "http"}  # Not implemented
        manager = AlertManager(settings)

        # Should not raise
        manager.notify("test_key", "Test message")

    def test_notify_webhook_channel(self):
        """Test notify with webhook channel (not implemented)."""
        settings = {"channel": "webhook"}
        manager = AlertManager(settings)

        # Should not raise
        manager.notify("test_key", "Test message")

    @patch("services.alerts.send_telegram")
    def test_notify_with_zero_cooldown(self, mock_send):
        """Test notify with zero cooldown sends every time."""
        mock_send.return_value = True

        settings = {
            "channel": "telegram",
            "cooldown_sec": 0.0,
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        manager.notify("test_key", "Message 1")
        manager.notify("test_key", "Message 2")
        manager.notify("test_key", "Message 3")

        # All should be sent (no cooldown)
        assert mock_send.call_count == 3


class TestAlertManagerEdgeCases:
    """Tests for AlertManager edge cases."""

    def test_cooldown_with_none_value(self):
        """Test cooldown with None value defaults to 0."""
        settings = {"cooldown_sec": None}
        manager = AlertManager(settings)

        assert manager.cooldown_sec == 0.0

    def test_cooldown_with_string_value(self):
        """Test cooldown coerces string to float."""
        settings = {"cooldown_sec": "60.0"}
        manager = AlertManager(settings)

        assert manager.cooldown_sec == 60.0

    @patch("services.alerts.send_telegram")
    def test_notify_empty_message(self, mock_send):
        """Test notify with empty message."""
        mock_send.return_value = True

        settings = {
            "channel": "telegram",
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        manager.notify("test_key", "")
        mock_send.assert_called_once_with("", settings["telegram"])

    @patch("services.alerts.send_telegram")
    def test_notify_unicode_message(self, mock_send):
        """Test notify with unicode characters."""
        mock_send.return_value = True

        settings = {
            "channel": "telegram",
            "telegram": {
                "bot_token": "test_token",
                "chat_id": "12345",
            },
        }
        manager = AlertManager(settings)

        message = "Test 测试 тест 🚀"
        manager.notify("test_key", message)
        mock_send.assert_called_once_with(message, settings["telegram"])


class TestSendTelegramEdgeCases:
    """Tests for send_telegram edge cases."""

    @patch("requests.post")
    def test_send_telegram_custom_env_var_names(self, mock_post):
        """Test send_telegram with custom env var names."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = {
            "bot_token_env": "CUSTOM_TOKEN_VAR",
            "chat_id_env": "CUSTOM_CHAT_VAR",
        }

        with patch.dict(
            os.environ, {"CUSTOM_TOKEN_VAR": "custom_token", "CUSTOM_CHAT_VAR": "custom_chat"}
        ):
            result = send_telegram("Test", config)
            assert result is True

    @patch("requests.post")
    def test_send_telegram_empty_text(self, mock_post):
        """Test send_telegram with empty text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = {
            "bot_token": "test_token",
            "chat_id": "12345",
        }

        result = send_telegram("", config)
        assert result is True

    @patch("requests.post")
    def test_send_telegram_config_overrides_env(self, mock_post):
        """Test config values override environment variables."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = {
            "bot_token": "config_token",
            "chat_id": "config_chat",
        }

        with patch.dict(
            os.environ, {"TELEGRAM_BOT_TOKEN": "env_token", "TELEGRAM_CHAT_ID": "env_chat"}
        ):
            send_telegram("Test", config)

            # Should use config values
            call_args = mock_post.call_args
            assert "config_token" in call_args[0][0]
