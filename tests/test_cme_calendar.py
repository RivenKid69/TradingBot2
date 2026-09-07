# -*- coding: utf-8 -*-
"""
tests/test_cme_calendar.py
Comprehensive tests for CME trading calendar.

Tests cover:
- CMETradingCalendar: Trading hours and holidays
- Session detection
- Next open/close calculations
- Holiday calendar

Target: 30+ tests per Phase 3B specification.
"""

import pytest
from datetime import date, datetime, time, timedelta

# Import calendar modules
from services.cme_calendar import (
    CMETradingCalendar,
    CMESession,
    TradingHoursInfo,
    DaySchedule,
    CME_HOLIDAYS,
    CME_EARLY_CLOSE,
    create_cme_calendar,
    is_cme_trading_now,
    get_next_cme_open,
    get_next_cme_close,
)


# =============================================================================
# CMESession Tests
# =============================================================================


class TestCMESession:
    """Test CMESession enum."""

    def test_session_values(self):
        """Test session enum values."""
        assert CMESession.REGULAR.value == "regular"
        assert CMESession.MAINTENANCE.value == "maintenance"
        assert CMESession.CLOSED.value == "closed"

    def test_session_is_string_enum(self):
        """Test session is a string enum."""
        assert isinstance(CMESession.REGULAR.value, str)


# =============================================================================
# CME Holiday Calendar Tests
# =============================================================================


class TestCMEHolidays:
    """Test CME holiday calendar."""

    def test_holidays_set_not_empty(self):
        """Test holidays set is not empty."""
        assert len(CME_HOLIDAYS) > 0

    def test_holidays_format(self):
        """Test holidays are in ISO format."""
        for holiday in CME_HOLIDAYS:
            # Should be parseable as date
            try:
                parsed = date.fromisoformat(holiday)
                assert parsed is not None
            except ValueError:
                pytest.fail(f"Invalid date format: {holiday}")

    def test_major_holidays_present_2024(self):
        """Test major 2024 holidays are present."""
        expected_2024 = [
            "2024-01-01",  # New Year
            "2024-07-04",  # Independence Day
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas
        ]
        for holiday in expected_2024:
            assert holiday in CME_HOLIDAYS

    def test_major_holidays_present_2025(self):
        """Test major 2025 holidays are present."""
        expected_2025 = [
            "2025-01-01",  # New Year
            "2025-07-04",  # Independence Day
            "2025-11-27",  # Thanksgiving
            "2025-12-25",  # Christmas
        ]
        for holiday in expected_2025:
            assert holiday in CME_HOLIDAYS

    def test_early_close_days_exist(self):
        """Test early close days are defined."""
        assert len(CME_EARLY_CLOSE) > 0

    def test_early_close_format(self):
        """Test early close times are valid."""
        for date_str, close_time in CME_EARLY_CLOSE.items():
            # Date should be parseable
            try:
                parsed = date.fromisoformat(date_str)
                assert parsed is not None
            except ValueError:
                pytest.fail(f"Invalid date format: {date_str}")

            # Time should be a time object
            assert isinstance(close_time, time)


# =============================================================================
# CMETradingCalendar Tests
# =============================================================================


class TestCMETradingCalendar:
    """Test CME trading calendar functionality."""

    def test_init_default_timezone(self):
        """Test calendar initializes with default timezone."""
        calendar = CMETradingCalendar()
        assert calendar._timezone_name == "US/Eastern"

    def test_globex_hours_constants(self):
        """Test Globex hours constants."""
        calendar = CMETradingCalendar()

        # Sunday open
        assert calendar.GLOBEX_SUNDAY_OPEN == time(18, 0)

        # Friday close
        assert calendar.GLOBEX_FRIDAY_CLOSE == time(17, 0)

    def test_maintenance_window_constants(self):
        """Test maintenance window constants."""
        calendar = CMETradingCalendar()

        # Maintenance starts at 4:15 PM ET
        assert calendar.MAINTENANCE_START == time(16, 15)

        # Maintenance ends at 4:30 PM ET
        assert calendar.MAINTENANCE_END == time(16, 30)

    def test_is_trading_hours_weekday_morning(self):
        """Test is_trading_hours for weekday morning."""
        calendar = CMETradingCalendar()

        # Tuesday 10:00 AM ET = 15:00 UTC
        dt = datetime(2024, 6, 18, 15, 0)  # UTC
        assert calendar.is_trading_hours(dt) is True

    def test_is_trading_hours_weekday_afternoon(self):
        """Test is_trading_hours for weekday afternoon."""
        calendar = CMETradingCalendar()

        # Tuesday 2:00 PM ET = 19:00 UTC
        dt = datetime(2024, 6, 18, 19, 0)  # UTC
        assert calendar.is_trading_hours(dt) is True

    def test_is_trading_hours_maintenance_window(self):
        """Test is_trading_hours during maintenance."""
        calendar = CMETradingCalendar()

        # Tuesday 4:20 PM ET = 21:20 UTC (in maintenance window)
        dt = datetime(2024, 6, 18, 21, 20)  # UTC
        assert calendar.is_trading_hours(dt, check_maintenance=True) is False

    def test_is_trading_hours_friday_before_close(self):
        """Test is_trading_hours Friday before close."""
        calendar = CMETradingCalendar()

        # Friday 4:00 PM ET = 21:00 UTC
        dt = datetime(2024, 6, 21, 21, 0)  # UTC (Friday)
        assert calendar.is_trading_hours(dt) is True

    def test_is_trading_hours_friday_after_close(self):
        """Test is_trading_hours Friday after close."""
        calendar = CMETradingCalendar()

        # Friday 5:30 PM ET = 22:30 UTC
        dt = datetime(2024, 6, 21, 22, 30)  # UTC (Friday)
        assert calendar.is_trading_hours(dt) is False

    def test_is_trading_hours_saturday(self):
        """Test is_trading_hours on Saturday."""
        calendar = CMETradingCalendar()

        # Saturday 12:00 PM ET = 17:00 UTC
        dt = datetime(2024, 6, 22, 17, 0)  # UTC (Saturday)
        assert calendar.is_trading_hours(dt) is False

    def test_is_trading_hours_sunday_before_open(self):
        """Test is_trading_hours Sunday before open."""
        calendar = CMETradingCalendar()

        # Sunday 3:00 PM ET = 20:00 UTC (before 6pm open)
        dt = datetime(2024, 6, 23, 20, 0)  # UTC (Sunday)
        assert calendar.is_trading_hours(dt) is False

    def test_is_trading_hours_sunday_after_open(self):
        """Test is_trading_hours Sunday after open."""
        calendar = CMETradingCalendar()

        # Sunday 7:00 PM ET = 00:00 UTC Monday (after 6pm open)
        # 23:30 UTC Sunday = 6:30 PM ET Sunday
        dt = datetime(2024, 6, 23, 23, 30)  # UTC (Sunday)
        assert calendar.is_trading_hours(dt) is True

    def test_is_trading_hours_holiday(self):
        """Test is_trading_hours on holiday."""
        calendar = CMETradingCalendar()

        # July 4, 2024 (Thursday) - Independence Day
        dt = datetime(2024, 7, 4, 15, 0)  # UTC
        assert calendar.is_trading_hours(dt, check_holidays=True) is False

    def test_is_holiday(self):
        """Test is_holiday for known holidays."""
        calendar = CMETradingCalendar()

        # Known holidays
        assert calendar.is_holiday(date(2024, 1, 1)) is True  # New Year
        assert calendar.is_holiday(date(2024, 7, 4)) is True  # Independence Day
        assert calendar.is_holiday(date(2024, 12, 25)) is True  # Christmas

        # Regular days
        assert calendar.is_holiday(date(2024, 6, 18)) is False

    def test_is_early_close(self):
        """Test is_early_close for known early close days."""
        calendar = CMETradingCalendar()

        # Day before Thanksgiving 2024
        assert calendar.is_early_close(date(2024, 11, 27)) is True

        # Regular day
        assert calendar.is_early_close(date(2024, 6, 18)) is False

    def test_get_session_regular(self):
        """Test get_session during regular hours."""
        calendar = CMETradingCalendar()

        # Tuesday 10:00 AM ET = 15:00 UTC
        dt = datetime(2024, 6, 18, 15, 0)
        session = calendar.get_session(dt)
        assert session == CMESession.REGULAR

    def test_get_session_maintenance(self):
        """Test get_session during maintenance."""
        calendar = CMETradingCalendar()

        # Tuesday 4:20 PM ET = 21:20 UTC
        dt = datetime(2024, 6, 18, 21, 20)
        session = calendar.get_session(dt)
        assert session == CMESession.MAINTENANCE

    def test_get_session_closed_weekend(self):
        """Test get_session during weekend."""
        calendar = CMETradingCalendar()

        # Saturday
        dt = datetime(2024, 6, 22, 15, 0)
        session = calendar.get_session(dt)
        assert session == CMESession.CLOSED

    def test_get_session_closed_holiday(self):
        """Test get_session during holiday."""
        calendar = CMETradingCalendar()

        # July 4, 2024
        dt = datetime(2024, 7, 4, 15, 0)
        session = calendar.get_session(dt)
        assert session == CMESession.CLOSED

    def test_get_trading_hours_info_regular(self):
        """Test get_trading_hours_info during regular session."""
        calendar = CMETradingCalendar()

        dt = datetime(2024, 6, 18, 15, 0)  # Tuesday
        info = calendar.get_trading_hours_info(dt)

        assert info.session == CMESession.REGULAR
        assert info.is_trading is True
        assert info.notes == ""

    def test_get_trading_hours_info_maintenance(self):
        """Test get_trading_hours_info during maintenance."""
        calendar = CMETradingCalendar()

        dt = datetime(2024, 6, 18, 21, 20)  # 4:20 PM ET
        info = calendar.get_trading_hours_info(dt)

        assert info.session == CMESession.MAINTENANCE
        assert info.is_trading is False
        assert "maintenance" in info.notes.lower()

    def test_get_next_open_during_maintenance(self):
        """Test get_next_open during maintenance window."""
        calendar = CMETradingCalendar()

        # 4:20 PM ET = 21:20 UTC
        dt = datetime(2024, 6, 18, 21, 20)
        next_open = calendar.get_next_open(dt)

        # Should be 4:30 PM ET same day
        assert next_open > dt

    def test_get_next_open_friday_after_close(self):
        """Test get_next_open Friday after close."""
        calendar = CMETradingCalendar()

        # Friday 6:00 PM ET = 23:00 UTC
        dt = datetime(2024, 6, 21, 23, 0)
        next_open = calendar.get_next_open(dt)

        # Should be Sunday 6:00 PM ET
        assert next_open > dt

    def test_get_next_open_when_trading(self):
        """Test get_next_open when already trading."""
        calendar = CMETradingCalendar()

        # Tuesday 10:00 AM ET
        dt = datetime(2024, 6, 18, 15, 0)
        next_open = calendar.get_next_open(dt)

        # Should return current time (already open)
        assert next_open == dt

    def test_get_next_close_during_regular(self):
        """Test get_next_close during regular session."""
        calendar = CMETradingCalendar()

        # Tuesday 10:00 AM ET = 15:00 UTC
        dt = datetime(2024, 6, 18, 15, 0)
        next_close = calendar.get_next_close(dt)

        # Should be maintenance window same day (4:15 PM ET)
        assert next_close > dt

    def test_get_day_schedule_weekday(self):
        """Test get_day_schedule for regular weekday."""
        calendar = CMETradingCalendar()

        schedule = calendar.get_day_schedule(date(2024, 6, 18))  # Tuesday

        assert schedule.is_holiday is False
        assert schedule.maintenance_start == time(16, 15)
        assert schedule.maintenance_end == time(16, 30)

    def test_get_day_schedule_friday(self):
        """Test get_day_schedule for Friday."""
        calendar = CMETradingCalendar()

        schedule = calendar.get_day_schedule(date(2024, 6, 21))  # Friday

        assert schedule.is_holiday is False
        assert schedule.close_time == time(17, 0)  # 5:00 PM

    def test_get_day_schedule_saturday(self):
        """Test get_day_schedule for Saturday."""
        calendar = CMETradingCalendar()

        schedule = calendar.get_day_schedule(date(2024, 6, 22))

        assert schedule.open_time is None
        assert schedule.close_time is None

    def test_get_day_schedule_sunday(self):
        """Test get_day_schedule for Sunday."""
        calendar = CMETradingCalendar()

        schedule = calendar.get_day_schedule(date(2024, 6, 23))

        assert schedule.open_time == time(18, 0)  # 6:00 PM

    def test_get_day_schedule_holiday(self):
        """Test get_day_schedule for holiday."""
        calendar = CMETradingCalendar()

        schedule = calendar.get_day_schedule(date(2024, 7, 4))

        assert schedule.is_holiday is True
        assert schedule.open_time is None

    def test_get_holidays_in_range(self):
        """Test get_holidays_in_range."""
        calendar = CMETradingCalendar()

        holidays = calendar.get_holidays_in_range(
            date(2024, 6, 1),
            date(2024, 7, 31),
        )

        # Should include Juneteenth and July 4
        holiday_strs = [h.isoformat() for h in holidays]
        assert "2024-06-19" in holiday_strs
        assert "2024-07-04" in holiday_strs

    def test_count_trading_days(self):
        """Test count_trading_days."""
        calendar = CMETradingCalendar()

        # One week (Mon-Fri) but June 19 is Juneteenth (holiday)
        count = calendar.count_trading_days(
            date(2024, 6, 17),  # Monday
            date(2024, 6, 21),  # Friday
        )

        # 5 weekdays minus 1 Juneteenth holiday = 4
        assert count == 4

    def test_count_trading_days_with_holiday(self):
        """Test count_trading_days with holiday."""
        calendar = CMETradingCalendar()

        # Week with July 4 (Thursday)
        count = calendar.count_trading_days(
            date(2024, 7, 1),  # Monday
            date(2024, 7, 5),  # Friday
        )

        # 5 weekdays minus 1 holiday = 4
        assert count == 4

    def test_add_trading_days(self):
        """Test add_trading_days."""
        calendar = CMETradingCalendar()

        # Add 5 trading days to Monday (June 19 Juneteenth is a holiday)
        result = calendar.add_trading_days(
            date(2024, 6, 17),  # Monday
            5,
        )

        # June 18 (1), June 20 (2), June 21 (3), June 24 (4), June 25 (5)
        # Note: June 19 Juneteenth is skipped
        assert result == date(2024, 6, 25)

    def test_add_trading_days_with_holiday(self):
        """Test add_trading_days crossing holiday."""
        calendar = CMETradingCalendar()

        # Add 3 trading days before July 4 (Thursday)
        result = calendar.add_trading_days(
            date(2024, 7, 1),  # Monday
            3,
        )

        # Should be Friday (skipping Thursday holiday)
        assert result == date(2024, 7, 5)


class TestCMECalendarConvenienceFunctions:
    """Test convenience functions."""

    def test_create_cme_calendar(self):
        """Test create_cme_calendar factory."""
        calendar = create_cme_calendar()
        assert isinstance(calendar, CMETradingCalendar)

    def test_is_cme_trading_now(self):
        """Test is_cme_trading_now returns boolean."""
        result = is_cme_trading_now()
        assert isinstance(result, bool)

    def test_get_next_cme_open(self):
        """Test get_next_cme_open returns datetime."""
        result = get_next_cme_open()
        assert isinstance(result, datetime)

    def test_get_next_cme_close(self):
        """Test get_next_cme_close returns datetime."""
        result = get_next_cme_close()
        assert isinstance(result, datetime)


class TestCMECalendarEdgeCases:
    """Edge case tests for CME calendar."""

    def test_daylight_saving_time_note(self):
        """Test DST handling (note: simplified implementation)."""
        # The current implementation doesn't handle DST
        # This test documents the limitation
        calendar = CMETradingCalendar()

        # During EST (winter) and EDT (summer), times differ by 1 hour
        # The simplified implementation uses fixed -5 offset
        # Production should use pytz for proper handling

    def test_maintenance_window_boundary_start(self):
        """Test exact maintenance start time."""
        calendar = CMETradingCalendar()

        # Exactly 4:15 PM ET = 21:15 UTC
        dt = datetime(2024, 6, 18, 21, 15)
        assert calendar.is_trading_hours(dt, check_maintenance=True) is False

    def test_maintenance_window_boundary_end(self):
        """Test exactly at maintenance end."""
        calendar = CMETradingCalendar()

        # Exactly 4:30 PM ET = 21:30 UTC
        dt = datetime(2024, 6, 18, 21, 30)
        assert calendar.is_trading_hours(dt, check_maintenance=True) is True

    def test_friday_close_boundary(self):
        """Test exact Friday close time."""
        calendar = CMETradingCalendar()

        # Exactly 5:00 PM ET = 22:00 UTC
        dt = datetime(2024, 6, 21, 22, 0)  # Friday
        assert calendar.is_trading_hours(dt) is False

    def test_sunday_open_boundary(self):
        """Test exact Sunday open time."""
        calendar = CMETradingCalendar()

        # Exactly 6:00 PM ET Sunday = 23:00 UTC
        dt = datetime(2024, 6, 23, 23, 0)  # Sunday
        assert calendar.is_trading_hours(dt) is True

    def test_year_boundary_holidays(self):
        """Test holidays at year boundary."""
        calendar = CMETradingCalendar()

        # New Year's Day
        assert calendar.is_holiday(date(2024, 1, 1)) is True
        assert calendar.is_holiday(date(2025, 1, 1)) is True

        # December 31 (early close, not holiday)
        assert calendar.is_holiday(date(2024, 12, 31)) is False
        assert calendar.is_early_close(date(2024, 12, 31)) is True

    def test_holidays_multiple_years(self):
        """Test holidays span 2024-2026."""
        calendar = CMETradingCalendar()

        # Check each year has holidays
        for year in [2024, 2025, 2026]:
            holidays_in_year = calendar.get_holidays_in_range(
                date(year, 1, 1),
                date(year, 12, 31),
            )
            assert len(holidays_in_year) >= 5  # At least 5 holidays per year


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
