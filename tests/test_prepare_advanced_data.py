"""
Tests for prepare_advanced_data module.

Verifies CSV handling and data preparation logic.
"""

import prepare_advanced_data


def test_prepare_advanced_data_handles_single_column_rows(tmp_path, monkeypatch):
    """Test that main() handles single-column CSV rows without error."""
    csv_path = tmp_path / "fear_greed.csv"
    csv_path.write_text("timestamp\n123\n456\n", encoding="utf-8")

    monkeypatch.setattr(prepare_advanced_data, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(prepare_advanced_data, "OUT_PATH", str(csv_path))
    monkeypatch.setattr(prepare_advanced_data, "fetch_fng", lambda limit=0: [])

    # Execute main - should complete without raising exceptions
    prepare_advanced_data.main()

    # Assert: output file was created/updated successfully
    assert csv_path.exists(), "Output CSV file should exist after main() execution"

    # Assert: file is readable and has expected structure
    content = csv_path.read_text(encoding="utf-8")
    assert "timestamp" in content, "Output file should contain timestamp column"
