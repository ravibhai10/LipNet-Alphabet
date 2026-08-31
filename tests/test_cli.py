"""
Tests for CLI entry point.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

from alphabet.cli import main, validate_cmd, inspect_cmd, ingest_cmd


class TestCLI:
    """Test CLI commands."""

    def test_validate_cmd(self):
        """Test validate command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()

            # Create minimal manifest
            manifest_path = dataset_root / "manifest.jsonl"
            manifest_path.write_text("")

            args = Mock()
            args.dataset_root = str(dataset_root)

            # Should not crash (will fail validation but not error)
            with patch('data.validate.validate_dataset', return_value=1):
                result = validate_cmd(args)
                assert result == 1

    def test_inspect_cmd(self):
        """Test inspect command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()

            args = Mock()
            args.dataset_root = str(dataset_root)

            with patch('data.inspect.inspect_dataset', return_value=0):
                result = inspect_cmd(args)
                assert result == 0

    def test_main_help(self):
        """Test main shows help."""
        with patch.object(sys, 'argv', ['alphabet', '--help']):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

    def test_main_validate(self):
        """Test main with validate-dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()

            with patch.object(sys, 'argv', ['alphabet', 'validate-dataset', str(dataset_root)]):
                with patch('alphabet.cli.validate_cmd', return_value=0) as mock_validate:
                    result = main()
                    assert result == 0
                    mock_validate.assert_called_once()

    def test_main_inspect(self):
        """Test main with inspect-dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()

            with patch.object(sys, 'argv', ['alphabet', 'inspect-dataset', str(dataset_root)]):
                with patch('alphabet.cli.inspect_cmd', return_value=0) as mock_inspect:
                    result = main()
                    assert result == 0
                    mock_inspect.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])