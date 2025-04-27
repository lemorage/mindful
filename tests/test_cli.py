import logging
from pathlib import Path
import threading

import pytest
from pytest_mock import MockerFixture

from mindful.cli import get_mindful_config_dir


def test_get_mindful_config_dir_creates_directory(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test that get_mindful_config_dir creates the .mindful directory if it doesn't exist."""
    # Arrange
    mock_home = tmp_path / "home"
    mock_home.mkdir()
    mocker.patch("pathlib.Path.home", return_value=mock_home)
    expected_dir = mock_home / ".mindful"
    spy_logger = mocker.spy(logging.getLogger("mindful"), "debug")

    # Act
    result = get_mindful_config_dir()

    # Assert
    assert result == expected_dir
    assert expected_dir.exists()
    assert spy_logger.call_count == 1
    assert "Mindful config directory ensured at" in spy_logger.call_args[0][0]


def test_get_mindful_config_dir_existing_directory(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test that get_mindful_config_dir handles an existing .mindful directory."""
    # Arrange
    mock_home = tmp_path / "home"
    mock_home.mkdir()
    expected_dir = mock_home / ".mindful"
    expected_dir.mkdir()
    mocker.patch("pathlib.Path.home", return_value=mock_home)
    spy_logger = mocker.spy(logging.getLogger("mindful"), "debug")

    # Act
    result = get_mindful_config_dir()

    # Assert
    assert result == expected_dir
    assert expected_dir.exists()
    assert spy_logger.call_count == 1
    assert "Mindful config directory ensured at" in spy_logger.call_args[0][0]


def test_get_mindful_config_dir_permission_error(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test that get_mindful_config_dir raises RuntimeError on permission error."""
    # Arrange
    mock_home = tmp_path / "home"
    mock_home.mkdir()
    mocker.patch("pathlib.Path.home", return_value=mock_home)
    mocker.patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied"))
    spy_logger = mocker.spy(logging.getLogger("mindful"), "error")

    # Act / Assert
    with pytest.raises(RuntimeError, match="Cannot create mindful config directory"):
        get_mindful_config_dir()
    assert spy_logger.call_count == 1
    assert "Failed to create config directory" in spy_logger.call_args[0][0]


def test_get_mindful_config_dir_thread_safety(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test that get_mindful_config_dir is thread-safe under concurrent calls."""
    # Arrange
    mock_home = tmp_path / "home"
    mock_home.mkdir()
    mocker.patch("pathlib.Path.home", return_value=mock_home)
    expected_dir = mock_home / ".mindful"
    results = []

    def call_get_mindful_config_dir() -> None:
        results.append(get_mindful_config_dir())

    # Act
    threads = [threading.Thread(target=call_get_mindful_config_dir) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert
    assert all(result == expected_dir for result in results)
    assert expected_dir.exists()
    assert len(results) == 10
