"""Pytest configuration and global fixtures for NetForecaster AI."""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
import pytest

# Ensure all temporary file operations stay strictly within the repository workspace
WORKSPACE_TEMP = Path(__file__).resolve().parent.parent / ".pytest_temp"
WORKSPACE_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(WORKSPACE_TEMP)
os.environ["TMP"] = str(WORKSPACE_TEMP)
tempfile.tempdir = str(WORKSPACE_TEMP)


@pytest.fixture
def tmp_path(monkeypatch) -> Path:
    """Fixture to provide an isolated temporary directory within the workspace."""
    test_dir = WORKSPACE_TEMP / f"test_{uuid.uuid4().hex[:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    try:
        shutil.rmtree(test_dir, ignore_errors=True)
    except Exception:
        pass
