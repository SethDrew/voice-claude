"""Pytest configuration — register custom markers."""

import os
import sys

import pytest

# Ensure src/ is on sys.path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests as integration tests (deselect with '-k not integration')")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-k not slow')")
    config.addinivalue_line("markers", "stress: marks tests as stress tests (deselect with '-k not stress')")
    config.addinivalue_line("markers", "asyncio: marks tests as async (requires pytest-asyncio)")
