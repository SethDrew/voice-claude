"""Pytest configuration — register custom markers."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests as integration tests (deselect with '-k not integration')")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-k not slow')")
