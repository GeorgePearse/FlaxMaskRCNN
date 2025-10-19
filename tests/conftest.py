"""Pytest configuration for Dax tests."""

def pytest_configure(config):
    """Configure pytest before collecting tests."""
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'
