from __future__ import annotations

import importlib


def test_legacy_package_name_resolves_submodules():
    legacy = importlib.import_module("wq_agent.db")
    current = importlib.import_module("alphagen_agent.db")

    assert legacy.Database.__name__ == current.Database.__name__ == "Database"


def test_project_exposes_new_brand_version():
    current = importlib.import_module("alphagen_agent")
    legacy = importlib.import_module("wq_agent")

    assert current.__version__ == legacy.__version__
