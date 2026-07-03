from __future__ import annotations

import importlib

from alphagen_agent.config import Settings


def test_legacy_package_name_resolves_submodules():
    legacy = importlib.import_module("wq_agent.db")
    current = importlib.import_module("alphagen_agent.db")

    assert legacy.Database.__name__ == current.Database.__name__ == "Database"


def test_project_exposes_new_brand_version():
    current = importlib.import_module("alphagen_agent")
    legacy = importlib.import_module("wq_agent")

    assert current.__version__ == legacy.__version__


def test_settings_reuse_legacy_database_when_new_name_is_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DB_PATH", raising=False)
    (tmp_path / "wq_agent.db").touch()

    assert Settings(_env_file=None).DB_PATH == "./wq_agent.db"


def test_settings_prefer_new_database_name_when_both_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DB_PATH", raising=False)
    (tmp_path / "wq_agent.db").touch()
    (tmp_path / "alphagen_agent.db").touch()

    assert Settings(_env_file=None).DB_PATH == "./alphagen_agent.db"
