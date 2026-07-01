from __future__ import annotations

from typer.testing import CliRunner

from wq_agent.cli import app


def test_init_copies_public_resources(tmp_path):
    target = tmp_path / "workspace"
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(target)])

    assert result.exit_code == 0, result.output
    wiki_dir = target / "wiki"
    assert wiki_dir.is_dir()
    assert list(wiki_dir.iterdir()) == []
    assert (target / "templates" / "alpha_templates.yaml").exists()
    assert (target / ".env.example").exists()


def test_init_skips_existing_template_files_without_overwrite(tmp_path):
    target = tmp_path / "workspace"
    existing = target / "templates" / "alpha_templates.yaml"
    existing.parent.mkdir(parents=True)
    existing.write_text("custom", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(target)])

    assert result.exit_code == 0, result.output
    assert existing.read_text(encoding="utf-8") == "custom"
    assert "skipped" in result.output


def test_autosubmit_commands_are_registered():
    runner = CliRunner()

    result = runner.invoke(app, ["autosubmit", "--help"])

    assert result.exit_code == 0, result.output
    assert "once" in result.output
    assert "daemon" in result.output


def test_autosubmit_daemon_requires_explicit_enable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["autosubmit", "daemon"])

    assert result.exit_code == 1
    assert "disabled" in result.output
