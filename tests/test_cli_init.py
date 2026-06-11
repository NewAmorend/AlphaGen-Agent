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
