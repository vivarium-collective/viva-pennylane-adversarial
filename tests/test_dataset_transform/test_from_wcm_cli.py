"""Tests for the `adversarial datasets from-wcm` CLI command."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import re

from typer.testing import CliRunner

from app.cli import cli

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI colour codes so option names aren't split by escapes."""
    return _ANSI.sub("", text)


def _hive_history() -> Path:
    root = Path(tempfile.mkdtemp()) / "history"
    for agent in ("00", "01"):
        d = root / f"experiment_id=exp1/variant=0/lineage_seed=0/generation=0/agent_id={agent}"
        d.mkdir(parents=True)
        pl.DataFrame({
            "global_time": [0.0, 1.0, 2.0, 3.0],
            "listeners__mass__cell_mass": [500.0, 510.0, 520.0, 530.0],
            "listeners__replication_data__number_of_oric": [1, 1, 2, 2],
        }).write_parquet(d / "100.pq")
    return root


def test_help():
    # Force a wide terminal: rich/typer truncate the options panel at the
    # default 80 cols in CI, dropping long option names. Strip ANSI too, since
    # rich wraps each flag in colour escapes that split "--feature-cols".
    result = runner.invoke(
        cli, ["datasets", "from-wcm", "--help"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--feature-cols" in out
    assert "--target-col" in out


def test_no_target_col_lists_candidates():
    history = _hive_history()
    result = runner.invoke(cli, [
        "datasets", "from-wcm", str(history),
        "-f", "listeners__mass__cell_mass",
    ])
    assert result.exit_code == 1
    assert "listeners__replication_data__number_of_oric" in result.output


def test_end_to_end_writes_formatted_dataset():
    history = _hive_history()
    out = Path(tempfile.mkdtemp()) / "wcm_out"
    result = runner.invoke(cli, [
        "datasets", "from-wcm", str(history),
        "-f", "listeners__mass__cell_mass",
        "-t", "listeners__replication_data__number_of_oric",
        "--train-ratio", "0.5",
        "--output-format", "json",
        "-o", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert (out.with_suffix(".json")).exists()
