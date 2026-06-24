"""CLI smoke tests — the offline synthetic demo must run end-to-end."""

from __future__ import annotations

from trust_probe.cli import main


def test_synthetic_eval_runs(capsys):
    rc = main(["eval", "--synthetic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fusion (WB + BB)" in out
    assert "FUSION WINS" in out


def test_no_command_prints_help():
    assert main([]) == 0


def test_eval_without_target_errors():
    assert main(["eval"]) == 1
