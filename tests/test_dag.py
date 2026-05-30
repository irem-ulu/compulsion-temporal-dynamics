"""Tests for the DAG renderer's drift-detection warning."""

import warnings

from ctd.data import generate as gen
from ctd.viz import dag


def test_render_dag_quiet_for_current_DGP(tmp_path):
    """If LAYOUT is in sync with CAUSAL_DAG, render_dag should not warn."""
    out = tmp_path / "dag.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dag.render_dag(out)
    relevant = [w for w in caught if "CAUSAL_DAG references" in str(w.message)]
    assert relevant == []
    assert out.exists()


def test_render_dag_warns_on_missing_layout_node(tmp_path, monkeypatch):
    """A new edge mentioning a node not in LAYOUT should produce a warning
    naming that node — the whole point of the recent change."""
    extended = list(gen.CAUSAL_DAG) + [("GhostNode", "Stress", "+")]
    monkeypatch.setattr(dag, "CAUSAL_DAG", extended)

    out = tmp_path / "dag.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dag.render_dag(out)
    relevant = [w for w in caught if "CAUSAL_DAG references" in str(w.message)]
    assert len(relevant) == 1
    assert "GhostNode" in str(relevant[0].message)
