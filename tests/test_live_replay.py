"""Live monitoring replay + walk-forward retrain simulation invariants."""

from src.monitoring.live_replay import compute_rolling_metrics, run_retrain_simulation


def test_live_rolling_metrics_match_designed_march_trigger():
    summary, baseline = compute_rolling_metrics()
    assert baseline > 0
    assert list(summary["month"]) == [
        "2010-01", "2010-02", "2010-03", "2010-04",
        "2010-05", "2010-06", "2010-07",
    ]
    assert "mae" in summary.columns
    assert bool(summary.loc[summary["month"] == "2010-03", "retrain"].iloc[0])
    assert not bool(summary.loc[summary["month"] == "2010-01", "retrain"].iloc[0])


def test_retrain_simulation_scores_only_decision_month():
    # End of April 2010: train through March, score only on April
    result = run_retrain_simulation(decision_month="2010-04")
    assert result["status"] == "ok"
    assert result["decision_month"] == "2010-04"
    assert result["eval_month"] == "2010-04"
    assert result["labeled_through"] == "2010-03"
    assert result["n_new_labels"] > 0
    assert result["n_eval"] > 0
    assert result["champion"]["n"] == result["n_eval"]
    assert result["challenger"]["n"] == result["n_eval"]
    if result["promote"]:
        assert result["challenger"]["rmse"] < result["champion"]["rmse"]
        assert "PROMOTE" in result["decision"]
    else:
        assert result["challenger"]["rmse"] >= result["champion"]["rmse"]
        assert "KEEP" in result["decision"]


def test_retrain_default_is_month_after_first_signal():
    result = run_retrain_simulation()  # no decision_month → after March signal
    assert result["status"] == "ok"
    assert result["decision_month"] == "2010-04"
    assert result["labeled_through"] == "2010-03"
