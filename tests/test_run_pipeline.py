from src.run_pipeline import REQUIRED_ARTIFACTS, resolve_stages, validate_artifacts


def test_resolve_stages_all_includes_expected_order():
    stages = resolve_stages("all")
    assert stages[0] == "profile"
    assert stages.index("prepare-annotations") < stages.index("prepare-gold")
    assert stages.index("prepare-gold") < stages.index("train-complaint")
    assert stages.index("train-complaint") < stages.index("compare-gold")
    assert stages.index("train-complaint") < stages.index("suggest-annotations")
    assert stages.index("prepare-aspect-annotations") < stages.index("prepare-aspect-gold")
    assert stages.index("prepare-aspect-gold") < stages.index("train-aspect")
    assert stages.index("train-aspect") < stages.index("aspect-sentiment")
    assert stages.index("train-aspect") < stages.index("suggest-aspect-annotations")
    assert stages.index("aspect-sentiment") < stages.index("gap-scoring")
    assert "gap-scoring" in stages
    assert stages[-1] == "export"


def test_validate_existing_gap_scoring_artifacts():
    validate_artifacts("gap-scoring")
    assert "evaluation" in REQUIRED_ARTIFACTS
