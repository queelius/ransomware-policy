"""Test for the no_tools ablation variant."""


def test_no_tools_variant_exists():
    """Verify the no_tools ablation variant exists and has empty tool list."""
    from evaluation.ablation import ABLATION_VARIANTS
    assert "no_tools" in ABLATION_VARIANTS
    assert ABLATION_VARIANTS["no_tools"] == []
