"""Smoke test: verify all package modules are importable."""


def test_import_simulator():
    from simulator import models, registry, generators, telemetry
    assert models is not None
    assert registry is not None
    assert generators is not None
    assert telemetry is not None


def test_import_tools():
    from tools import inspection, memory, parser
    assert inspection is not None
    assert memory is not None
    assert parser is not None


def test_import_environment():
    from environment import env, reward
    assert env is not None
    assert reward is not None


def test_import_training():
    from training import prompts, scenarios, train_grpo
    assert prompts is not None
    assert scenarios is not None
    assert train_grpo is not None


def test_import_evaluation():
    from evaluation import metrics, baselines, ablation
    assert metrics is not None
    assert baselines is not None
    assert ablation is not None
