"""Tests for src/components/registry.py: registering and building components by name."""

import pytest

from src.components.registry import Registry, component_spec


class Base:
    name = "base"

    def settings(self):
        return {}


class Plain(Base):
    name = "plain"


class WithSettings(Base):
    name = "with_settings"

    def __init__(self, alpha=1.0, beta="x"):
        self.alpha, self.beta = alpha, beta

    def settings(self):
        return {"alpha": self.alpha, "beta": self.beta}


@pytest.fixture
def reg():
    r = Registry("widget", Base)
    r.register(Plain)
    r.register(WithSettings)
    return r


def test_registry_is_a_mapping_of_names_to_classes(reg):
    assert sorted(reg) == ["plain", "with_settings"]
    assert reg["plain"] is Plain and "plain" in reg and len(reg) == 2


def test_create_from_instance_name_and_settings(reg):
    inst = WithSettings(alpha=3)
    assert reg.create(inst) is inst
    assert type(reg.create("plain")) is Plain
    built = reg.create(("with_settings", {"alpha": 2.0}))
    assert (built.alpha, built.beta) == (2.0, "x")


def test_defaults_only_reach_constructors_that_accept_them(reg):
    # Plain() takes no arguments; the default must be dropped, not raise.
    assert type(reg.create("plain", alpha=5.0)) is Plain
    assert reg.create("with_settings", alpha=5.0).alpha == 5.0
    # Explicit settings win over defaults.
    assert reg.create(("with_settings", {"alpha": 1.5}), alpha=5.0).alpha == 1.5


def test_component_spec_round_trips(reg):
    spec = component_spec(WithSettings(alpha=0.25, beta="y"))
    assert spec == {"name": "with_settings", "settings": {"alpha": 0.25, "beta": "y"}}
    rebuilt = reg.create((spec["name"], spec["settings"]))
    assert rebuilt.settings() == spec["settings"]


def test_unknown_name_and_bad_specs(reg):
    with pytest.raises(ValueError, match="Unknown widget 'nope'"):
        reg.create("nope")
    with pytest.raises(TypeError, match="widget spec"):
        reg.create(42)
    with pytest.raises(TypeError, match="must subclass Base"):
        reg["bad"] = int


def test_register_rejects_a_different_class_under_a_taken_name(reg):
    class Other(Base):
        name = "plain"

    with pytest.raises(ValueError, match="already registered"):
        reg.register(Other)
    reg.register(Plain)  # re-registering the same class is fine
    reg.register(Other, name="other")
    assert reg["other"] is Other


def test_register_needs_a_name(reg):
    class Nameless(Base):
        name = ""

    with pytest.raises(ValueError, match="non-empty string"):
        reg.register(Nameless)
