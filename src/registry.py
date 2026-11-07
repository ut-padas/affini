"""
Registries of named, interchangeable components.

Each kind of component the refinement loop can swap (local model, split strategy,
sampler, acceptance criterion, trust region) has one ``Registry``. A registry maps a
name to a class, so a component can be chosen by name from the configuration, the
launcher or a saved model, and an extension only has to register its class::

    from src.sampling import SAMPLERS, Sampler

    @SAMPLERS.register
    class MySampler(Sampler):
        name = "my_sampler"
        def sample(self, clusters, n_new, rng): ...

``Registry.create`` turns a *component spec* into an instance. A spec is either an
instance (used as is), a registered name, or a ``(name, settings)`` pair whose
settings are passed to the constructor. Under MPI every rank builds its own
components from the same spec, so specs must describe the same component on every
rank.
"""

import inspect
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any, Generic, TypeVar

T = TypeVar("T")

ComponentSpec = Any  # instance | str | tuple[str, Mapping[str, Any]]


class Registry(MutableMapping, Generic[T]):
    """Name -> class mapping for one kind of component (a ``MutableMapping``)."""

    def __init__(self, kind: str, base: type):
        self.kind = kind  # e.g. "local model", used in error messages
        self.base = base
        self._classes: dict[str, type] = {}

    # --- Mapping interface -------------------------------------------------
    def __getitem__(self, name: str) -> type:
        return self._classes[name]

    def __setitem__(self, name: str, cls: type) -> None:
        if not (isinstance(cls, type) and issubclass(cls, self.base)):
            raise TypeError(f"A {self.kind} must subclass {self.base.__name__}; got {cls!r}.")
        self._classes[name] = cls

    def __delitem__(self, name: str) -> None:
        del self._classes[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._classes)

    def __len__(self) -> int:
        return len(self._classes)

    # --- Registration and construction -------------------------------------
    def register(self, cls: type | None = None, *, name: str | None = None):
        """
        Class decorator registering ``cls`` under ``name`` (default: ``cls.name``).
        Usable as ``@REGISTRY.register`` or ``@REGISTRY.register(name="...")``.
        Registering a different class under a taken name raises ``ValueError``.
        """

        def decorate(c: type) -> type:
            key = name or getattr(c, "name", None)
            if not isinstance(key, str) or not key:
                raise ValueError(f"A registered {self.kind} needs a non-empty string `name`.")
            if key in self._classes and self._classes[key] is not c:
                raise ValueError(
                    f"A different {self.kind} is already registered as {key!r}: "
                    f"{self._classes[key].__qualname__}."
                )
            self[key] = c
            return c

        return decorate(cls) if cls is not None else decorate

    def create(self, spec: ComponentSpec, **defaults: Any) -> T:
        """
        Return the component described by ``spec``.

        - an instance of the base class is returned unchanged;
        - a name builds ``REGISTRY[name](**defaults)``;
        - ``(name, settings)`` builds ``REGISTRY[name](**{**defaults, **settings})``.

        ``defaults`` are only used for names; they let a caller pass settings that
        come from elsewhere in the configuration (e.g. tolerances). Each default is
        passed only if the constructor accepts it; explicit ``settings`` always are.
        """
        if isinstance(spec, self.base):
            return spec
        settings: Mapping[str, Any] = {}
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], Mapping):
            spec, settings = spec
        if not isinstance(spec, str):
            raise TypeError(
                f"A {self.kind} spec must be a {self.base.__name__} instance, a "
                f"registered name or a (name, settings) pair; got {spec!r}."
            )
        if spec not in self._classes:
            raise ValueError(
                f"Unknown {self.kind} {spec!r}; choose one of {sorted(self._classes)}."
            )
        cls = self._classes[spec]
        return cls(**{**_accepted(cls, defaults), **settings})


def _accepted(cls: type, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of ``kwargs`` that ``cls(...)`` accepts as keyword arguments."""
    if not kwargs:
        return {}
    params = inspect.signature(cls).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return dict(kwargs)
    names = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return {k: v for k, v in kwargs.items() if k in names}


def component_spec(component: Any) -> dict[str, Any]:
    """
    A serializable description of a component, ``{"name": ..., "settings": ...}``,
    from its ``name`` attribute and ``settings()`` method (``{}`` if it has none).
    ``Registry.create((spec["name"], spec["settings"]))`` rebuilds it.
    """
    settings = component.settings() if hasattr(component, "settings") else {}
    return {"name": component.name, "settings": dict(settings)}
