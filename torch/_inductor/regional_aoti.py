"""Experimental regional AOTInductor marking and discovery API."""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import Any, overload, TYPE_CHECKING, TypeVar

import torch.nn as nn
from torch._inductor import config as inductor_config
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable


_REGION_CONFIG_ATTR = "_torch_aoti_region_config"

__all__ = [
    "AOTIRegionDiscoveryError",
    "AOTIRegionError",
    "RegionConfig",
    "RegionResult",
    "mark_region",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AOTIRegionError(RuntimeError):
    """Base class for all regional AOTI errors."""


class AOTIRegionDiscoveryError(AOTIRegionError):
    """Raised when selected regions cannot be represented safely."""


# ---------------------------------------------------------------------------
# Config / info dataclasses
# ---------------------------------------------------------------------------
class _UnsetType(Enum):
    UNSET = "unset"

    def __repr__(self) -> str:
        return "_UNSET"


_UNSET = _UnsetType.UNSET


def _convert_containers(value: Any, dict_type: type, list_type: type) -> Any:
    """Copy nested dicts, lists and tuples; leaves (e.g. ``Dim`` objects) stay shared."""
    if isinstance(value, dict):
        return dict_type(
            {
                key: _convert_containers(item, dict_type, list_type)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return list_type(
            _convert_containers(item, dict_type, list_type) for item in value
        )
    if isinstance(value, tuple):
        items = [_convert_containers(item, dict_type, list_type) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    return value


def _freeze(value: Any) -> Any:
    """Copy nested dicts and lists into ``immutable_dict`` and ``immutable_list``."""
    return _convert_containers(value, immutable_dict, immutable_list)


def _thaw(value: Any) -> Any:
    """Copy a frozen ``RegionConfig`` value back into plain dicts and lists.

    Use it before handing ``dynamic_shapes`` to ``torch.export`` (which matches
    container types against the inputs) or ``inductor_configs`` to
    ``aoti_compile_and_package`` (which writes into the dict it receives).
    """
    return _convert_containers(value, dict, list)


def _unknown_inductor_config_keys(configs: dict[Any, Any]) -> list[Any]:
    """Keys that ``aoti_compile_and_package`` cannot apply as config patches."""
    unknown = []
    for key in configs:
        if key == "extern_node_serializer":
            continue
        if not isinstance(key, str):
            unknown.append(key)
            continue
        normalized = key.replace("-", "_")
        if normalized in inductor_config._config:  # type: ignore[attr-defined]
            continue
        from torch._inductor.codegen.common import (
            get_compile_option_owner,
            init_backend_registration,
        )

        init_backend_registration()
        if get_compile_option_owner(normalized) is None:
            unknown.append(key)
    return unknown


@dataclasses.dataclass(kw_only=True, frozen=True)
class RegionConfig:
    """Artifact-related configuration for one selected region.

    Every field is keyword-only and starts unset. When marks are merged, an
    unset field falls back to the class mark and then to its default; an
    explicit ``None`` is a value and overrides the class mark. Values are
    validated at construction. The config is frozen all the way down: dicts and
    lists in its values are stored as ``immutable_dict`` and ``immutable_list``
    copies (leaves such as ``Dim`` objects stay shared), so neither later
    changes to the caller's dicts nor edits through the config affect it, and
    the config is hashable when its leaves are. ``torch.export`` and
    ``aoti_compile_and_package`` need plain containers, so the compile path
    hands them plain copies made by ``_thaw``.

    Attributes:
        dynamic_shapes: Dynamic-shape specs keyed by ``forward`` argument name,
            as in the dict form of ``torch.export.export(dynamic_shapes=...)``.
            Defaults to ``None``.
        inductor_configs: Inductor config overrides passed to
            ``aoti_compile_and_package``. Keys are the names that function
            accepts: ``torch._inductor.config`` keys (``-`` or ``_``),
            device-namespaced backend options and ``extern_node_serializer``.
            Defaults to ``None``.
        rtol: Relative tolerance for the compiled-versus-eager accuracy check.
            Defaults to ``1e-3``.
        atol: Absolute tolerance for the compiled-versus-eager accuracy check.
            Defaults to ``1e-3``.
    """

    dynamic_shapes: dict[str, Any] | None | _UnsetType = _UNSET
    inductor_configs: dict[str, Any] | None | _UnsetType = _UNSET
    rtol: float | None | _UnsetType = _UNSET
    atol: float | None | _UnsetType = _UNSET

    def __post_init__(self) -> None:
        dynamic_shapes = self.dynamic_shapes
        if dynamic_shapes is not _UNSET and dynamic_shapes is not None:
            if not isinstance(dynamic_shapes, dict) or not all(
                isinstance(key, str) for key in dynamic_shapes
            ):
                raise TypeError(
                    "RegionConfig.dynamic_shapes must be a dict keyed by forward "
                    f"argument name, got {dynamic_shapes!r}"
                )
            object.__setattr__(self, "dynamic_shapes", _freeze(dynamic_shapes))
        inductor_configs = self.inductor_configs
        if inductor_configs is not _UNSET and inductor_configs is not None:
            if not isinstance(inductor_configs, dict):
                raise TypeError(
                    "RegionConfig.inductor_configs must be a dict, got "
                    f"{type(inductor_configs).__name__}"
                )
            unknown = _unknown_inductor_config_keys(inductor_configs)
            if unknown:
                raise ValueError(f"Unknown Inductor config keys: {unknown!r}")
            object.__setattr__(self, "inductor_configs", _freeze(inductor_configs))
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if value is _UNSET or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"RegionConfig.{name} must be a number, got {type(value).__name__}"
                )
            if math.isnan(value) or value < 0:
                raise ValueError(f"RegionConfig.{name} must be >= 0, got {value!r}")


@dataclasses.dataclass
class RegionResult:
    """One marked region and everything recorded about it.

    Attributes:
        name: Path of the module in the model (``""`` for the root).
        module: The marked ``nn.Module`` instance.
        config: Effective ``RegionConfig`` (class mark, instance mark, defaults).
        aliases: Every path at which the same module instance appears.
    """

    name: str
    module: nn.Module = dataclasses.field(repr=False)
    config: RegionConfig = dataclasses.field(repr=False)
    aliases: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Marking / discovery
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS = {
    "dynamic_shapes": None,
    "inductor_configs": None,
    "rtol": 1e-3,
    "atol": 1e-3,
}


def _validate_config(config: object) -> RegionConfig:
    if config is None:
        return RegionConfig()
    if not isinstance(config, RegionConfig):
        raise TypeError(f"config must be RegionConfig, got {type(config)!r}")
    return config


_Marked = TypeVar("_Marked", bound="nn.Module | type[nn.Module]")


@overload
def mark_region(
    module: _Marked, /, *, config: RegionConfig | None = None
) -> _Marked: ...


@overload
def mark_region(
    *, config: RegionConfig | None = None
) -> Callable[[_Marked], _Marked]: ...


def mark_region(module: Any = _UNSET, /, *, config: RegionConfig | None = None) -> Any:
    """Mark an ``nn.Module`` instance or an ``nn.Module`` class as a region.

    Call it directly, ``mark_region(module_or_class, config=...)``, or use it
    as a class decorator, bare (``@mark_region``) or with arguments
    (``@mark_region(config=...)``).

    Args:
        module: The ``nn.Module`` instance or subclass to mark.
        config: Settings for the region. Unset fields fall back to the class
            mark, then to the ``RegionConfig`` defaults.

    Returns:
        The marked instance or class, or a decorator when ``module`` is
        omitted.

    Marking the same instance or class again replaces its earlier config. An
    instance mark affects only that instance. A class mark is process-wide: it
    applies to every existing and future instance and cannot be removed, so
    ``mark_region(nn.Linear)`` makes every ``nn.Linear`` a region.

    A class mark applies to every class whose ``forward`` is the marked
    class's ``forward``: the class itself, subclasses that inherit that
    ``forward``, and the classes swapped in by ``fully_shard``, ``replicate``
    and parametrizations. Marks on such classes merge field by field, nearest
    class first. A subclass that defines its own ``forward`` (even one that
    only calls ``super().forward()``) is a region only if it is marked, and a
    mark on a mixin that does not supply the resolved ``forward`` never
    applies, whatever the base order. Marking a class without a ``forward``
    raises.

    A marked module nested inside another marked module is not a separate
    region: the outermost mark wins, so marking the root makes the whole model
    one region.
    """
    config = _validate_config(config)

    def decorate(obj: _Marked) -> _Marked:
        if isinstance(obj, type):
            if not issubclass(obj, nn.Module):
                raise TypeError("mark_region can decorate only nn.Module classes")
            if obj.forward is nn.Module.forward:
                raise TypeError(
                    f"mark_region needs a class with a forward, got {obj.__name__}"
                )
            setattr(obj, _REGION_CONFIG_ATTR, config)
            return obj
        if not isinstance(obj, nn.Module):
            raise TypeError(f"mark_region expects an nn.Module, got {type(obj)!r}")
        # Instance marks bypass nn.Module.__setattr__ and never touch the class.
        obj.__dict__[_REGION_CONFIG_ATTR] = config
        return obj

    return decorate if module is _UNSET else decorate(module)


def _first_set_values(*configs: RegionConfig | None) -> dict[str, Any]:
    """Per field, the first value that is set across ``configs``, else unset."""
    values = {}
    for field in dataclasses.fields(RegionConfig):
        values[field.name] = next(
            (
                getattr(config, field.name)
                for config in configs
                if config is not None and getattr(config, field.name) is not _UNSET
            ),
            _UNSET,
        )
    return values


def _merge_configs(
    class_config: RegionConfig | None, instance_config: RegionConfig | None
) -> RegionConfig:
    values = _first_set_values(instance_config, class_config)
    for name, value in values.items():
        if value is _UNSET:
            values[name] = _CONFIG_DEFAULTS[name]
    return RegionConfig(**values)


def _class_region_config(cls: type) -> RegionConfig | None:
    """Class marks that apply to instances of ``cls``, merged nearest first.

    A mark applies when the marked class resolves to the same ``forward`` as
    ``cls``. Class swaps by ``fully_shard``, ``replicate`` and parametrizations
    add no ``forward``, so the original class's mark still applies.
    """
    forward = cls.forward
    configs: list[RegionConfig] = []
    for klass in cls.__mro__:
        config = klass.__dict__.get(_REGION_CONFIG_ATTR)
        if isinstance(config, RegionConfig) and klass.forward is forward:
            configs.append(config)
    if not configs:
        return None
    return RegionConfig(**_first_set_values(*configs))


def _get_region_config(module: nn.Module) -> RegionConfig | None:
    instance_config = module.__dict__.get(_REGION_CONFIG_ATTR)
    class_config = _class_region_config(type(module))
    if not isinstance(instance_config, RegionConfig):
        instance_config = None
    if instance_config is None and class_config is None:
        return None
    return _merge_configs(class_config, instance_config)


def _has_other_marked_ancestor(
    name: str, owner: int, marked_paths: dict[str, int]
) -> bool:
    """True if a strict ancestor path of ``name`` belongs to another marked module."""
    if name == "":
        return False
    parts = name.split(".")
    ancestors = [""] + [".".join(parts[:i]) for i in range(1, len(parts))]
    return any(marked_paths.get(path, owner) != owner for path in ancestors)


def _find_marked_regions(model: nn.Module) -> list[RegionResult]:
    """Walk ``model.named_modules()`` and return only top-level marked regions.

    Nested marked regions are skipped in favor of their marked ancestor. A
    marked module that is both nested in another region and a top-level region
    through an alias is rejected. Sharing between regions is not otherwise
    tracked: a submodule, marked or not, that is reachable from two regions is
    part of both.
    """
    marked: dict[int, tuple[nn.Module, RegionConfig, list[str]]] = {}
    unmarked: OrderedSet[int] = OrderedSet()
    for name, mod in model.named_modules(remove_duplicate=False):
        if id(mod) in marked:
            marked[id(mod)][2].append(name)
            continue
        if id(mod) in unmarked:
            continue
        cfg = _get_region_config(mod)
        if cfg is None:
            unmarked.add(id(mod))
            continue
        marked[id(mod)] = (mod, cfg, [name])

    marked_paths = {
        name: owner for owner, (_, _, aliases) in marked.items() for name in aliases
    }
    regions: list[RegionResult] = []
    for owner, (mod, cfg, aliases) in marked.items():
        nested = [
            _has_other_marked_ancestor(name, owner, marked_paths) for name in aliases
        ]
        # Nested in a marked ancestor and also a top-level region through
        # another alias: reject instead of picking a side.
        if any(nested) and not all(nested):
            paths = ", ".join(repr(name or "__root__") for name in aliases)
            raise AOTIRegionDiscoveryError(
                f"Aliased selected module crosses a selected-region boundary: {paths}"
            )
        if all(nested):
            continue
        regions.append(
            RegionResult(name=aliases[0], module=mod, config=cfg, aliases=aliases)
        )
    return regions
