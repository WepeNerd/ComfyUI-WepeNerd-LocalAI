"""Comfy folder-path integration for GGUF models and projectors."""

from __future__ import annotations

from pathlib import Path
import re
import threading
import time


PROJECTOR_TOKENS = ("mmproj", "projector", "vision")
_CACHE_TTL_SECONDS = 5.0
_cache_lock = threading.RLock()
_cache: dict[bool, tuple[float, dict[str, Path]]] = {}
_registered = False


def _folder_paths():
    try:
        import folder_paths
    except ImportError:
        return None
    return folder_paths


def _register_folders():
    global _registered
    folder_paths = _folder_paths()
    if folder_paths is None:
        return None
    if not _registered:
        primary = Path(folder_paths.models_dir) / "LLM"
        primary.mkdir(parents=True, exist_ok=True)
        try:
            folder_paths.add_model_folder_path("LLM", str(primary), is_default=True)
        except TypeError:
            folder_paths.add_model_folder_path("LLM", str(primary))
        legacy = Path(folder_paths.models_dir) / "llm_gguf"
        if legacy.is_dir():
            try:
                folder_paths.add_model_folder_path("llm_gguf", str(legacy), is_default=True)
            except TypeError:
                folder_paths.add_model_folder_path("llm_gguf", str(legacy))
        _registered = True
    return folder_paths


def _is_projector(name: str) -> bool:
    filename = Path(name).name.lower()
    return any(token in filename for token in PROJECTOR_TOKENS)


def _registered_index(projectors: bool) -> dict[str, Path]:
    folder_paths = _register_folders()
    if folder_paths is None:
        return _fallback_index(projectors)

    result: dict[str, Path] = {}
    categories = ["LLM"]
    try:
        folder_paths.get_folder_paths("llm_gguf")
        categories.append("llm_gguf")
    except KeyError:
        pass

    for category in categories:
        try:
            names = folder_paths.get_filename_list(category)
        except KeyError:
            continue
        for name in names:
            if Path(name).suffix.lower() != ".gguf" or _is_projector(name) != projectors:
                continue
            full_path = folder_paths.get_full_path(category, name)
            if full_path:
                result[f"{category}/{Path(name).as_posix()}"] = Path(full_path).resolve()
    return result


def _fallback_index(projectors: bool) -> dict[str, Path]:
    primary = Path.cwd() / "models" / "LLM"
    result: dict[str, Path] = {}
    if not primary.is_dir():
        return result
    root = primary.resolve()
    for path in primary.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".gguf" or _is_projector(path.name) != projectors:
            continue
        resolved = path.resolve()
        if root in resolved.parents:
            result[f"LLM/{path.relative_to(primary).as_posix()}"] = resolved
    return result


def _index(projectors: bool) -> dict[str, Path]:
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(projectors)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return dict(cached[1])
    result = dict(sorted(_registered_index(projectors).items(), key=lambda item: item[0].lower()))
    with _cache_lock:
        _cache[projectors] = (now, result)
    return dict(result)


def clear_discovery_cache() -> None:
    with _cache_lock:
        _cache.clear()


def discover_models() -> list[str]:
    return list(_index(projectors=False))


def discover_projectors() -> list[str]:
    return ["(none)", *_index(projectors=True)]


def resolve_choice(choice: str, projector: bool = False) -> str:
    path = _index(projectors=projector).get(choice)
    if path is None:
        clear_discovery_cache()
        path = _index(projectors=projector).get(choice)
    if path is None:
        kind = "projector" if projector else "model"
        raise ValueError(
            f"Unknown GGUF {kind} selection: {choice!r}. Refresh ComfyUI after "
            "adding or moving model files under ComfyUI/models/LLM."
        )
    return str(path)


_NAME_NOISE = {"mmproj", "projector", "vision", "model", "gguf"}
_QUANT_TOKEN = re.compile(r"^(?:i?q\d+\w*|f16|f32|bf16|fp16|fp32|[a-z]|xs|xxs|nl)$")


def _name_key(path: Path) -> str:
    """Model identity from a filename, ignoring projector words and quantization tags."""
    tokens = re.split(r"[-_.\s]+", path.stem.lower())
    return "-".join(tok for tok in tokens if tok and tok not in _NAME_NOISE and not _QUANT_TOKEN.match(tok))


def match_projector(model_path: str) -> str | None:
    """Pick the projector that belongs to a model, or None when there is no confident match.

    1. Name match: the projector's name parts (minus mmproj/quant tags) appear, in order
       and as whole parts, in the model's name. The longest match wins; ties prefer the model's own folder.
    2. Folder match: the model's folder holds exactly one model and one projector, and
       that projector has a generic name such as ``mmproj-model-f16.gguf``.
    """
    model = Path(model_path).resolve()
    model_key = _name_key(model)
    projectors = list(_index(projectors=True).values())

    scored = []
    for projector in projectors:
        key = _name_key(projector)
        if len(key.replace("-", "")) >= 4 and f"-{key}-" in f"-{model_key}-":
            scored.append((len(key), projector.parent == model.parent, projector))
    if scored:
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best = scored[0]
        if len(scored) == 1 or (best[0], best[1]) != (scored[1][0], scored[1][1]):
            return str(best[2])
        return None

    same_folder = [p for p in projectors if p.parent == model.parent]
    models_in_folder = [m for m in _index(projectors=False).values() if m.parent == model.parent]
    generic = len(same_folder) == 1 and len(_name_key(same_folder[0]).replace("-", "")) < 4
    if generic and len(models_in_folder) == 1:
        return str(same_folder[0])
    return None
