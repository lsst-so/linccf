from __future__ import annotations

import importlib.resources
import tomllib
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, model_validator

# ImportArguments fields that are always set programmatically — not allowed in import_args config
_IMPORT_ARGS_MANAGED = frozenset({
    "output_path",
    "output_artifact_name",
    "input_file_list",
    "file_reader",
    "use_schema_file",  # handled via the top-level use_schema_file bool
})


class RunConfig(BaseModel):
    instrument: str
    repo: str
    version: str
    collection: str
    output_dir: Path
    run: Optional[str] = None
    visit_table_name: str = "visit_table"

    @property
    def butler_collection(self) -> str:
        if self.run:
            return f"{self.instrument}/runs/DRP/{self.run}/{self.version}/{self.collection}"
        return f"{self.instrument}/runs/DRP/{self.version}/{self.collection}"

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw" / self.version

    @property
    def hats_dir(self) -> Path:
        return self.output_dir / "hats" / self.version

    @property
    def validation_dir(self) -> Path:
        return self.output_dir / "validation" / self.version


class StagesConfig(BaseModel):
    enabled: list[str] = [
        "butler",
        "raw_sizes",
        "import",
        "postprocess",
        "nesting",
        "collections",
        "crossmatch",
        "generate_json",
    ]


_ALL_CATALOGS = [
    "dia_object",
    "dia_source",
    "dia_object_forced_source",
    "object",
    "source",
    "object_forced_source",
]


class CatalogConfig(BaseModel):
    dims: list[str] = []       # dimension columns added to index files from refs CSV
    group_by: list[str] = []   # columns to group index batch files by
    flux_columns: list[str] = []
    add_mjds: bool = False
    use_schema_file: bool = False
    chunksize: int = 500_000   # DimensionParquetReader batch size
    import_args: dict[str, Any] = {}

    @model_validator(mode="after")
    def _validate_import_args(self) -> CatalogConfig:
        managed = set(self.import_args.keys()) & _IMPORT_ARGS_MANAGED
        if managed:
            raise ValueError(
                f"These fields are managed automatically and cannot be set in import_args: "
                f"{', '.join(sorted(managed))}"
            )
        return self


class CatalogsConfig(BaseModel):
    """Holds the enabled list and the per-catalog config tables.

    TOML shape:
        [catalogs]
        enabled = ["dia_object", "object"]

        [catalogs.dia_object]
        dims = ["tract"]
        ...
    """
    enabled: list[str] = list(_ALL_CATALOGS)
    configs: dict[str, CatalogConfig] = {}

    @model_validator(mode="before")
    @classmethod
    def _split_enabled_and_configs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        configs = {k: v for k, v in data.items() if k != "enabled" and isinstance(v, dict)}
        result: dict[str, Any] = {"configs": configs}
        if "enabled" in data:
            result["enabled"] = data["enabled"]
        return result


class NestedConfig(BaseModel):
    object_catalog: str
    join_id: str                      # e.g. "diaObjectId" or "objectId"
    source_catalogs: list[str]
    nested_column_names: list[str]    # parallel to source_catalogs
    sort_column: str = "midpointMjdTai"
    margin_radius_arcsec: int = 2
    pixel_threshold: int = 15_000
    highest_healpix_order: int = 11
    skymap_alt_orders: list[int] = [2, 4, 6]
    row_group_kwargs: dict[str, Any] = {"subtile_order_delta": 1}
    default_columns: list[str] = []   # hats_cols_default; empty = all columns


class CollectionConfig(BaseModel):
    nested_catalog: str
    margin_threshold: float = 5.0
    index_column: str


class NestedConfigs(BaseModel):
    """Holds the enabled list and per-nested-catalog config tables.

    TOML shape:
        [nested]
        enabled = ["object_lc"]   # optional — omit to run all

        [nested.object_lc]
        object_catalog = "object"
        ...
    """
    enabled: Optional[list[str]] = None
    configs: dict[str, NestedConfig] = {}

    @model_validator(mode="before")
    @classmethod
    def _split_enabled_and_configs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        configs = {k: v for k, v in data.items() if k != "enabled" and isinstance(v, dict)}
        result: dict[str, Any] = {"configs": configs}
        if "enabled" in data:
            result["enabled"] = data["enabled"]
        return result


class CollectionsConfig(BaseModel):
    """Holds the enabled list and per-collection config tables.

    TOML shape:
        [collections]
        enabled = ["object_collection"]   # optional — omit to run all

        [collections.object_collection]
        nested_catalog = "object_lc"
        ...
    """
    enabled: Optional[list[str]] = None
    configs: dict[str, CollectionConfig] = {}

    @model_validator(mode="before")
    @classmethod
    def _split_enabled_and_configs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        configs = {k: v for k, v in data.items() if k != "enabled" and isinstance(v, dict)}
        result: dict[str, Any] = {"configs": configs}
        if "enabled" in data:
            result["enabled"] = data["enabled"]
        return result


class CrossmatchSurveyConfig(BaseModel):
    path: str
    radius_arcsec: float = 0.2
    n_neighbors: int = 20
    suffix: str           # e.g. "_ztf" — appended by crossmatch, used to derive xmatch col name
    join_id_column: str   # ID column in the external catalog, e.g. "objectid"
    s3_endpoint_url: Optional[str] = None
    s3_anon: bool = False


class CrossmatchConfig(BaseModel):
    surveys: dict[str, CrossmatchSurveyConfig] = {}


class DaskConfig(BaseModel):
    """Global Dask client settings and per-stage overrides.

    TOML shape:
        [dask]
        n_workers = 8
        threads_per_worker = 1
        # any other kwargs accepted by dask.distributed.Client

        [dask.stages.import]
        n_workers = 8
        memory_limit = "16GB"

        [dask.stages.postprocess]
        n_workers = 16
    """
    global_kwargs: dict[str, Any] = {}
    stages: dict[str, dict[str, Any]] = {}

    @model_validator(mode="before")
    @classmethod
    def _split_global_and_stages(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        stages = data.get("stages", {})
        global_kwargs = {k: v for k, v in data.items() if k != "stages"}
        return {"global_kwargs": global_kwargs, "stages": stages}

    def for_stage(self, stage: str) -> dict[str, Any]:
        """Return merged client kwargs: global defaults overridden by stage-specific settings."""
        return {**self.global_kwargs, **self.stages.get(stage, {})}


class PipelineConfig(BaseModel):
    run: RunConfig
    stages: StagesConfig = StagesConfig()
    catalogs: CatalogsConfig = CatalogsConfig()
    nested: NestedConfigs = NestedConfigs()
    collections: CollectionsConfig = CollectionsConfig()
    crossmatch: CrossmatchConfig = CrossmatchConfig()
    dask: DaskConfig = DaskConfig()

    def enabled_catalogs(self, filter: Optional[list[str]] = None) -> dict[str, CatalogConfig]:
        """Return configs for enabled catalogs, optionally filtered to a subset by name."""
        names = self.catalogs.enabled
        if filter is not None:
            names = [n for n in names if n in filter]
        result = {}
        for name in names:
            if name not in self.catalogs.configs:
                raise ValueError(f"Catalog '{name}' is listed in catalogs.enabled but has no config section")
            result[name] = self.catalogs.configs[name]
        return result

    def enabled_nestings(self, filter: Optional[list[str]] = None) -> dict[str, NestedConfig]:
        """Return configs for enabled nested catalogs, optionally filtered to a subset by name."""
        names = self.nested.enabled if self.nested.enabled is not None else list(self.nested.configs.keys())
        if filter is not None:
            names = [n for n in names if n in filter]
        return {n: self.nested.configs[n] for n in names if n in self.nested.configs}

    def enabled_collections(self, filter: Optional[list[str]] = None) -> dict[str, CollectionConfig]:
        """Return configs for enabled collections, optionally filtered to a subset by name."""
        names = self.collections.enabled if self.collections.enabled is not None else list(self.collections.configs.keys())
        if filter is not None:
            names = [n for n in names if n in filter]
        return {n: self.collections.configs[n] for n in names if n in self.collections.configs}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Dicts are merged; all other types are replaced."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_builtin_defaults() -> dict:
    ref = importlib.resources.files("linccf_dash").joinpath("default_config.toml")
    with ref.open("rb") as f:
        return tomllib.load(f)


def load_config(paths: Path | list[Path]) -> PipelineConfig:
    """Load pipeline config by merging built-in defaults with one or more TOML files.

    Files are applied left to right; later files override earlier ones.
    """
    if isinstance(paths, Path):
        paths = [paths]

    merged = _load_builtin_defaults()
    for path in paths:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        merged = _deep_merge(merged, data)

    return PipelineConfig.model_validate(merged)
