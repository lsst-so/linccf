from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import lsdb
from hats_import import pipeline_with_client
from hats_import.catalog import ImportArguments
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments

from typing import Optional

from linccf_dash.config import NestedConfig, PipelineConfig
from linccf_dash.utils.dask_client import dask_client

STAGE = "nesting"


def run_nesting(cfg: PipelineConfig, nesting_filter: Optional[list[str]] = None) -> None:
    hats_dir = cfg.run.hats_dir

    with dask_client(cfg.dask.for_stage(STAGE)) as client:
        for nested_name, nested_cfg in cfg.enabled_nestings(nesting_filter).items():
            _build_nested_catalog(
                nested_name=nested_name,
                nested_cfg=nested_cfg,
                hats_dir=hats_dir,
                client=client,
            )


def _build_nested_catalog(
    nested_name: str,
    nested_cfg: NestedConfig,
    hats_dir: Path,
    client,
) -> None:
    margin_tmp = tempfile.TemporaryDirectory()
    margin_dir = Path(margin_tmp.name)

    try:
        # Build margin caches for all source catalogs
        for source_name in nested_cfg.source_catalogs:
            args = MarginCacheArguments(
                input_catalog_path=hats_dir / source_name,
                output_path=margin_dir,
                margin_threshold=nested_cfg.margin_radius_arcsec,
                output_artifact_name=f"{source_name}_{nested_cfg.margin_radius_arcsec}arcs",
                simple_progress_bar=True,
                resume=False,
            )
            pipeline_with_client(args, client)

        # Load object catalog
        obj_cat = lsdb.read_hats(hats_dir / nested_cfg.object_catalog)

        # Load and join each source catalog
        nested_cat = obj_cat
        for source_name, column_name in zip(
            nested_cfg.source_catalogs, nested_cfg.nested_column_names
        ):
            margin_path = margin_dir / f"{source_name}_{nested_cfg.margin_radius_arcsec}arcs"
            src_cat = lsdb.read_hats(hats_dir / source_name, margin_cache=margin_path)
            nested_cat = nested_cat.join_nested(
                src_cat,
                left_on=nested_cfg.join_id,
                right_on=nested_cfg.join_id,
                nested_column_name=column_name,
            )

        # Sort sources within each object by MJD
        source_cols = nested_cfg.nested_column_names
        nested_cat = nested_cat.map_partitions(
            lambda df: _sort_nested_sources(df, source_cols, nested_cfg.sort_column)
        )

        # Save intermediate and reimport with production settings
        intermediate_path = hats_dir / f"{nested_name}_intermediate"
        nested_cat.to_hats(intermediate_path, catalog_name=nested_name)

        # Compute hats_cols_default if default columns are specified
        addl_props: dict = {}
        if nested_cfg.default_columns:
            actual_cols = set(_full_column_names(nested_cat))
            valid_default_cols = [c for c in nested_cfg.default_columns if c in actual_cols]
            missing = sorted(set(nested_cfg.default_columns) - actual_cols)
            if missing:
                print(f"Warning: requested default columns missing from {nested_name}: {', '.join(missing)}")
            addl_props["hats_cols_default"] = ",".join(valid_default_cols)

        reimport_args = ImportArguments.reimport_from_hats(
            intermediate_path,
            output_dir=hats_dir,
            highest_healpix_order=nested_cfg.highest_healpix_order,
            pixel_threshold=nested_cfg.pixel_threshold,
            skymap_alt_orders=nested_cfg.skymap_alt_orders,
            row_group_kwargs=nested_cfg.row_group_kwargs,
            **({"addl_hats_properties": addl_props} if addl_props else {}),
        )
        pipeline_with_client(reimport_args, client)
        shutil.rmtree(intermediate_path)

    finally:
        margin_tmp.cleanup()


def _sort_nested_sources(df, source_cols: list[str], sort_col: str):
    for col in source_cols:
        flat = df[col].nest.to_flat()
        df = df.drop(columns=[col])
        df = df.join_nested(
            flat.sort_values([flat.index.name, sort_col]), col
        )
    return df


def _full_column_names(cat):
    """Yield all column names including nested sub-columns as 'nested_col.field'."""
    for c in cat.columns:
        cc = cat[c]
        if not hasattr(cc, "nest"):
            yield c
        else:
            for f in cc.nest.columns:
                yield f"{c}.{f}"
