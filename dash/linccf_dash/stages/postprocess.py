from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import astropy.units as u
import hats
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dask.distributed import as_completed
from hats.catalog import PartitionInfo
from hats.io import paths
from hats.io.parquet_metadata import write_parquet_metadata
from tqdm.auto import tqdm

from linccf_dash.config import PipelineConfig
from linccf_dash.utils.dask_client import dask_client

STAGE = "postprocess"

# Positional and high-precision time columns that must stay float64
_PRESERVE_FLOAT64 = frozenset([
    "ra", "dec", "raErr", "decErr",
    "x", "y", "xErr", "yErr",
    "coord_ra", "coord_dec", "coord_raErr", "coord_decErr",
    "midpointMjdTai",
])


def run_postprocess(cfg: PipelineConfig, catalog_filter: Optional[list[str]] = None) -> None:
    raw_dir = cfg.run.raw_dir
    hats_dir = cfg.run.hats_dir

    # Load visit map once — only needed when any catalog uses add_mjds
    catalogs = cfg.enabled_catalogs(catalog_filter)
    visit_map: dict = {}
    if any(c.add_mjds for c in catalogs.values()):
        visit_table = pd.read_parquet(
            raw_dir / f"{cfg.run.visit_table_name}.parquet", dtype_backend="pyarrow"
        )
        visit_map = visit_table.set_index("visitId")["expMidptMJD"].to_dict()

    with dask_client(cfg.dask.for_stage(STAGE)) as client:
        for catalog_name, catalog_cfg in catalogs.items():
            _postprocess_catalog(
                catalog_name=catalog_name,
                hats_dir=hats_dir,
                flux_col_prefixes=catalog_cfg.flux_columns,
                add_mjds=catalog_cfg.add_mjds,
                visit_map=visit_map,
                client=client,
            )


def _postprocess_catalog(
    catalog_name: str,
    hats_dir: Path,
    flux_col_prefixes: list[str],
    add_mjds: bool,
    visit_map: dict,
    client,
) -> None:
    catalog_dir = hats_dir / catalog_name
    catalog = hats.read_hats(catalog_dir)
    futures = [
        client.submit(
            _process_partition,
            catalog_dir=catalog_dir,
            target_pixel=pixel,
            flux_col_prefixes=flux_col_prefixes,
            add_mjds=add_mjds,
            visit_map=visit_map,
        )
        for pixel in catalog.get_healpix_pixels()
    ]
    for future in tqdm(as_completed(futures), desc=catalog_name, total=len(futures)):
        if future.status == "error":
            raise future.exception()
    _rewrite_catalog_metadata(catalog, hats_dir)


def _process_partition(
    catalog_dir: Path,
    target_pixel,
    flux_col_prefixes: list[str],
    add_mjds: bool,
    visit_map: dict,
) -> None:
    file_path = hats.io.pixel_catalog_file(catalog_dir, target_pixel)
    table = pd.read_parquet(file_path, dtype_backend="pyarrow")
    if flux_col_prefixes:
        table = _append_mag_and_magerr(table, flux_col_prefixes)
    if add_mjds:
        table = _add_mjd_from_visit(table, visit_map)
    table = _cast_columns_float32(table)
    pq.write_table(
        pa.Table.from_pandas(table, preserve_index=False).replace_schema_metadata(),
        file_path.path,
    )


def _append_mag_and_magerr(table: pd.DataFrame, flux_col_prefixes: list[str]) -> pd.DataFrame:
    mag_cols: dict = {}
    for prefix in flux_col_prefixes:
        flux = table[f"{prefix}Flux"]
        mag_cols[f"{prefix}Mag"] = u.nJy.to(u.ABmag, flux)
        err_col = f"{prefix}FluxErr"
        if err_col in table.columns:
            flux_err = table[err_col]
            upper = u.nJy.to(u.ABmag, flux + flux_err)
            lower = u.nJy.to(u.ABmag, flux - flux_err)
            mag_cols[f"{prefix}MagErr"] = -(upper - lower) / 2
    mag_frame = pd.DataFrame(mag_cols, dtype=pd.ArrowDtype(pa.float32()), index=table.index)
    return pd.concat([table, mag_frame], axis=1)


def _add_mjd_from_visit(table: pd.DataFrame, visit_map: dict) -> pd.DataFrame:
    if "visit" not in table.columns:
        raise ValueError("`visit` column is missing")
    if "midpointMjdTai" in table.columns:
        raise ValueError("`midpointMjdTai` is already present in table")
    mjds = [visit_map.get(v, pa.NA) for v in table["visit"]]
    table["midpointMjdTai"] = pd.Series(mjds, dtype=pd.ArrowDtype(pa.float64()), index=table.index)
    return table


def _cast_columns_float32(table: pd.DataFrame) -> pd.DataFrame:
    cols_to_cast = [
        col for col, dtype in table.dtypes.items()
        if col not in _PRESERVE_FLOAT64 and dtype == pd.ArrowDtype(pa.float64())
    ]
    return table.astype({col: pd.ArrowDtype(pa.float32()) for col in cols_to_cast})


def _rewrite_catalog_metadata(catalog, hats_dir: Path) -> None:
    dest = hats_dir / catalog.catalog_name
    parquet_rows = write_parquet_metadata(dest)
    partition_info = PartitionInfo.read_from_dir(dest)
    partition_info.write_to_file(paths.get_partition_info_pointer(dest))
    now = datetime.now(tz=timezone.utc)
    catalog.catalog_info.copy_and_update(
        total_rows=parquet_rows,
        hats_creation_date=now.strftime("%Y-%m-%dT%H:%M%Z"),
    ).to_properties_file(dest)
