from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hats.io.validation import is_valid_catalog
from hats_import import pipeline_with_client
from hats_import.catalog.arguments import ImportArguments
from lsst.resources import ResourcePath

from linccf_dash.config import PipelineConfig
from linccf_dash.utils.dask_client import dask_client
from linccf_dash.utils.readers import DimensionParquetReader

STAGE = "import"


def run_import(cfg: PipelineConfig, catalog_filter: Optional[list[str]] = None) -> None:
    raw_dir = cfg.run.raw_dir
    hats_dir = cfg.run.hats_dir
    hats_dir.mkdir(parents=True, exist_ok=True)

    with dask_client(cfg.dask.for_stage(STAGE)) as client:
        for catalog_name, catalog_cfg in cfg.enabled_catalogs(catalog_filter).items():
            if is_valid_catalog(hats_dir / catalog_name):
                print(f"Skipping {catalog_name} — already imported.\n")
                continue
            print(f"Starting import for {catalog_name}...\n")

            index_files = list((raw_dir / "index" / catalog_name).glob("*.csv"))

            schema_file: Optional[Path] = None
            if catalog_cfg.use_schema_file:
                dimension_columns = set(pd.read_csv(index_files[0]).columns) - {"path"}
                schema_file = _download_schema(catalog_name, raw_dir, list(dimension_columns))

            args = ImportArguments(
                output_path=hats_dir,
                output_artifact_name=catalog_name,
                input_file_list=index_files,
                file_reader=DimensionParquetReader(chunksize=catalog_cfg.chunksize),
                **({"use_schema_file": schema_file} if schema_file else {}),
                **catalog_cfg.import_args,
            )
            pipeline_with_client(args, client)


def _download_schema(
    catalog_name: str,
    raw_dir: Path,
    dimension_columns: list[str],
) -> Path:
    """Download and cache the schema for catalogs that need consistent column ordering."""
    paths_file = raw_dir / "paths" / f"{catalog_name}.txt"
    first_path = paths_file.read_text(encoding="utf8").splitlines()[0].strip()

    with ResourcePath(first_path).open("rb") as f:
        schema = pq.read_schema(f).remove_metadata()

    schema_table = pa.table({field.name: pa.array([], type=field.type) for field in schema})
    for col in dimension_columns:
        if col not in schema_table.column_names:
            schema_table = schema_table.append_column(col, pa.array([], type=pa.int64()))

    schema_path = raw_dir / f"{catalog_name}_schema.parquet"
    pq.write_table(schema_table, schema_path)
    return schema_path
