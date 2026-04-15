from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import lsst.daf.butler as dafButler
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from linccf_dash.config import PipelineConfig


def run_butler(cfg: PipelineConfig, catalog_filter: Optional[list[str]] = None) -> None:
    raw_dir = cfg.run.raw_dir
    for subdir in ("paths", "refs", "sizes"):
        (raw_dir / subdir).mkdir(parents=True, exist_ok=True)

    col_butler = dafButler.Butler(cfg.run.repo)

    # Expand your pattern into a real list
    collections = list(col_butler.registry.queryCollections(cfg.run.butler_collection))

    if len(collections) > 1:
        print(f"Found {len(collections)} collections matching pattern '{cfg.run.butler_collection}':")
        print(", ".join(collections))

    butler = dafButler.Butler(cfg.run.repo, collections=collections)

    for catalog_name in cfg.enabled_catalogs(catalog_filter):
        _get_uris_from_butler(butler, catalog_name, raw_dir)

    _get_visits_from_butler(butler, cfg.run.instrument, cfg.run.visit_table_name, raw_dir)


def _get_uris_from_butler(butler, dataset_type: str, raw_dir: Path) -> None:
    start = time.perf_counter()
    refs = butler.query_datasets(dataset_type, limit=None)
    uris = butler._datastore.getManyURIs(refs)
    paths = [value.primaryURI.geturl() for value in uris.values()]

    (raw_dir / "paths" / f"{dataset_type}.txt").write_text(
        "\n".join(paths) + "\n", encoding="utf8"
    )

    ref_ids = [ref.dataId.mapping for ref in refs]
    pd.DataFrame(ref_ids).to_csv(raw_dir / "refs" / f"{dataset_type}.csv", index=False)

    print(
        f"Found {len(paths):>6} files for {dataset_type:>30} "
        f"in {time.perf_counter() - start:10.2f}s"
    )


def _get_visits_from_butler(butler, instrument: str, visits_type: str, raw_dir: Path) -> None:
    visits = butler.get(visits_type, dataId={"instrument": instrument})
    parquet_path = raw_dir / f"{visits_type}.parquet"
    pq.write_table(pa.Table.from_pandas(visits.to_pandas()), parquet_path)
    print(f"Saved {len(visits)} visit rows to {parquet_path}")
