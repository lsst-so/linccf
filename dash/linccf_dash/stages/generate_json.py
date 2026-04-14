from __future__ import annotations

import json
import subprocess
from typing import Optional

import lsdb

from linccf_dash.config import PipelineConfig


def run_generate_json(cfg: PipelineConfig, collection_filter: Optional[list[str]] = None) -> None:
    hats_dir = cfg.run.hats_dir
    run_cfg = cfg.run

    collections_json = [
        _generate_collection_json(collection_name, hats_dir, run_cfg)
        for collection_name in cfg.enabled_collections(collection_filter)
    ]

    out_path = hats_dir / f"{run_cfg.version}.json"
    with open(out_path, "w") as f:
        json.dump(collections_json, f)
    print(f"Saved {out_path}")


def _generate_collection_json(collection_name: str, hats_dir, run_cfg) -> dict:
    collection_path = hats_dir / collection_name
    catalog = lsdb.read_hats(collection_path)

    version = run_cfg.version
    run = run_cfg.run
    collection_tag = run_cfg.collection

    name = f"{run} {version} {collection_name}" if run else f"{version} {collection_name}"
    description = (
        f"DRP/{run}/{version}/{collection_tag} {collection_name}"
        if run
        else f"DRP/{version}/{collection_tag} {collection_name}"
    )

    return {
        "label": f"{version}/{collection_name}",
        "name": name,
        "description": description,
        "urls": {"catalog": str(collection_path)},
        "other_urls": [
            {"label": "Column descriptions", "url": "https://sdm-schemas.lsst.io/imsim.html"},
            {"label": "Jira Ticket", "url": f"https://rubinobs.atlassian.net/browse/{collection_tag}"},
        ],
        "metadata": {
            "numRows": len(catalog),
            "numColumns": len(catalog.all_columns),
            "numPartitions": len(catalog.get_healpix_pixels()),
            "sizeOnDisk": _directory_size(collection_path),
            "hatsBuilder": catalog.hc_structure.catalog_info.extra_dict()["hats_builder"],
        },
        "badges": [{"title": "Available only on USDF"}],
    }


def _directory_size(path) -> str:
    SIZE_UNITS = {"G": "GiB", "M": "MiB", "K": "KiB", "T": "TiB"}
    result = subprocess.run(["du", "-sh", path], capture_output=True, text=True, check=True)
    size_str = result.stdout.split("\t")[0]
    unit = size_str[-1]
    return f"{size_str[:-1]} {SIZE_UNITS[unit]}"
