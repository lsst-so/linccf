from __future__ import annotations

from typing import Optional

import lsdb
from upath import UPath

from linccf_dash.config import CrossmatchSurveyConfig, PipelineConfig
from linccf_dash.utils.dask_client import dask_client

STAGE = "crossmatch"


def run_crossmatch(cfg: PipelineConfig, collection_filter: Optional[list[str]] = None) -> None:
    from lsdb.io.to_association import to_association

    hats_dir = cfg.run.hats_dir

    with dask_client(cfg.dask.for_stage(STAGE)) as client:  # noqa: F841
        collections = [
            lsdb.open_catalog(hats_dir / name) for name in cfg.enabled_collections(collection_filter)
        ]

        for survey_name, survey_cfg in cfg.crossmatch.surveys.items():
            survey_cat = _open_survey(survey_cfg)

            for collection in collections:
                collection_props = collection.hc_collection.collection_properties
                collection_name = collection_props.name
                lsst_id_col = next(iter(collection_props.all_indexes))

                xmatch = collection.crossmatch(
                    survey_cat,
                    radius_arcsec=survey_cfg.radius_arcsec,
                    n_neighbors=survey_cfg.n_neighbors,
                    suffixes=("", survey_cfg.suffix),
                )

                xmatch_name = f"{collection.hc_structure.catalog_name}_x_{survey_name}"
                join_col = f"{survey_cfg.join_id_column}{survey_cfg.suffix}"

                to_association(
                    xmatch[[lsst_id_col, join_col, "_dist_arcsec"]],
                    catalog_name=xmatch_name,
                    base_catalog_path=hats_dir / collection_name / xmatch_name,
                    primary_catalog_dir=hats_dir / collection_name,
                    primary_column_association=lsst_id_col,
                    primary_id_column=lsst_id_col,
                    join_catalog_dir=survey_cat.hc_structure.catalog_path,
                    join_column_association=join_col,
                    join_id_column=survey_cfg.join_id_column,
                )
                print(f"Saved {xmatch_name}")


def _open_survey(survey_cfg: CrossmatchSurveyConfig):
    if survey_cfg.s3_endpoint_url:
        s3_kwargs = {"endpoint_url": survey_cfg.s3_endpoint_url, "anon": survey_cfg.s3_anon}
        return lsdb.open_catalog(UPath(survey_cfg.path, **s3_kwargs))
    return lsdb.open_catalog(survey_cfg.path)
