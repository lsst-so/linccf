## ML-based Similarity Search

**Mission:** Find new transients which look like something already classified by people in the First Look or in the alerts stream. We are going to convert lightcurves to ML embeddings and use similarity search to query with known objects.

1. Load "good" DIA light curves, use non-forced DIA sources, and filter by the flags.
2. Below are a few transients the community has identified the LSST alerts and on the First Look images. I haven’t checked DP2 lightcurves; check visually that you see a transient, and select a few objects to use as “queries”.

```
187.4565 8.213469
186.016373037 8.4117596341
149.261451 1.291222
51.620551 -28.114117
10.581959 -45.278411
10.918621 -44.157346
10.178225 -45.842358
52.718802 -28.362480
61.964820 -48.713443
52.786108 -27.341361
```

3. Convert light curves to their embeddings. You can use one of the available models in [`light_curve.embed`](https://light-curve.snad.space/dev/embed/). ATCAT is a good starting point.
4. Check if the pipeline benefits from running on a GPU (arnor vs gondor or RM vs GPU on bridges2) and if yes, set up the Dask Client to utilize GPU efficiently.
5. Save embeddings as a new HATS catalog. Run another pipeline to query it for the top-5 smallest cosine / L2 similarity for each “query” object. Visually inspect these top-5 light curves.

**Definition of done:**

- Embeddings for filtered non-forced DIA light curves are generated and saved as a HATS catalog.
- There is a quantitative benchmark comparing CPU vs GPU execution (throughput/runtime) and a documented compute choice.
- For each selected query transient, the pipeline returns top-5 nearest neighbors (cosine/L2).

## Installation

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
pixi shell
pixi run jupyter lab --no-browser
# Copy URL to VSCode
# Select "Python 3" kernel for the local pixi environment
```

## Feedback

### Notebook 1.

- I think this is a known issue but the "double-nested" representation is not well formatted in Jupyter.

### Notebook 2.

- Is it useful to have the Deep Drilling Fields definitions in the `lsdb-rubin` package?

- I tried to concat two catalogs but it failed with Arrow issue:

  ```python
  ecdfs.concat(edfs)
  ```

- I tried the following alternatives to remove objects with no `diaSource`:

  ```python
  ecdfs = ecdfs.dropna(subset="diaSource")
  ecdfs = ecdfs[~ecdfs["diaSource"].isna()]
  ```

  They ceased to exist in operations, but `.query()` works.

- When `Catalog.map_rows` gets an empty partition it fails with metadata mismatch. The following workaround works:

  ```
  def _hack(df):
      if len(df) == 0:
          df["embeddings.value"] = pd.Series([], dtype=np.float32)
          return df
      return df.map_rows(
          compute_embeddings,
          columns=[...],
          row_container="args",
          append_columns=True,
      )

  ecdfs.map_partitions(lambda df: _hack(df))
  ```

- `Catalog.head` shows compute bar for each partition processed.

### Notebook 3.

- Getting the N-closest matches on the whole sky is not trivial with parallelization.
  - Should we have a tutorial that exemplifies how to do it at scale?

### Notebook 5.

- It seems like compute with a distributed Client is much slower than using the local Client?
