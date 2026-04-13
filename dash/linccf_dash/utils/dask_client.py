from __future__ import annotations

import tempfile
from contextlib import contextmanager
from typing import Any

from dask.distributed import Client


@contextmanager
def dask_client(client_kwargs: dict[str, Any] | None = None):
    """Context manager that creates a Dask client with a temporary local directory.

    Args:
        client_kwargs: Keyword arguments forwarded to ``dask.distributed.Client``.
            ``local_directory`` is set automatically to a temp dir unless already provided.
    """
    kwargs = dict(client_kwargs or {})
    tmp = None
    if "local_directory" not in kwargs:
        tmp = tempfile.TemporaryDirectory()
        kwargs["local_directory"] = tmp.name
    client = Client(**kwargs)
    try:
        yield client
    finally:
        client.close()
        if tmp is not None:
            tmp.cleanup()
