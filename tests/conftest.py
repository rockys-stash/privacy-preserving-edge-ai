"""Tests run on a few subjects so the suite stays fast; no reported result uses this subset."""

import os

os.environ.setdefault("EDGEBENCH_SUBJECTS", "1,2,3,5,6,7,8,9")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest

from edgebench import data

HAVE_DATA = (data.RAW / data.FOLDER / "train" / "y_train.txt").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="dataset not fetched (make data)")
