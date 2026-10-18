"""Heart risk demo: contract, pipeline, and serving code.

Layout:
    schema.py   the data contract and the frozen design-matrix layout
    data.py     load, validate, clean, and build matrices
    train.py    fit and serialise the frozen artifact
    predict.py  load the artifact and answer with a declared coverage level
"""

from .schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
