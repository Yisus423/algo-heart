"""The frozen artifact: what it holds, and how it is written and read back.

``Artifact`` lives in its own module on purpose. Python executes
``python -m heartrisk.train`` as the ``__main__`` module, so any class defined in
``heartrisk/train.py`` would be pickled as ``__main__.Artifact`` and could never
be unpickled by an application that imports ``heartrisk`` normally. Keeping the
class in a module that is never executed as ``__main__`` is what makes the
documented training command produce a loadable file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from .data import REPO_ROOT
from .schema import FEATURE_ORDER, SCHEMA_VERSION, SchemaError

ARTIFACT_PATH = REPO_ROOT / "artifacts" / "model.joblib"
METRICS_PATH = REPO_ROOT / "artifacts" / "metrics.json"

#: The forest is stored as hundreds of full trees, so the raw dump is about
#: 31 MB. Compression brings it under a megabyte at no measurable read cost.
COMPRESSION = 3


@dataclass(frozen=True, slots=True)
class Artifact:
    """A trained model plus everything needed to serve it safely.

    Attributes:
        model: the fitted classifier.
        feature_order: design-matrix layout it was trained on.
        schema_version: contract version, checked at load time.
        threshold: probability cutoff selected for the training recall target.
        imputation: substituted value per non-categorical field. The median for
            numeric fields, the majority value for boolean and binary fields, so
            a filled-in value always lies inside the training support.
        ood_bounds: ``(p01, p99)`` per numeric field, used to flag an input the
            model has effectively never seen.
        metrics: measured performance, including the interview-only baseline.
        trained_at: UTC timestamp, for provenance.
    """

    model: RandomForestClassifier
    feature_order: tuple[str, ...]
    schema_version: str
    threshold: float
    imputation: dict[str, float]
    ood_bounds: dict[str, tuple[float, float]]
    metrics: dict[str, object] = field(default_factory=dict)
    trained_at: str = ""


def save_artifact(artifact: Artifact, path: Path | str = ARTIFACT_PATH) -> Path:
    """Write the artifact, creating its directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=COMPRESSION)
    return path


def load_artifact(path: Path | str = ARTIFACT_PATH) -> Artifact:
    """Load an artifact, refusing one built against a different contract.

    A stale artifact is dangerous rather than merely wrong: the column order may
    have moved, and the model would keep answering with full confidence while
    reading a different feature.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no artifact at {path}; build one with `python -m heartrisk.train`"
        )
    artifact: Artifact = joblib.load(path)

    if artifact.schema_version != SCHEMA_VERSION:
        raise SchemaError(
            f"artifact schema {artifact.schema_version!r} does not match the "
            f"current schema {SCHEMA_VERSION!r}; rebuild with "
            "`python -m heartrisk.train`"
        )
    if tuple(artifact.feature_order) != FEATURE_ORDER:
        raise SchemaError(
            "artifact feature layout does not match the contract; "
            "rebuild with `python -m heartrisk.train`"
        )
    return artifact
