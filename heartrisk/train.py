"""Fit the model and freeze everything the app needs into one artifact.

The project's original script trained a model inside a function and printed a
confusion matrix whose labels described mortality and survival for a target that
is actually ``has_heart_disease``. It never wrote the model or the scaler
anywhere, so there was nothing for an application to load. This module produces
the artifact instead; :mod:`heartrisk.artifact` owns its shape and its pickle
identity.

The artifact is deliberately self-describing: it carries the schema version, the
exact column layout, the imputation values, the out-of-distribution bounds, the
decision threshold and the measured metrics. Serving code can therefore refuse a
stale artifact instead of silently reading the wrong column.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .artifact import (
    ARTIFACT_PATH,
    METRICS_PATH,
    Artifact,
    load_artifact,
    save_artifact,
)
from .data import DEFAULT_RAW_PATH, clean, load_raw, to_features
from .schema import (
    ASKABLE_FIELDS,
    FEATURE_ORDER,
    FIELDS,
    SCHEMA_VERSION,
    TARGET,
    SchemaError,
)

__all__ = [
    "ARTIFACT_PATH",
    "METRICS_PATH",
    "Artifact",
    "choose_threshold",
    "fit",
    "load_artifact",
    "main",
    "save_artifact",
    "write_metrics_report",
]

TEST_SIZE = 0.25
RANDOM_STATE = 42
N_ESTIMATORS = 300

#: The demo is a screening-shaped product: telling someone they are fine when
#: they are not is the expensive mistake, so the decision threshold is chosen to
#: reach this recall on held-out data rather than to maximise accuracy. Accuracy
#: is the wrong headline metric here: 30% of the rows are positive, so a model
#: that always answers "no" already scores 70%.
TARGET_RECALL = 0.85


def choose_threshold(
    y_true: object, probability: object, target_recall: float = TARGET_RECALL
) -> float:
    """Return the strictest cutoff that still reaches ``target_recall``.

    Walking the candidate cutoffs from high to low and stopping at the first one
    that reaches the target returns the highest threshold that qualifies, which
    is the one that produces the fewest false alarms for the required recall.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        raise SchemaError("cannot choose a threshold without any positive examples")

    for candidate in np.unique(p)[::-1]:
        if y[p >= candidate].sum() / positives >= target_recall:
            return float(candidate)
    return 0.0


def _column_owner(column: str) -> str:
    """Map a design-matrix column back to the contract field it came from."""
    return column.split("=", 1)[0]


def _imputation_values(cleaned: pd.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    for contract_field in FIELDS:
        if contract_field.kind == "categorical":
            continue
        series = cleaned[contract_field.name]
        if contract_field.kind == "numeric":
            values[contract_field.name] = float(series.median())
        else:
            # Keep a boolean or binary field inside {0, 1}.
            values[contract_field.name] = float(round(float(series.median())))
    return values


def _ood_bounds(cleaned: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        contract_field.name: (
            float(cleaned[contract_field.name].quantile(0.01)),
            float(cleaned[contract_field.name].quantile(0.99)),
        )
        for contract_field in FIELDS
        if contract_field.kind == "numeric"
    }


def _interview_only_baseline(
    cleaned: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    n_estimators: int,
    target_recall: float,
) -> dict[str, float]:
    """Measure how the model performs with interview answers alone.

    This is the number behind the app's confidence band. Without it the band
    would be a stylistic choice rather than a measured fact, and the measured
    fact is uncomfortable: at the same recall, dropping the measurements roughly
    halves precision, so a conversation alone reaches the same sensitivity only
    by flagging far more healthy people.
    """
    features = to_features(cleaned)
    askable = [c for c in FEATURE_ORDER if _column_owner(c) in set(ASKABLE_FIELDS)]
    y = cleaned[TARGET].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(
        features[askable], y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    ).fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_test, probability, target_recall)
    predicted = (probability >= threshold).astype(int)

    return {
        "feature_count": len(askable),
        "threshold": threshold,
        "recall": float(recall_score(y_test, predicted)),
        "precision": float(y_test[predicted == 1].sum() / max(predicted.sum(), 1)),
        "pr_auc": float(average_precision_score(y_test, probability)),
    }


def fit(
    cleaned: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_estimators: int = N_ESTIMATORS,
    target_recall: float = TARGET_RECALL,
) -> Artifact:
    """Train on a cleaned frame and return the artifact, without writing it."""
    features = to_features(cleaned)
    y = cleaned[TARGET].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise SchemaError("training needs both classes present")

    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    ).fit(x_train, y_train)

    probability = model.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_test, probability, target_recall)
    predicted = (probability >= threshold).astype(int)

    metrics: dict[str, object] = {
        "rows": int(len(cleaned)),
        "features": int(features.shape[1]),
        "positive_rate": float(y.mean()),
        "test_rows": int(len(y_test)),
        "threshold": threshold,
        "target_recall": target_recall,
        "at_threshold": {
            "recall": float(recall_score(y_test, predicted)),
            "precision": float(y_test[predicted == 1].sum() / max(predicted.sum(), 1)),
        },
        "ranking": {
            "pr_auc": float(average_precision_score(y_test, probability)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
        },
        "interview_only": _interview_only_baseline(
            cleaned,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            target_recall=target_recall,
        ),
    }

    return Artifact(
        model=model,
        feature_order=FEATURE_ORDER,
        schema_version=SCHEMA_VERSION,
        threshold=threshold,
        imputation=_imputation_values(cleaned),
        ood_bounds=_ood_bounds(cleaned),
        metrics=metrics,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def write_metrics_report(artifact: Artifact, path: Path | str = METRICS_PATH) -> Path:
    """Write the measured metrics next to the artifact, for the README."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(artifact)["metrics"], indent=2) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heartrisk.train",
        description="Train the heart risk model and write the frozen artifact.",
    )
    parser.add_argument(
        "--raw", type=Path, default=DEFAULT_RAW_PATH,
        help="raw release to read (default: data/raw/heart_disease_risk_2026.csv)",
    )
    parser.add_argument(
        "--out", type=Path, default=ARTIFACT_PATH,
        help="where to write the artifact (default: artifacts/model.joblib)",
    )
    parser.add_argument(
        "--metrics", type=Path, default=METRICS_PATH,
        help="where to write the metrics report (default: artifacts/metrics.json)",
    )
    parser.add_argument("--trees", type=int, default=N_ESTIMATORS,
                        help=f"number of trees (default: {N_ESTIMATORS})")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help=f"held-out fraction (default: {TEST_SIZE})")
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL,
                        help=f"recall the threshold is chosen for (default: {TARGET_RECALL})")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the artifact from the raw release: ``python -m heartrisk.train``."""
    args = build_parser().parse_args(argv)

    cleaned = clean(load_raw(args.raw))
    artifact = fit(
        cleaned,
        test_size=args.test_size,
        n_estimators=args.trees,
        target_recall=args.target_recall,
    )

    print(f"rows={len(cleaned)} features={artifact.metrics['features']}")
    print(f"threshold={artifact.threshold:.4f} target_recall={args.target_recall}")
    print(f"ranking={artifact.metrics['ranking']}")
    print(f"at_threshold={artifact.metrics['at_threshold']}")
    print(f"interview_only={artifact.metrics['interview_only']}")

    written = save_artifact(artifact, args.out)
    report = write_metrics_report(artifact, args.metrics)
    print(f"wrote {written}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
