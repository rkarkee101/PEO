from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ProcessDataset:
    df: pd.DataFrame
    tool_params: List[str]
    target_props: List[str]
    categorical_params: List[str]
    metadata: Dict[str, Any]


class DataLoader:
    """Read, validate, and preprocess process data.

    Preprocessing is done via a sklearn ColumnTransformer so the same transform
    can be reused later (inverse design, querying).
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def load_csv(self, path: str | Path) -> ProcessDataset:
        data_cfg = self.cfg["data"]
        p = Path(path)

        df = pd.read_csv(
            p,
            delimiter=data_cfg.get("delimiter", ","),
            encoding=data_cfg.get("encoding", "utf-8"),
        )

        tool_params = list(data_cfg["tool_parameters"])
        target_props = list(data_cfg["target_properties"])

        required_cols = set(tool_params + target_props)
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        cat_cfg = data_cfg.get("categorical_params", {}) or {}
        if isinstance(cat_cfg, list):
            cat_cfg = {str(k): [] for k in cat_cfg}
        categorical_params = sorted(set(cat_cfg.keys()))

        meta = {
            "n_samples": int(len(df)),
            "n_tool_params": int(len(tool_params)),
            "n_targets": int(len(target_props)),
            "source": str(p),
        }

        return ProcessDataset(
            df=df,
            tool_params=tool_params,
            target_props=target_props,
            categorical_params=categorical_params,
            metadata=meta,
        )

    def build_preprocessor(self, dataset: ProcessDataset) -> Tuple[ColumnTransformer, List[str]]:
        """Build and fit a reusable sklearn preprocessor."""
        cat_cols = [c for c in dataset.tool_params if c in dataset.categorical_params]
        num_cols = [c for c in dataset.tool_params if c not in dataset.categorical_params]

        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

        pre = ColumnTransformer(transformers=transformers, remainder="drop")
        pre.fit(dataset.df)

        feature_names: List[str] = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            ohe: OneHotEncoder = pre.named_transformers_["cat"]
            feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

        return pre, feature_names

    def to_xy(self, dataset: ProcessDataset, preprocessor: ColumnTransformer) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        X = preprocessor.transform(dataset.df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)

        ys: Dict[str, np.ndarray] = {}
        for t in dataset.target_props:
            ys[t] = np.asarray(dataset.df[t].values, dtype=float)

        return X, ys
