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
    """Read, validate, and preprocess process data."""

    def __init__(self, delimiter: str = ",", encoding: str = "utf-8"):
        self.delimiter = delimiter
        self.encoding = encoding

    def load_csv(
        self,
        csv_path: str | Path,
        tool_params: List[str],
        target_props: List[str],
        categorical_params: List[str] | None = None,
        units: Dict[str, str] | None = None,
    ) -> ProcessDataset:
        p = Path(csv_path)
        df = pd.read_csv(p, sep=self.delimiter, encoding=self.encoding)

        tool_params = [str(x) for x in tool_params]
        target_props = [str(x) for x in target_props]
        categorical_params = [str(x) for x in (categorical_params or [])]

        required = set(tool_params + target_props)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Normalize categorical columns to strings (for stable encoding)
        for c in categorical_params:
            if c in df.columns:
                df[c] = df[c].astype(str)

        meta = {"csv": str(p.resolve()), "units": dict(units or {})}

        return ProcessDataset(
            df=df,
            tool_params=tool_params,
            target_props=target_props,
            categorical_params=categorical_params,
            metadata=meta,
        )

    def build_preprocessor(
        self,
        dataset: ProcessDataset,
        *,
        fit_df: pd.DataFrame | None = None,
        categorical_levels: Dict[str, List[str]] | None = None,
    ) -> Tuple[ColumnTransformer, List[str]]:
        """Build and fit a reusable sklearn preprocessor.

        Important:
        - To avoid train/test leakage, callers may pass fit_df (typically the training split).
        - For categorical columns, callers may provide categorical_levels (schema) to ensure
          a stable one-hot encoding across splits and iterations (active learning).
        """
        cat_cols = [c for c in dataset.tool_params if c in dataset.categorical_params]
        num_cols = [c for c in dataset.tool_params if c not in dataset.categorical_params]

        df_fit = fit_df if fit_df is not None else dataset.df

        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            levels = categorical_levels or {}
            cats = []
            for c in cat_cols:
                lv = levels.get(c)
                if lv is None or len(lv) == 0:
                    # Infer levels from the full dataset (schema inference).
                    # This is not a distributional leak; it only stabilizes the encoding.
                    lv = sorted(list(pd.Series(dataset.df[c].dropna().astype(str)).unique()))
                cats.append(list(map(str, lv)))

            # sklearn parameter name changed from 'sparse' -> 'sparse_output' in newer versions.
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", categories=cats, sparse_output=True)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", categories=cats, sparse=True)

            transformers.append(("cat", ohe, cat_cols))

        pre = ColumnTransformer(transformers=transformers, remainder="drop")
        pre.fit(df_fit[dataset.tool_params])

        feature_names: List[str] = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            ohe: OneHotEncoder = pre.named_transformers_["cat"]
            feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

        return pre, feature_names

    def to_xy(self, dataset: ProcessDataset, preprocessor: ColumnTransformer) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        X = preprocessor.transform(dataset.df[dataset.tool_params])
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)

        ys: Dict[str, np.ndarray] = {}
        for t in dataset.target_props:
            ys[str(t)] = np.asarray(dataset.df[t], dtype=float).ravel()
        return X, ys
