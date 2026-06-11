"""Feature preprocessing implemented from first principles."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import as_2d_features


@dataclass
class StandardScalerScratch:
    """Z-score scaler that stores train-set statistics only."""

    mean_: NDArray[np.float64] | None = None
    scale_: NDArray[np.float64] | None = None

    def fit(self, x: ArrayLike) -> StandardScalerScratch:
        x_arr = as_2d_features(x)
        self.mean_ = x_arr.mean(axis=0)
        scale = x_arr.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before transform.")
        x_arr = as_2d_features(x)
        return (x_arr - self.mean_) / self.scale_

    def fit_transform(self, x: ArrayLike) -> NDArray[np.float64]:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict[str, list[float]]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cannot serialize an unfitted scaler.")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> StandardScalerScratch:
        return cls(
            mean_=np.asarray(payload["mean"], dtype=float),
            scale_=np.asarray(payload["scale"], dtype=float),
        )
