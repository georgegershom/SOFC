from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

EDGE_LEFT = 0
EDGE_RIGHT = 1
EDGE_BOTTOM = 2
EDGE_TOP = 3

EDGE_ID_TO_NAME = {EDGE_LEFT: "left", EDGE_RIGHT: "right", EDGE_BOTTOM: "bottom", EDGE_TOP: "top"}
EDGE_NAME_TO_ID = {v: k for k, v in EDGE_ID_TO_NAME.items()}


@dataclass
class Domain2D:
    """二维矩形域，支持时间维与反应区掩码。"""

    length_x: float
    length_y: float
    t_final: float
    include_time: bool
    reactive_x_min: Optional[float] = None
    reactive_x_max: Optional[float] = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        self.random_generator: np.random.Generator = np.random.default_rng(self.random_seed)

    def is_reactive_mask(self, x: np.ndarray) -> np.ndarray:
        if self.reactive_x_min is None or self.reactive_x_max is None:
            return np.zeros_like(x, dtype=bool)
        return (x >= float(self.reactive_x_min)) & (x <= float(self.reactive_x_max))

    def sample_interior(self, num_points: int) -> Dict[str, np.ndarray]:
        rng = self.random_generator
        x = rng.uniform(0.0, self.length_x, size=num_points)
        y = rng.uniform(0.0, self.length_y, size=num_points)
        if self.include_time and self.t_final > 0.0:
            t = rng.uniform(0.0, self.t_final, size=num_points)
        else:
            t = np.zeros(num_points, dtype=np.float64)
        is_reactive = self.is_reactive_mask(x)
        return {
            "x": x.astype(np.float64),
            "y": y.astype(np.float64),
            "t": t.astype(np.float64),
            "is_reactive": is_reactive.astype(np.int32),
        }

    def _sample_edge_param(self, n: int) -> np.ndarray:
        return self.random_generator.uniform(0.0, 1.0, size=n)

    def _edge_to_xy(self, edge_id: int, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if edge_id == EDGE_LEFT:
            x = np.zeros_like(s)
            y = s * self.length_y
        elif edge_id == EDGE_RIGHT:
            x = np.full_like(s, self.length_x)
            y = s * self.length_y
        elif edge_id == EDGE_BOTTOM:
            x = s * self.length_x
            y = np.zeros_like(s)
        elif edge_id == EDGE_TOP:
            x = s * self.length_x
            y = np.full_like(s, self.length_y)
        else:
            raise ValueError(f"Invalid edge_id: {edge_id}")
        return x, y

    def _edge_normal(self, edge_id: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if edge_id == EDGE_LEFT:
            nx = -np.ones(n)
            ny = np.zeros(n)
        elif edge_id == EDGE_RIGHT:
            nx = np.ones(n)
            ny = np.zeros(n)
        elif edge_id == EDGE_BOTTOM:
            nx = np.zeros(n)
            ny = -np.ones(n)
        elif edge_id == EDGE_TOP:
            nx = np.zeros(n)
            ny = np.ones(n)
        else:
            raise ValueError(f"Invalid edge_id: {edge_id}")
        return nx.astype(np.float64), ny.astype(np.float64)

    def sample_boundary(self, num_per_edge: int) -> Dict[str, np.ndarray]:
        xs, ys, ts, nxs, nys, edge_ids = [], [], [], [], [], []
        for edge_id in (EDGE_LEFT, EDGE_RIGHT, EDGE_BOTTOM, EDGE_TOP):
            s = self._sample_edge_param(num_per_edge)
            x, y = self._edge_to_xy(edge_id, s)
            nx, ny = self._edge_normal(edge_id, num_per_edge)
            if self.include_time and self.t_final > 0.0:
                t = self.random_generator.uniform(0.0, self.t_final, size=num_per_edge)
            else:
                t = np.zeros(num_per_edge, dtype=np.float64)
            xs.append(x); ys.append(y); ts.append(t)
            nxs.append(nx); nys.append(ny)
            edge_ids.append(np.full(num_per_edge, edge_id, dtype=np.int32))
        return {
            "x": np.concatenate(xs).astype(np.float64),
            "y": np.concatenate(ys).astype(np.float64),
            "t": np.concatenate(ts).astype(np.float64),
            "nx": np.concatenate(nxs).astype(np.float64),
            "ny": np.concatenate(nys).astype(np.float64),
            "edge_id": np.concatenate(edge_ids).astype(np.int32),
        }
