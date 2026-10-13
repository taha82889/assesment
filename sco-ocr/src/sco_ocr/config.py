"""Environment settings and layout.yaml schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LAYOUT = Path(__file__).resolve().parents[2] / "config" / "layout.yaml"
ROI_NAMES = ("title", "meta", "status_text", "items", "total", "to_pay")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    rtsp_url: str = ""
    redis_url: str = "redis://localhost:6379/0"
    sink: str = "stdout"
    fps: float = Field(3.0, gt=0, le=30)
    log_level: str = "INFO"
    ocr_device: Literal["cpu", "cuda"] = "cpu"
    crop_retention: Literal["off", "lowconf", "all"] = "off"
    capture_dir: Path = Path("captures")

    @property
    def sinks(self) -> list[str]:
        return [s.strip() for s in self.sink.split(",") if s.strip()]


class Rect(BaseModel):
    """Axis-aligned rectangle in fractions of its parent (0..1)."""

    x: float
    y: float
    w: float
    h: float

    @model_validator(mode="after")
    def _inside(self) -> Rect:
        if self.w <= 0 or self.h <= 0:
            raise ValueError("w and h must be positive")
        eps = 1e-6
        if self.x < -eps or self.y < -eps or self.x + self.w > 1 + eps or self.y + self.h > 1 + eps:
            raise ValueError(f"rect leaves its parent: {self}")
        return self


class GridCfg(BaseModel):
    cols: int = 5
    rows: int = 3
    bounds: Rect = Rect(x=0, y=0, w=1, h=1)
    unused: list[int] = [12, 13, 14]

    @field_validator("unused")
    @classmethod
    def _valid_idx(cls, v: list[int]) -> list[int]:
        if any(i < 0 for i in v):
            raise ValueError("unused indices must be >= 0")
        return v


class CropCfg(BaseModel):
    dark_threshold: float = 12.0
    min_fraction: float = 0.2  # max fraction that may be cropped per side


class ChangeCfg(BaseModel):
    stable_frames: int = 2
    thresholds: dict[str, float] = {
        "items": 4.0,
        "status_text": 4.0,
        "total": 4.0,
        "to_pay": 4.0,
        "meta": 4.0,
    }


class CorruptionCfg(BaseModel):
    blockiness_ratio: float = 1.6
    min_laplacian_var: float = 15.0
    max_dark_fraction: float = 0.3
    streak_ratio: float = 2.0


class Layout(BaseModel):
    grid: GridCfg = GridCfg()
    crop: CropCfg = CropCfg()
    rois: dict[str, Rect]
    change: ChangeCfg = ChangeCfg()
    corruption: CorruptionCfg = CorruptionCfg()
    status_vocabulary: dict[str, list[str]] = {}

    @field_validator("rois")
    @classmethod
    def _known(cls, v: dict[str, Rect]) -> dict[str, Rect]:
        missing = set(ROI_NAMES) - set(v)
        if missing:
            raise ValueError(f"missing rois: {sorted(missing)}")
        return v

    @model_validator(mode="after")
    def _unused_in_grid(self) -> Layout:
        n = self.grid.cols * self.grid.rows
        if any(i >= n for i in self.grid.unused):
            raise ValueError("unused index outside grid")
        return self


def load_layout(path: Path | str = DEFAULT_LAYOUT) -> Layout:
    with open(path, encoding="utf-8") as f:
        return Layout.model_validate(yaml.safe_load(f))


def save_layout(layout: Layout, path: Path | str = DEFAULT_LAYOUT) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(layout.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
