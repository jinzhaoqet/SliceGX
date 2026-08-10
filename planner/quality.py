from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class QualityCalibrationPoint:
    sample_ratio: float
    observed_max_error: float
    observed_confidence: float
    observations: int = 1

    def __post_init__(self):
        if not 0.0 < self.sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be within (0, 1].")
        if not 0.0 <= self.observed_max_error <= 1.0:
            raise ValueError("observed_max_error must be within [0, 1].")
        if not 0.0 <= self.observed_confidence <= 1.0:
            raise ValueError("observed_confidence must be within [0, 1].")
        if self.observations <= 0:
            raise ValueError("observations must be positive.")


class ApproximationQualityProfile:
    def __init__(self, points: Iterable[QualityCalibrationPoint] = ()):
        self.points: List[QualityCalibrationPoint] = sorted(points, key=lambda point: point.sample_ratio)

    def select(
        self,
        max_error: Optional[float],
        min_confidence: Optional[float],
        minimum_ratio: float = 0.05,
    ) -> Optional[QualityCalibrationPoint]:
        candidates = [
            point
            for point in self.points
            if point.sample_ratio >= minimum_ratio
            and (max_error is None or point.observed_max_error <= max_error)
            and (min_confidence is None or point.observed_confidence >= min_confidence)
        ]
        return min(candidates, key=lambda point: point.sample_ratio) if candidates else None

    def add(self, point: QualityCalibrationPoint) -> None:
        self.points.append(point)
        self.points.sort(key=lambda item: item.sample_ratio)
