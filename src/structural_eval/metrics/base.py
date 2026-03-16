from __future__ import annotations
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """구조 보존 지표의 추상 베이스 클래스.

    서브클래스는 반드시 ``compute``를 구현해야 한다.
    ``compute``는 0~1 범위의 float를 반환해야 한다.
    """

    @abstractmethod
    def compute(self, source: str, target: str) -> float:
        """원문(source)과 번안문(target)을 비교해 0~1 점수를 반환한다.

        Args:
            source: 영어 원문
            target: 번안문 (쉬운 영어)

        Returns:
            0.0 (구조 완전 소실) ~ 1.0 (구조 완전 보존)
        """
        ...

    def safe_compute(self, source: str, target: str) -> float:
        """compute를 실행하고 결과가 [0, 1] 범위임을 보장한다."""
        result = self.compute(source, target)
        if not isinstance(result, float):
            result = float(result)
        return max(0.0, min(1.0, result))
