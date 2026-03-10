# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
from pathlib import Path

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.icpc import ICPCEvaluator
from nemo_skills.evaluation.evaluator.ioi import IOIEvaluator
from nemo_skills.utils import nested_dataclass


@nested_dataclass(kw_only=True)
class CCCEvaluatorConfig(BaseEvaluatorConfig):
    test_file: str = "test_metadata.json"
    test_batch_size: int = 16
    time_scale: float = 1.0
    benchmark_type: str | None = None


class CCCEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        self.eval_cfg = CCCEvaluatorConfig(_init_nested=True, **config)
        self.benchmark_type = self.eval_cfg.benchmark_type or self._detect_benchmark_type(self.eval_cfg.test_file)
        if self.benchmark_type not in {"ioi", "icpc"}:
            raise ValueError(
                f"Unsupported ccc benchmark_type={self.benchmark_type!r}. Expected 'ioi' or 'icpc'."
            )
        delegate_cls = IOIEvaluator if self.benchmark_type == "ioi" else ICPCEvaluator
        self.delegate = delegate_cls(config, num_parallel_requests)

    @staticmethod
    def _detect_benchmark_type(test_file: str) -> str:
        path = Path(test_file)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file {test_file} does not exist.")

        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not metadata:
            raise ValueError(f"Metadata file {test_file} is empty.")

        first_value = next(iter(metadata.values()))
        if isinstance(first_value, dict):
            if {"compile", "run", "tests", "sample_tests"}.issubset(first_value.keys()):
                return "icpc"
            nested = next(iter(first_value.values()), None)
            if isinstance(nested, dict) and {"compile", "run", "tests"}.issubset(nested.keys()):
                return "ioi"

        raise ValueError(
            f"Could not infer benchmark type from metadata file {test_file}. "
            "Please set eval_config.benchmark_type to 'ioi' or 'icpc'."
        )

    async def eval_full(self, input_files=None):  # type: ignore[override]
        if input_files is None:
            return await self.delegate.eval_full()
        return await self.delegate.eval_full(input_files)

    async def eval_single(self, data_point: dict):
        result = await self.delegate.eval_single(data_point)
        result["ccc_benchmark_type"] = self.benchmark_type
        return result

    def supports_single_eval(self) -> bool:
        return self.delegate.supports_single_eval()
