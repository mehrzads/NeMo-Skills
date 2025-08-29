from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig

import json
from dataclasses import asdict, field, is_dataclass

from nemo_skills.utils import chunk_data, nested_dataclass, remove_thinking
from nemo_skills.utils import get_logger_name
from nemo_skills.utils import get_help_message
from nemo_skills.utils import setup_logging 
import time
import random
from dataclasses import asdict
import logging
import hydra
from nemo_skills.inference.model import server_params
import sys
import copy
from nemo_skills.evaluation.evaluator.ioi import extract_final_cpp_block
LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ParallelGenConfig(GenerateSolutionsConfig):
    """Parallel generation parameters.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    num_generations: int = 10


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_parallel_gen_config", node=ParallelGenConfig)





class ParallelGenTask(GenerationTask):
    def __init__(self, cfg: ParallelGenConfig):
        LOG.info("ParallelGenTask init")
        super().__init__(cfg)

    def load_data(self):
        data = []
        with open(self.cfg.input_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data.append(json.loads(line))
        # replicate data num_generations times
        data = [copy.deepcopy(item) for _ in range(self.cfg.num_generations) for item in data]  # deepcopy to avoid reference issues
        for i in range(len(data)):
            data[i]['unique_id'] = i
        # chunk the dataset if required
        if self.cfg.num_chunks is not None and self.cfg.chunk_id is not None:
            data, self.cfg.output_file = chunk_data(data, self.cfg.output_file, self.cfg.chunk_id, self.cfg.num_chunks)
            LOG.info(
                f"Chunking the data into {self.cfg.num_chunks} chunks and processing chunk {self.cfg.chunk_id}.\n"
                f"Number of samples in the chunk: {len(data)}"
            )

        if self.cfg.max_samples > 0:
            data = data[: self.cfg.max_samples]

        return data

    def preprocess_data(self, data):
        """A placeholder for any data preprocessing that needs to be done before generation."""
        return data


    async def process_single_datapoint(self, data_point, all_data):
        # change random seed for each generation
        self.cfg.inference.random_seed = random.randint(1, 2**32 - 1)
        # Handle inference config - check if it's a dataclass or already a dict
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            # Already a dict from Hydra
            inference_params = dict(self.cfg.inference)

        generation_params = {
            "prompt": self.fill_prompt(data_point, all_data),
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
            **inference_params,
            **self.extra_generate_params,
        }

        if self.cfg.code_execution:
            if self.cfg.override_max_code_executions and self.cfg.total_code_executions_in_prompt is not None:
                generation_params["max_code_executions"] = data_point["total_code_executions"]

        result = await self.llm.generate_async(**generation_params)
        completion = extract_final_cpp_block(result["generation"])
        print(completion)
        return result

GENERATION_TASK_CLASS = ParallelGenTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_parallel_gen_config')
def generate(cfg: ParallelGenConfig):
    cfg = ParallelGenConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = ParallelGenTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    ParallelGenConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()