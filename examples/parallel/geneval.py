from nemo_skills.pipeline.eval import eval
from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

benchmark = "icpc25"
if benchmark == "ioi25":


    eval(
            ctx=wrap_arguments(
                "++skip_filled=True "
                "++prompt_config=generic/default "
                "++inference.temperature=1.0 "
                "++inference.top_p=1.0 "
                "++inference.tokens_to_generate=120000 " 
                "++inference.extra_body.reasoning_effort=high "
                "++max_concurrent_requests=1024 "
            ),
            cluster="oci-ord-mz",
            with_sandbox=True,
            expname=f"ioi25_eval_run",
            model=f"/hf_models/gpt-oss-120b",
            server_type='vllm',
            server_gpus=8,
            num_jobs=100,
            benchmarks="ioi25:1000",
            data_dir="/workspace/llmcoding/eval_dataset/",
            output_dir="/workspace/eval/ioi25_1000",
            split="test",
            time_min="04:00:00",
            extra_eval_args=f"++eval_config.test_file=/workspace/llmcoding/eval_dataset/ioi25/test_metadata.json",
    )
elif benchmark == "ioi25_diversity":


    eval(
            ctx=wrap_arguments(
                "++skip_filled=True "
                "++prompt_config=/workspace/prompts/diversity.yaml "
                "++inference.temperature=1.0 "
                "++inference.top_p=1.0 "
                "++inference.tokens_to_generate=120000 " 
                "++inference.extra_body.reasoning_effort=high "
                "++max_concurrent_requests=1024 "
            ),
            cluster="oci-ord-mz",
            with_sandbox=True,
            expname=f"ioi25_diversity_eval_run",
            model=f"/hf_models/gpt-oss-120b",
            server_type='vllm',
            server_gpus=8,
            num_jobs=100,
            benchmarks="ioi25:1000",
            data_dir="/workspace/llmcoding/eval_dataset/",
            output_dir="/workspace/eval/ioi25_1000_diversity",
            split="test",
            time_min="04:00:00",
            extra_eval_args=f"++eval_config.test_file=/workspace/llmcoding/eval_dataset/ioi25/test_metadata.json",
    )

elif benchmark == "icpc25":
    eval(
            ctx=wrap_arguments(
                "++skip_filled=True "
                "++prompt_config=generic/default "
                "++inference.temperature=1.0 "
                "++inference.top_p=1.0 "
                "++inference.tokens_to_generate=120000 " 
                "++inference.extra_body.reasoning_effort=high "
                "++max_concurrent_requests=1024 "
            ),
            cluster="hsg",
            with_sandbox=True,
            expname=f"icpc_eval_run",
            model=f"/hf_models/gpt-oss-120b",
            server_type='vllm',
            server_gpus=4,
            num_jobs=200,
            benchmarks="icpc25:1000",
            data_dir="/workspace/llmcoding/eval_dataset/",
            output_dir="/workspace/eval/icpc_1000/",
            split="test",
            time_min="04:00:00",
            extra_eval_args=f"++eval_config.test_file=/workspace/llmcoding/eval_dataset/icpc25/test_metadata.json",
    )
elif benchmark == "icpc25_diversity":
    eval(
            ctx=wrap_arguments(
                "++skip_filled=True "
                "++prompt_config=/workspace/prompts/diversity.yaml "
                "++inference.temperature=1.0 "
                "++inference.top_p=1.0 "
                "++inference.tokens_to_generate=120000 " 
                "++inference.extra_body.reasoning_effort=high "
                "++max_concurrent_requests=1024 "
            ),
            cluster="iad",
            with_sandbox=True,
            expname=f"icpc_diversity_eval_run",
            model=f"/hf_models/gpt-oss-120b",
            server_type='vllm',
            server_gpus=8,
            num_jobs=100,
            benchmarks="icpc25:1000",
            data_dir="/workspace/llmcoding/eval_dataset/",
            output_dir="/workspace/eval/icpc_1000_diversity/",
            split="test",
            time_min="04:00:00",
            extra_eval_args=f"++eval_config.test_file=/workspace/llmcoding/eval_dataset/icpc25/test_metadata.json",
    )
else:
    raise ValueError(f"Benchmark {benchmark} not supported")