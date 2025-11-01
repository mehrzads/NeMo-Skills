from nemo_skills.pipeline.eval import eval
from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments



eval(
    ctx=wrap_arguments(
            "++skip_filled=True "
            "++prompt_config=generic/default "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++inference.tokens_to_generate=120000 " 
            "++inference.extra_body.reasoning_effort=high "
            "++max_concurrent_requests=1024 "
            "++eval_config.test_file=/workspace/llmcoding/eval_dataset/icpc25/test_metadata.json"
        ),
        cluster="iad",
        with_sandbox=True,
        keep_mounts_for_sandbox=True,
        expname=f"icpc_eval_run",
        model=f"/hf_models/gpt-oss-120b",
        server_type='vllm',
        server_gpus=8,
        num_jobs=1,
        benchmarks="icpc25:1",
        data_dir="/workspace/llmcoding/eval_dataset/",
        output_dir="/workspace/gpt-oss-120b/generation/icpc25_1/",
        split="test",
        time_min="04:00:00",
        )
        