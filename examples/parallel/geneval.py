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
        ),
        cluster="oci-ord-mz",
        with_sandbox=True,
        expname=f"ioi25_eval_run",
        model=f"/hf_models/gpt-oss-120b",
        server_type='vllm',
        server_gpus=8,
        num_jobs=1,
        benchmarks="ioi25:1",
        data_dir="/workspace/llmcoding/eval_dataset/",
        output_dir="/workspace/test/",
        split="test",
        time_min="00:30:00",
        extra_eval_args=f"++eval_config.test_file=/workspace/llmcoding/eval_dataset/ioi25/test_metadata.json",
)