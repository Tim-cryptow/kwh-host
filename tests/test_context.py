"""D8: a host certifies and serves at one context length, 1024 to 8192, default 8192."""

import json

from click.testing import CliRunner
from kwh_bench import reference as ref

from kwh_host.bench import context_mismatch
from kwh_host.cli import main
from kwh_host.config import HostConfig
from kwh_host.engine import make_engine


def vllm_report(mml):
    return {"engine": {"name": "vllm", "launch_args": ["vllm", "serve", ref.MODEL_ID, "--max-model-len", str(mml)]}}


def test_config_defaults_to_8192_and_the_engine_launches_with_it(home):
    cfg = HostConfig(engine_mode="bare-metal")
    assert cfg.max_model_len == 8192
    engine = make_engine(cfg)
    assert engine.max_model_len == 8192
    assert engine._server_args()[engine._server_args().index("--max-model-len") + 1] == "8192"


def test_serving_context_must_match_the_certified_one():
    assert context_mismatch(vllm_report(8192), 8192) is None
    problem = context_mismatch(vllm_report(1024), 8192)
    assert problem and "certified at --max-model-len 1024" in problem and "kwh-host bench" in problem
    assert context_mismatch({"engine": {"name": "mock", "launch_args": ["step_ms=0.05"]}}, 8192) is None


def test_init_takes_the_context_length_within_the_certified_range(home):
    runner = CliRunner()
    out = runner.invoke(main, ["init", "--engine", "bare-metal", "--allow-bare-metal", "--max-model-len", "4096"])
    assert out.exit_code == 0, out.output
    assert json.loads(out.output)["max_model_len"] == 4096 == HostConfig.load().max_model_len
    bad = runner.invoke(main, ["init", "--engine", "bare-metal", "--allow-bare-metal", "--max-model-len", "16384"])
    assert bad.exit_code != 0 and "16384" in bad.output
