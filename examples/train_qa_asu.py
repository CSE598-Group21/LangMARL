#!/usr/bin/env python3
#"""Small test run: 2-agent QA on ASU models."""
  
import os
import langmarl
from langmarl.config.llm import LLMConfig

BASE_URL = os.environ["OPENAI_BASE_URL"]
API_KEY = os.environ["OPENAI_API_KEY"]

actor_llm = LLMConfig(
    name="gemma4-31b-it",
    model_string="gemma4-31b-it",
    base_url=BASE_URL,
    api_key=API_KEY,
    max_tokens=4096,
)

critic_llm = LLMConfig(
    name="qwen38-27b",
    model_string="qwen38-27b",
    base_url=BASE_URL,
    api_key=API_KEY,
    max_tokens=8192,
)

HERE = os.path.dirname(__file__)

config = langmarl.LanguageTaskConfig(
    exp_name="qa_2agent_asu_test",
    paradigm="central_credit",
    task_type="qa",
    benchmark_path=os.path.join(HERE, "..", "env", "lang_benchmark", "HotPotQA"),
    data_limit=50,
    num_agents=2,
    num_iterations=4,
    trajectories_per_iteration=50,
    actor_llm=actor_llm,
    critic_llm=critic_llm,
    optimizer_llm=critic_llm,
    experiment_dir=os.path.join(HERE, "..", "experiments"),
    checkpoint_dir=os.path.join(HERE, "..", "experiments", "ckpt_qa_asu_test"),
    episode_generation_workers=2,
    optimizer_workers=2,
    max_workers=4,
    log_level="INFO",
)
config.judge_model = "qwen38-27b"

env = langmarl.make_env("language", config)
critic = langmarl.CentralizedCritic(config)
optimizer = langmarl.PolicyGradientOptimizer(config.get_optimizer_llm())

trainer = langmarl.MonteCarloTrainer(
    config=config,
    env=env,
    critic=critic,
    optimizer=optimizer,
    callbacks=[langmarl.LoggingCallback(), langmarl.CheckpointCallback()],
)

if __name__ == "__main__":
    metrics = trainer.train()
    print("\n--- Training curve ---")
    for m in metrics or []:
        if m.get("type") == "iteration":
            print(f"  Iteration {m['iteration']}: avg_reward={m.get('avg_reward', 0):.3f}")
