import math
import os
import string
import re
from typing import Any, Optional

import sympy

import yaml
import argparse

from autogen_ext.tools.mcp import StdioServerParams
from agentlightning import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger, DevTaskLoader

from octotools.solver import construct_solver
from datetime import datetime
import uuid, json
from filelock import FileLock
import asyncio

configure_logger()

calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])

# TODO: clean up the verbose check_model, namedsource

# Copied and adapted from https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py


def normalize_option(option: str) -> str:
    """
    >>> normalize_option("  (A)  \n")
    'A'
    """
    return re.sub(r"(\s+|\(|\))", "", option)


def is_option_result(result: str) -> bool:
    """
    >>> is_option_result("  A)  \n")
    True
    >>> is_option_result("  23/7 ")
    False
    """
    return normalize_option(result) in list(string.ascii_letters)


def float_eval(input_str: str) -> float:
    if " = around " in input_str:
        input_str = input_str.split(" = around ")[0]
    expr = sympy.parse_expr(input_str, evaluate=True)
    return float(expr.evalf())


def scalar_are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    pred_result = str(pred_result) if pred_result is not None else ""
    true_result = str(true_result) if true_result is not None else ""

    if pred_result.strip() == true_result.strip():
        return True

    if is_option_result(true_result):
        # The task is to select correct option
        true_result = normalize_option(true_result)
        pred_result = normalize_option(pred_result)
        return pred_result == true_result

    # The task is to calculate the result as a number
    try:
        pred_float = float_eval(pred_result)
        true_float = float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except Exception:
        pass

    return False


@reward
async def eval(prediction: str, ground_truth: str, val: bool = False) -> float:
    """
    Calculates the reward using a subtractive penalty for repetition,
    which applies regardless of the answer's correctness.
    """

    base_reward = float(scalar_are_results_same(prediction, ground_truth, 1e-2))

    # if not val:
    #     words = re.findall(r'\w+', str(prediction).lower())
        
    #     repetition_cost = 0.0
    #     if len(words) > 0:
    #         unique_words_ratio = len(set(words)) / len(words)
    #         repetition_cost = 1.0 - unique_words_ratio
    #     penalty_magnitude = 0.25
    #     final_reward = base_reward - (penalty_magnitude * repetition_cost)
    # else:
    #     final_reward = base_reward
    final_reward = base_reward
    
    return final_reward


class OctotoolsCalcAgent:
    def __init__(
        self,
        resources: NamedResources,
        llm_engine_name: str = "gpt-4o",
        enabled_tools: list[str] = ["Generalist_Solution_Generator_Tool"],
        output_types: str = "final,direct",
        max_steps: int = 3, # TODO: update max_steps
        max_time: int = 300,
        max_tokens: int = 2048,
        base_url="http://localhost:8888",
        check_model=False,  # TODO: delete check_model
        verbose: bool = True,
        temperature: float = 0.0,
    ):
        self.resources = resources
        self.llm_engine = llm_engine_name
        prefix = "" if "gpt" in llm_engine_name else "vllm-"
        self.solver = construct_solver(
            llm_engine_name=prefix + llm_engine_name,
            enabled_tools=enabled_tools,
            output_types=output_types,
            max_steps=max_steps,
            max_time=max_time,
            max_tokens=max_tokens,
            base_url=base_url,
            check_model=check_model,
            verbose=verbose,
            temperature = temperature
        )
        self.verbose = verbose

    def solve(self, question: str, image_path: Optional[str] = None) -> dict:
        result = self.solver.solve(question, image_path)
        if self.verbose:
            print(f"\n==> 📝 Solver Result:")
            print(f"""
            *******************************
            RESULT
            {result}
            RESULT
            *******************************
            """)

        return result


def get_agent(model, openai_base_url, temperature, resources: NamedResources, tools = ["Generalist_Solution_Generator_Tool"]):
    llm_engine_name = model
    if openai_base_url and openai_base_url != "https://api.openai.com/v1":
        vllm_base_url = openai_base_url
    else:
        vllm_base_url = None

    agent = OctotoolsCalcAgent(
        resources = resources,
        llm_engine_name=llm_engine_name,
        enabled_tools=tools,
        verbose=True,
        base_url=vllm_base_url,
        check_model=False,
        temperature = temperature,
    )
    return agent


class CalcAgent(LitAgent):

    def __init__(self):
        super().__init__()
        # Agents will be initialized on the first call
        self.training_agent = None
        self.validation_agent = None
        self.val_step_n = None
        
        path_config = config.get("paths", {})
        agent_specific_config = config.get("agent_config", {})
        data_config = config.get("data", {})

        self.base_rollout_dir = path_config.get("base_output_dir", "./rollout_data") # e.g., "./out"
        self.run_folder_format = path_config.get("run_folder_format", "{model_name}_{timestamp}")

        self.tools = agent_specific_config.get("tools", ["Generalist_Solution_Generator_Tool"])
        self.train_batch_size = data_config.get("train_batch_size", 8)
        self.rollout_num = agent_specific_config.get("rollout_num", 4)
        
        print(f"Agent configured with: base_output_dir='{self.base_rollout_dir}', train_batch_size={self.train_batch_size}, rollout_num={self.rollout_num}")

        # 以下属性将在 _initialize_run_once 中被动态设置
        self.rollout_dir = None
        self.train_rollout_dir = None
        self.val_rollout_dir = None
        self.train_lock_file = None
        self.val_lock_file = None

        # 基于从配置中读取的基础目录来定义同步文件路径
        self.run_info_file = os.path.join(self.base_rollout_dir, ".run_info")
        self.init_lock_file = os.path.join(self.base_rollout_dir, ".init.lock")
        

    async def _solve_and_evaluate(self, calc_agent: OctotoolsCalcAgent, task: Any, step_n: int, val: bool = False):
        """A helper function to run the agent, parse the result, and evaluate it."""
    
        # self.train_batch_size = calc_agent.resources.get("data").train_batch_size
        # self.rollout_num = calc_agent.resources.get("actor_rollout_ref").rollout.n

        try:
            output_format = "Output the answer when you are ready. The answer should be surrounded by <answer>...</answer>. DO NOT generate after the </answer> tag."
            prompt = task["question"] + " " + output_format
            result = calc_agent.solve(question=prompt)
            
            # Safely check for and extract the final answer
            if "final_output" in result and result["final_output"]:
                final_output = result["final_output"]
                all_matches = re.findall(r"<answer>(.*?)</answer>", final_output, re.DOTALL)
                if all_matches:
                    answer = all_matches[-1].strip()
                else:
                    answer = final_output
            else:
                print("Warning: Result has no final_output or final_output is empty.")
                answer = "None"
        except Exception as e:
            print(f"Failure during agent execution: {str(e)}. Defaulting to 'None'.")
            answer = "None"

        # Evaluate the answer against the ground truth
        reward_value = await eval(answer, str(task["result"]), val)  # reward is tracked with the decorator
        print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward_value))

        idx = task.get("extra_info", {}).get("idx", "unknown_idx")

        # TODO: update saving logic
        rollout_data = {
            "idx": idx,
            "id": task.get("id", ""),
            "prompt": task["question"],
            "model":calc_agent.llm_engine,
            "tools":self.tools,
            "groundtruth": task.get("extra_info", {}).get("groundtruth", task["result"]),
            "answer_extracted": answer,
            "total_result":result,
            "reward": reward_value,
            "timestamp": datetime.now().isoformat(),
        }

        data_id = str(uuid.uuid4())
        filename = f"rollout_{data_id}.json"

        save_dir = self.val_rollout_dir if val else self.train_rollout_dir

        # --- MODIFICATION START ---
        # The logic for determining step_n is removed from here and moved to the calling functions.
        # This function now uses the `step_n` passed as an argument.
        step_dir = os.path.join(save_dir, f"step_{step_n}")
        # --- MODIFICATION END ---
        
        idx_dir = os.path.join(step_dir, f"idx_{idx}")
        os.makedirs(idx_dir, exist_ok=True)

        json_count = sum(
            len([f for f in files if f.endswith(".json")])
            for root, dirs, files in os.walk(idx_dir)
        )
        assert json_count < self.rollout_num, \
            f"Too many rollouts for idx {idx}: already {json_count} >= {self.rollout_num}"

        save_path = os.path.join(idx_dir, filename)

        with open(save_path, "w") as f:
            json.dump(rollout_data, f, indent=2)

        print(f"Rollout data saved to: {save_path}")


    async def _initialize_run_once(self, resources: NamedResources):
        """
        Ensures that the rollout directory is set up only once per run,
        in a process-safe way.
        """
        if self.rollout_dir is not None:
            return

        os.makedirs(self.base_rollout_dir, exist_ok=True)
        
        init_lock = FileLock(self.init_lock_file, timeout=60)
        with init_lock:
            if os.path.exists(self.run_info_file):
                with open(self.run_info_file, 'r') as f:
                    final_rollout_dir = f.read().strip()
            else:
                model_name_full = resources.get("main_llm").model
                model_name_simple = model_name_full.split('/')[-1]
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                run_folder_name = self.run_folder_format.format(
                    model_name=model_name_simple,
                    timestamp=timestamp
                )
                
                final_rollout_dir = os.path.join(self.base_rollout_dir, run_folder_name)
                
                with open(self.run_info_file, 'w') as f:
                    f.write(final_rollout_dir)
                print(f"Run directory created by process {os.getpid()}: {final_rollout_dir}")

        self.rollout_dir = final_rollout_dir
        self.train_rollout_dir = os.path.join(self.rollout_dir, "train")
        self.val_rollout_dir = os.path.join(self.rollout_dir, "validation")
        
        os.makedirs(self.train_rollout_dir, exist_ok=True)
        os.makedirs(self.val_rollout_dir, exist_ok=True)
        
        self.train_lock_file = os.path.join(self.train_rollout_dir, ".train.lock")
        self.val_lock_file = os.path.join(self.val_rollout_dir, ".val.lock")
        
    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources, val: bool = False) -> Any:
        await self._initialize_run_once(resources)


        # Lazy initialization of the training agent
        if self.training_agent is None:
            print("Initializing training agent...")
            llm: LLM = resources.get("main_llm")
            self.training_agent = get_agent(
                llm.model,
                llm.endpoint,
                llm.sampling_parameters.get("temperature", 0.7),
                tools = self.tools,
                resources = resources
            )
        
        # filelock to determine step_n ---
        lock = FileLock(self.train_lock_file)
        with lock:
            step_dirs = [d for d in os.listdir(self.train_rollout_dir) if d.startswith("step_")]
            step_nums = [int(d.replace("step_", "")) for d in step_dirs if d.replace("step_", "").isdigit()]
            
            current_step_n = 0
            if step_nums:
                current_step_n = max(step_nums)

            current_step_dir = os.path.join(self.train_rollout_dir, f"step_{current_step_n}")
            if os.path.exists(current_step_dir):
                num_items_in_step = len(os.listdir(current_step_dir))
                if num_items_in_step >= self.train_batch_size:
                    current_step_n += 1
            
            step_n = current_step_n

        await self._solve_and_evaluate(self.training_agent, task, step_n, val)



    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources, val: bool = True) -> Any:
        await self._initialize_run_once(resources)

        # Lazy initialization of the agent and one-time determination of the validation step number.
        # This lock ensures that only the first validation task of a run calculates the step number,
        # preventing the creation of thousands of folders.
        val_lock = FileLock(self.val_lock_file, timeout=30)
        with val_lock:
            if self.validation_agent is None:
                print("Initializing validation agent and determining validation step...")
                llm: LLM = resources.get("main_llm")
                self.validation_agent = get_agent(
                    llm.model,
                    llm.endpoint,
                    temperature=0.0,
                    tools = self.tools,
                    resources = resources
                )

            print(f"Scanning '{self.train_rollout_dir}' to find current training step...")
            train_step_dirs = [d for d in os.listdir(self.train_rollout_dir) if d.startswith("step_")]
            train_step_nums = [int(d.replace("step_", "")) for d in train_step_dirs if d.replace("step_", "").isdigit()]
            
            current_train_step = max(train_step_nums) if train_step_nums else 0
            self.val_step_n = current_train_step
            print(f"Validation run started. Synchronizing with training progress. Saving results to validation step folder: {self.val_step_n}")

        await self._solve_and_evaluate(self.validation_agent, task, self.val_step_n, val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to the configuration YAML file.")
    args, unknown = parser.parse_known_args()

    print(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    agent = CalcAgent(config=config)

    port = config.get("verl_params", {}).get("port", 9998)
    
    # 2. 动态构建 Trainer 的连接地址
    trainer_endpoint = f"http://localhost:{port}"
    print(f"Connecting Trainer to endpoint: {trainer_endpoint}")

    Trainer(n_workers=10).fit(agent, trainer_endpoint)