import math
import os
import string
import re
from typing import Any, Optional

import sympy

from autogen_ext.tools.mcp import StdioServerParams
from agentlightning import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger, DevTaskLoader

from octotools.solver import construct_solver
from datetime import datetime
import uuid, json

configure_logger()

calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])


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
        max_steps: int = 1,
        max_time: int = 300,
        max_tokens: int = 4096,
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
        # Agents will be initialized on the first call to their respective rollouts.
        self.training_agent = None
        self.validation_agent = None

        self.rollout_dir = "./rollout_data/calc_octo"
        self.train_rollout_dir = os.path.join(self.rollout_dir, "train")
        self.val_rollout_dir = os.path.join(self.rollout_dir, "validation")
        self.tools = ["Generalist_Solution_Generator_Tool"]
        self._solve_call_count = 0
        os.makedirs(self.train_rollout_dir, exist_ok=True)
        os.makedirs(self.val_rollout_dir, exist_ok=True)
        print(f"Rollout data will be saved to: {self.rollout_dir}")

    async def _solve_and_evaluate(self, calc_agent: OctotoolsCalcAgent, task: Any, val: bool = False):
        """A helper function to run the agent, parse the result, and evaluate it."""
    
        # self.train_batch_size = calc_agent.resources.get("data").train_batch_size
        # self.rollout_num = calc_agent.resources.get("actor_rollout_ref").rollout.n
        self.train_batch_size = 8
        self.rollout_num = 2

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

        if not val:
            self._solve_call_count += 1
            step_n = self._solve_call_count // self.train_batch_size
        else:
            step_dirs = [d for d in os.listdir(save_dir) if d.startswith("step_")]
            step_nums = []
            for d in step_dirs:
                try:
                    k = int(d.replace("step_", ""))
                    step_nums.append(k)
                except ValueError:
                    continue
            step_n = max(step_nums) + 1 if step_nums else 0

        step_dir = os.path.join(save_dir, f"step_{step_n}")
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

        

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources, val: bool = False) -> Any:
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
        
        await self._solve_and_evaluate(self.training_agent, task, val)


    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources, val: bool = True) -> Any:
        # Lazy initialization of the validation agent
        if self.validation_agent is None:
            print("Initializing validation agent...")
            llm: LLM = resources.get("main_llm")
            # For validation, we use a temperature of 0 for deterministic outputs
            self.validation_agent = get_agent(
                llm.model,
                llm.endpoint,
                temperature=0.0,
                tools = self.tools,
                resources = resources
            )
        
        await self._solve_and_evaluate(self.validation_agent, task, val)


if __name__ == "__main__":
    Trainer(n_workers=10).fit(CalcAgent(), "http://localhost:9998/")