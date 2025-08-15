import math
import os
import string
import re
from typing import Any, Optional

import sympy

from autogen_ext.tools.mcp import StdioServerParams
from agentlightning import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger, DevTaskLoader

from octotools.solver import construct_solver

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
async def eval(prediction: str, ground_truth: str) -> float:
    return float(scalar_are_results_same(prediction, ground_truth, 1e-2))


class OctotoolsCalcAgent:
    def __init__(
        self,
        llm_engine_name: str = "gpt-4o",
        enabled_tools: list[str] = ["Generalist_Solution_Generator_Tool"],
        output_types: str = "final,direct",
        max_steps: int = 1,
        max_time: int = 300,
        max_tokens: int = 3072,
        base_url="http://localhost:8888",
        check_model=False,  # TODO: delete check_model
        verbose: bool = True,
        temperature: float = 0.0
    ):
        self.solver = construct_solver(
            llm_engine_name="vllm-" + llm_engine_name,
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


def get_agent(model, openai_base_url, temperature):
    llm_engine_name = model
    if openai_base_url and openai_base_url != "https://api.openai.com/v1":
        vllm_base_url = openai_base_url
    else:
        vllm_base_url = None

    agent = OctotoolsCalcAgent(
        llm_engine_name=llm_engine_name,
        enabled_tools=["Generalist_Solution_Generator_Tool"],
        verbose=True,
        base_url=vllm_base_url,
        check_model=False,
        temperature = temperature
    )
    return agent


class CalcAgent(LitAgent):

    def __init__(self):
        super().__init__()
        # Agents will be initialized on the first call to their respective rollouts.
        self.training_agent = None
        self.validation_agent = None

    async def _solve_and_evaluate(self, calc_agent: OctotoolsCalcAgent, task: Any):
        """A helper function to run the agent, parse the result, and evaluate it."""
        try:
            output_format = "Output the answer when you are ready. The answer should be surrounded by three sharps (`###`), in the form of ### ANSWER: [SILENT] ###."
            prompt = task["question"] + " " + output_format
            result = calc_agent.solve(question=prompt)
            
            # Safely check for and extract the final answer
            if "final_output" in result and result["final_output"]:
                final_output = result["final_output"]
                answer_match = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", final_output)
                if answer_match:
                    answer = answer_match.group(1)
                else:
                    answer = final_output
            else:
                print("Warning: Result has no final_output or final_output is empty.")
                answer = "None"
        except Exception as e:
            print(f"Failure during agent execution: {str(e)}. Defaulting to 'None'.")
            answer = "None"

        # Evaluate the answer against the ground truth
        reward_value = await eval(answer, str(task["result"]))  # reward is tracked with the decorator
        print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward_value))


    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        # Lazy initialization of the training agent
        if self.training_agent is None:
            print("Initializing training agent...")
            llm: LLM = resources.get("main_llm")
            self.training_agent = get_agent(
                llm.model,
                llm.endpoint,
                llm.sampling_parameters.get("temperature", 0.7),
            )
        
        await self._solve_and_evaluate(self.training_agent, task)


    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        # Lazy initialization of the validation agent
        if self.validation_agent is None:
            print("Initializing validation agent...")
            llm: LLM = resources.get("main_llm")
            # For validation, we use a temperature of 0 for deterministic outputs
            self.validation_agent = get_agent(
                llm.model,
                llm.endpoint,
                temperature=0.0,
            )
        
        await self._solve_and_evaluate(self.validation_agent, task)


if __name__ == "__main__":
    Trainer(n_workers=10).fit(CalcAgent(), "http://localhost:9999/")