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
from filelock import FileLock
import asyncio

configure_logger()

calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])

# TODO: clean up the verbose check_model, namedsource

# Copied and adapted from https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py

import re
import ast
import builtins


@reward
async def eval(groundtruth, answer_extracted, val: bool = False) -> float:
    def extract_numbers(gt_list):
        for expr in gt_list:
            expr = expr.replace('\u00d7', '*').replace('×', '*').replace('−', '-').replace(' ', '')
            nums = re.findall(r'\d+', expr)
            if len(nums) == 4:
                return sorted(int(x) for x in nums)
        return None

    target_nums = extract_numbers(groundtruth)
    if target_nums is None:
        return 0.0

    expr = str(answer_extracted).strip()
    if not expr:
        return 0.0

    expr_clean = expr.replace('\u00d7', '*').replace('×', '*').replace('−', '-').replace(' ', '')

    used_strs = re.findall(r'\d+', expr_clean)
    if len(used_strs) != 4:
        return 0.0
    try:
        used_nums = sorted(int(x) for x in used_strs)
    except:
        return 0.0

    if used_nums != target_nums:
        return 0.0  # num not match

    if not re.fullmatch(r'[\d\+\-\*\/\(\)]+', expr_clean):
        return 0.0

    try:
        ast.parse(expr_clean, mode='eval')  
        result = builtins.eval(expr_clean) 
        return 1.0 if abs(result - 24.0) < 1e-5 else 0.0
    except:
        return 0.0


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
        # Agents will be initialized on the first call to their respective rollouts.
        self.training_agent = None
        self.validation_agent = None
        self.val_step_n = None

        self.rollout_dir = None
        self.train_rollout_dir = None
        self.val_rollout_dir = None
        self.train_lock_file = None
        self.val_lock_file = None

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_rollout_dir = f"./rollout_data/octo_game24_{timestamp}" 
        self.tools = ["Generalist_Solution_Generator_Tool","Python_Code_Generator_Tool","Google_Search_Tool","Wikipedia_Knowledge_Searcher_Tool"]
        self._solve_call_count = 0
        
        self.run_info_file = os.path.join(self.base_rollout_dir, ".run_info") # 存储本次运行最终目录路径的文件
        self.init_lock_file = os.path.join(self.base_rollout_dir, ".init.lock") # 用于确保初始化只执行一次的全局锁

        # Added locks and state variables for async-safe step management.
        self.train_batch_size = 8 # As defined in the original code logic
        self.rollout_num = 200 # As defined in the original code logic

    async def _solve_and_evaluate(self, calc_agent: OctotoolsCalcAgent, task: Any, step_n: int, val: bool = False):
        """A helper function to run the agent, parse the result, and evaluate it."""
    
        # self.train_batch_size = calc_agent.resources.get("data").train_batch_size
        # self.rollout_num = calc_agent.resources.get("actor_rollout_ref").rollout.n

        try:
            output_format = "When ready, output the answer enclosed in <answer> and </answer> tags. For multiple-choice questions, provide only the uppercase letter of the correct option (e.g., A, B, C, ...). Do not generate any content after the </answer> tag."
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


    # --- NEW METHOD START ---
    # 3. 创建一个新的、进程安全的单次初始化方法
    async def _initialize_run_once(self, resources: NamedResources):
        """
        Ensures that the rollout directory is set up only once per run,
        in a process-safe way.
        """
        # 如果当前进程已经初始化过了，直接返回
        if self.rollout_dir is not None:
            return

        # 确保基础目录存在，以便存放锁和信息文件
        os.makedirs(self.base_rollout_dir, exist_ok=True)
        
        init_lock = FileLock(self.init_lock_file, timeout=60)
        with init_lock:
            # 检查是否已有其他进程完成了初始化并留下了信息文件
            if os.path.exists(self.run_info_file):
                # 如果是，直接读取最终的目录路径
                with open(self.run_info_file, 'r') as f:
                    final_rollout_dir = f.read().strip()
            else:
                # 如果不是，说明这是所有进程中第一个执行此代码的
                # 由它来创建目录名，并写入共享文件
                model_name = resources.get("main_llm").model
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") # 格式化的时间戳
                model_name = model_name.rsplit('/', 1)[-1]
                # 构建符合您要求的目录名：base_name_model_timestamp
                final_rollout_dir = os.path.join(
                    self.base_rollout_dir, f"{model_name}_{timestamp}"
                )
                
                # 将最终确定的路径写入共享文件，供其他进程使用
                with open(self.run_info_file, 'w') as f:
                    f.write(final_rollout_dir)
                print(f"Run directory created by process {os.getpid()}: {final_rollout_dir}")

        # 至此，所有进程都获得了相同的 `final_rollout_dir`
        # 设置当前进程实例的路径属性
        self.rollout_dir = final_rollout_dir
        self.train_rollout_dir = os.path.join(self.rollout_dir, "train")
        self.val_rollout_dir = os.path.join(self.rollout_dir, "validation")
        
        # 创建实际的子目录
        os.makedirs(self.train_rollout_dir, exist_ok=True)
        os.makedirs(self.val_rollout_dir, exist_ok=True)
        
        # 定义各自的锁文件路径
        self.train_lock_file = os.path.join(self.train_rollout_dir, ".train.lock")
        self.val_lock_file = os.path.join(self.val_rollout_dir, ".val.lock")
    # --- NEW METHOD END ---
        
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
    Trainer(n_workers=10).fit(CalcAgent(), "http://localhost:9999/")