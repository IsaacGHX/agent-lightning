import os
from agentlightning import Trainer, DevTaskLoader, LLM
from game24_octo import CalcAgent

import pandas as pd

def dev_task_loader() -> DevTaskLoader:
    parquet_path = "data/gameof24/test.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # 读取 parquet 并只取前 50 条
    df = pd.read_parquet(parquet_path)
    df = df.head(10)  # ✅ 只保留前 50 行

    if 'question' not in df.columns or 'result' not in df.columns:
        raise ValueError(f"Parquet file must have 'question' and 'result' columns. Found: {list(df.columns)}")

    ground_truth_col = 'ground_truth' if 'ground_truth' in df.columns else 'result'

    tasks = []
    for idx, row in df.iterrows():
        task = {
            "question": str(row["question"]),
            "result": str(row["result"]),  # 统一转为字符串
            "extra_info": {
                "ground_truth": str(row[ground_truth_col]),
                "idx": int(idx),  # 使用 DataFrame 的索引作为 idx
                # 可选：添加其他列
                # **{k: v for k, v in row.items() if k not in ['question', 'result', ground_truth_col]}
            }
        }
        tasks.append(task)

    return DevTaskLoader(
        tasks=tasks,
        resources={
            "main_llm": LLM(
                endpoint="https://api.openai.com/v1",
                model="gpt-4o",
                sampling_parameters={"temperature": 0.7}
            ),
        },
    )

if __name__ == "__main__":
    Trainer(n_workers=1, dev=True, max_tasks=200).fit(CalcAgent(), "http://localhost:10001/", dev_task_loader())
