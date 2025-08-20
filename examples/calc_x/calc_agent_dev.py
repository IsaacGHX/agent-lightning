import os
from agentlightning import Trainer, DevTaskLoader, LLM
from calc_octo import CalcAgent


def dev_task_loader() -> DevTaskLoader:
    return DevTaskLoader(
        tasks=[
            {
                "question": "What is 2 + 2?",
                "result": "4",
                "extra_info":{'ground_truth': '4', 'idx': 0}
            },
            {
                "question": "What is 3 * 5?",
                "result": "15",
                "extra_info":{'ground_truth': '15', 'idx': 1}
            },
            {
                "question": "What is the square root of 16?",
                "result": "4",
                "extra_info":{'ground_truth': '4', 'idx': 2}
            },
        ],
        resources={
            "main_llm": LLM(
                endpoint="https://api.openai.com/v1", model="gpt-4o-mini", sampling_parameters={"temperature": 0.7}
            ),
        },
    )


if __name__ == "__main__":
    Trainer(n_workers=1, dev=True, max_tasks=2).fit(CalcAgent(), "http://localhost:9998/", dev_task_loader())
