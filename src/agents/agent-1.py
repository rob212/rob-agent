# %%

## Agent 1
##
## Prompt directly to LLM.
##

import asyncio
import json
import pandas as pd
from collections import Counter
from litellm import acompletion
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset

MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5",
]


class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""


load_dotenv(find_dotenv())

GAIA_SYSTEM_PROMPT = """ You are a general AI assistant. 
I will ask you a question. First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly. 
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer". 
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason". 
Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending on whether the element is a number or a string.”
"""

PROVIDER_SEMAPHORES = {
    "openai": asyncio.Semaphore(30),
    "anthropic": asyncio.Semaphore(10),
}


def get_provider(model: str) -> str:
    """Extract provider name from model string."""
    return "anthropic" if model.startswith("anthropic/") else "openai"


async def solve_problem(model: str, question: str) -> GaiaOutput:
    """Solve a single problem and return structured output."""
    provider = get_provider(model)

    async with PROVIDER_SEMAPHORES[provider]:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": GAIA_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format=GaiaOutput,
            num_retries=2,
        )
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content

        if finish_reason == "refusal" or content is None:
            return GaiaOutput(
                is_solvable=False,
                unsolvable_reason=f"Model refused to answer (finish_reason: {finish_reason})",
                final_answer="",
            )
        return GaiaOutput.model_validate_json(content)


def is_correct(prediction: str | None, answer: str) -> bool:
    """Check exact match between prediction and answer (case-insensitive)."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()


async def evaluate_gaia_single(problem: dict, model: str) -> dict:
    """Evaluate a single problem-model pair and return result."""
    try:
        output = await solve_problem(model, problem["Question"])
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": is_correct(output.final_answer, problem["Final answer"]),
            "is_solvable": output.is_solvable,
            "prediction": output.final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": output.unsolvable_reason,
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": False,
            "is_solvable": None,
            "prediction": None,
            "answer": problem["Final answer"],
            "error": str(e),
        }


async def run_experiment(
    problems: list[dict],
    models: list[str],
) -> dict[str, list]:
    """Evaluate all models on all problems."""
    tasks = [
        evaluate_gaia_single(problem, model) for problem in problems for model in models
    ]

    all_results = await tqdm_asyncio.gather(*tasks)

    # Group results by model
    results = {model: [] for model in models}
    for result in all_results:
        results[result["model"]].append(result)

    return results


def generate_accuracy_table(data: dict[str, list]) -> pd.DataFrame:
    table = []

    for model, tasks in data.items():
        total = len(tasks)
        if total == 0:
            judged_accuracy = "0/0 (0%)"
            judged_solvable = "0/0 (0%)"
        else:
            correct_count = sum(1 for t in tasks if t.get("correct") == True)
            solvable_count = sum(1 for t in tasks if t.get("is_solvable") == True)

            judged_accuracy = (
                f"{correct_count}/{total} ({correct_count / total * 100:.0f}%)"
            )
            judged_solvable = (
                f"{solvable_count}/{total} ({solvable_count / total * 100:.0f}%)"
            )

        table.append(
            {
                "Model": model,
                "Judged Accuracy": judged_accuracy,
                "Judged Solvable": judged_solvable,
            }
        )

    df = pd.DataFrame(table)
    # Optional: sort by accuracy descending
    df = df.sort_values("Judged Accuracy", ascending=False).reset_index(drop=True)
    return df


def generate_unsolvable_summary(data: dict[str, list]) -> pd.DataFrame:
    """
    Aggregates identical unsolvable_reason strings across all models
    and returns a count table.
    """

    reasons = []

    for model, tasks in data.items():
        for task in tasks:
            reason = task.get("unsolvable_reason", "")
            if reason:  # ignore empty strings
                reasons.append(reason.strip())

    # Count identical reasons
    reason_counts = Counter(reasons)

    # Convert to DataFrame
    df = pd.DataFrame(reason_counts.items(), columns=["Unsolvable Reason", "Count"])

    # Sort by most frequent
    df = df.sort_values("Count", ascending=False).reset_index(drop=True)

    return df


level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")

subset = level1_problems.select(range(20))
results = await run_experiment(subset, MODELS)
# print(json.dumps(results, indent=4))
stats = generate_accuracy_table(results)
unsolvable = generate_unsolvable_summary(results)
print(stats)
print(unsolvable)
