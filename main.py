import asyncio
import json
import os
import random

import click
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import seaborn as sns

from llm import utils as llm_utils
from llm.types import ChatModelName

openai.api_key = "YOUR_API_KEY"

_SYSTEM_MESSAGE = "You are a helpful assistant."


async def generate_job_description() -> str:
  prompt = "Generate a brief description of a job position."
  return await llm_utils.answer_one_shot_chat(
    system_message=_SYSTEM_MESSAGE,
    user_message=prompt,
    model_name=ChatModelName.GPT_4_TURBO,
    max_tokens=400,
    temperature=0.7,
  )


async def generate_applicant_description(job_description: str) -> str:
  prompt = "Generate a brief description of a hypothetical job applicant.\nChoose randomly from a wide range of skill levels."
  return await llm_utils.answer_one_shot_chat(
    system_message=_SYSTEM_MESSAGE,
    user_message=prompt,
    model_name=ChatModelName.GPT_4_TURBO,
    max_tokens=400,
    temperature=0.7,
  )


async def rate_applicants(job: str, job_applicants: list[str]) -> list[float]:
  applicant_strings = "\n".join(
    [f"{i + 1}. {applicant}" for i, applicant in enumerate(job_applicants)]
  )
  prompt = f"On a scale from 1 to 10, how qualified is each applicant for the following job?\nRespond with the candidate number followed by a rating, like this:\n1. <rating>\n2. <rating>\n...\n\nDo not include any explanation, just the number.\nJob: {job}\nApplicants:\n{applicant_strings}"

  num_tries = 3
  for i in range(num_tries):
    response = await llm_utils.answer_one_shot_chat(
      system_message=_SYSTEM_MESSAGE,
      user_message=prompt,
      model_name=ChatModelName.GPT_4_TURBO,
      max_tokens=10 * len(job_applicants),
      temperature=0.0,
    )
    try:
      lines = [l.strip() for l in response.split("\n") if l.strip()]
      if len(lines) != len(job_applicants):
        raise ValueError("model response has wrong number of lines")

      result: list[float] = []
      for i, l in enumerate(lines):
        num, rating = l.split(". ")
        if int(num) != i + 1:
          raise ValueError("model response has wrong order")
        result.append(float(rating))

      return result
    except ValueError:
      if i < num_tries - 1:
        continue
      else:
        print("ERROR: ", response)
        raise

  assert False


async def run_trials(
  jobs: list[str], applicants: list[list[str]], num_trials: int
) -> list[list[list[float]]]:
  print("Running trials")
  results: list[list[list[float]]] = []
  for job_idx, (job, job_applicants) in enumerate(zip(jobs, applicants)):
    print(f"Running trials for job: {job_idx + 1}")
    trial_results: list[list[float]] = []
    for _ in range(num_trials):
      random.shuffle(job_applicants)
      ratings = await rate_applicants(job, job_applicants)
      trial_results.append(ratings)
    results.append(trial_results)
  return results


def analyze_results(results: list[list[list[float]]]) -> None:
  # (job, trial, position)
  data = np.array(results)
  avg_ratings = np.mean(data, axis=(0, 1))
  print("(job, trial, position)", data.shape)
  print("avg_ratings", avg_ratings)

  records = []
  for i, job_results in enumerate(results):
    for j, trial_results in enumerate(job_results):
      for k, rating in enumerate(trial_results):
        records.append({"job": i, "trial": j, "position": k, "rating": rating})

  df = pd.DataFrame.from_records(records)
  sns.lineplot(data=df, x="position", y="rating").set_title(
    "Serial Position-Negativity Effect n=300"
  )
  plt.show()


async def run_experiment(num_jobs: int, num_candidates: int, num_trials: int) -> None:
  data_file = "job_applicant_data.json"

  if not os.path.exists(data_file):
    print("Generating jobs")
    jobs = await asyncio.gather(*(generate_job_description() for _ in range(num_jobs)))
    print("Generating applicants")
    applicants = [
      await asyncio.gather(*(generate_applicant_description(job) for _ in range(num_candidates)))
      for job in jobs
    ]
    data = {"jobs": jobs, "applicants": applicants}
    with open(data_file, "w") as f:
      json.dump(data, f)
  else:
    print("Loading existing job/applicant data")
    with open(data_file, "r") as f:
      data = json.load(f)
      jobs = data["jobs"]
      applicants = data["applicants"]

  results_path = "position_negativity_results.json"
  if os.path.exists(results_path):
    with open(results_path, "r") as f:
      results = json.load(f)
  else:
    results = await run_trials(jobs, applicants, num_trials)
    with open(results_path, "w") as f:
      json.dump(results, f)

  analyze_results(results)


@click.command()
@click.option("--num-jobs", default=3, help="Number of jobs to generate.")
@click.option("--num-candidates", default=10, help="Number of candidates per job.")
@click.option("--num-trials", default=100, help="Number of trials (randomized orderings).")
def main(num_jobs: int, num_candidates: int, num_trials: int):
  asyncio.run(run_experiment(num_jobs, num_candidates, num_trials))


if __name__ == "__main__":
  main()
