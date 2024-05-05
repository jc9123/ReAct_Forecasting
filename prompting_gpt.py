import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from query_llm import construct_openai_message, call_openai, Model
from parse_answers_gpt import system_prompt as parse_answers_prompt

paths = ["prompt2/gpt4.csv", "prompt2/gpt35.csv"]
forecasting_models = [Model.GPT4.value, Model.GPT3_5_TURBO_1106.value]
parser_model = Model.GPT4.value
forecasting_prompt = re.sub(
    r"\s+",
    " ",
    """In this chat, you are a superforecaster who has a strong track record of accurate forecasting. You evaluate past data and trends carefully for potential clues to future events, while recognizing that the past is an imperfect guide to the future so you will need to put probabilities on possible future outcomes (ranging from 0 to 100%). Your specific goal is to maximize the accuracy of these probability judgments by minimising the Brier scores that your probability judgments receive once future outcomes are known. Brier scores have two key components: calibration (across all questions you answer, the probability estimates you assign to possible future outcomes should correspond as closely as possible to the objective frequency with which outcomes occur) and resolution (across all questions, aim to assign higher probabilities to events that occur than to events that do not occur).
        You outline your reasons for each forecast: list the strongest evidence and arguments for
        making lower or higher estimates and explain how you balance the evidence to make your
        own forecast. You begin this analytic process by looking for reference or comparison classes of similar events and grounding your initial estimates in base rates of occurrence (how often do events of this sort occur in situations that look like the present one?). You then adjust that initial estimate in response to the latest news and distinctive features of the present situation, recognising the need for flexible adjustments but also the risks of over-adjusting and excessive volatility. Superforecasting requires weighing the risks of opposing errors: e.g., of failing to
        learn from useful historical patterns vs. over-relying on misleading patterns. In this process of error balancing, you draw on several principles of superforecasting:
            * You may find relevant time series data online. If the data is recent, you can use zeroth-order or first-order approximations (but for longer time horizons, recall that these may break down due to saturation effects).
            * Don't ignore base rates for low-probability events that haven't happened, but rather attempt to estimate those probabilities to inform your estimate.
            * Attempt to approximate the same quantity in multiple different ways. You are welcome to split your reasoning into multiple sections, where in each section you describe how you reached an estimate independently. If your estimates in these sections are very far off, reason through why the estimates are so different, potentially update those estimates, and theorize how confident you are in each estimate. Then, combine those forecasts using a weighted or trimmed mean or median.
            * Some data can be modeled well using a common distribution, e.g. normal, log-normal, power law, or Poisson distribution (but this is by no means a guarantee). If there's a good reason to believe this is the case, you should use information about those distributions.
        Then, you will provide me with a number between 0.00 and 1.00 (up to 2 decimal places) that is your best prediction of the probability of event happening. Take a deep breath and work on this problem step-by-step.
        The question that you are forecasting as well as some background information and resolution criteria are below. Read them carefully before making your prediction.
        Please use the Python calculator tool to double check that your reasoning is mathematically reasonable.
        Most importantly: you will provide me with a numerical answer under all circumstances, even in the absence of crucial data.
    """,
).replace("\n", " ")

for forecasting_model, path in zip(forecasting_models, paths):
    print("Processing", path)
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if "GPT Answer" not in df:
        df["GPT Answer"] = ["" for _ in range(len(df))]
    if "Model Output" not in df:
        df["Model Output"] = [np.nan for _ in range(len(df))]

    for i, row in tqdm(list(df.iterrows())):
        try:
            if not np.isnan(row['GPT Answer']): continue
        except TypeError:
            ...

        question = str(row["question"])
        try:
            if not np.isnan(row["question_description"]):
                ...
        except TypeError:
            question += " " + str(row["question_description"])
            
        request_body = construct_openai_message(
            forecasting_prompt + " " + question, model=forecasting_model
        )
        try:
            model_output = call_openai(request_body, forecasting_model)
        except Exception as e:
            model_output = "Error"
            print(e)

        request_body = construct_openai_message(
            parse_answers_prompt + " " + model_output, model=parser_model
        )
        try:
            numerical_answer = float(call_openai(request_body, parser_model))
        except Exception as e:
            numerical_answer = np.nan

        model_output = re.sub(
            r"\s+",
            " ",
            model_output,
        ).replace("\n", " ")

        df.loc[i, "GPT Answer"] = numerical_answer
        df.loc[i, "Model Output"] = model_output
        df.to_csv(path)
