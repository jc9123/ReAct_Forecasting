import os
import re
import csv
import numpy as np
from tqdm import tqdm

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentType, load_tools, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain.agents.initialize import initialize_agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper

from langchain.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper

# TODO: adding the package below for agents
#  pip install --upgrade --quiet  wikibase-rest-api-client mediawikiapi
#  pip install xmltodict
#  pip install google-search-results
#  pip install -U langchain-community tavily-python
from langchain.tools.render import render_text_description_and_args
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.prompts import MessagesPlaceholder

# from langchain.schema import ChatMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager

from llamp.mp.agents import (
    MPSummaryExpert,
    MPThermoExpert,
    MPElasticityExpert,
    MPDielectricExpert,
    MPPiezoelectricExpert,
    MPMagnetismExpert,
    MPElectronicExpert,
    MPSynthesisExpert,
    MPStructureRetriever,
)

# from llamp.arxiv.agents import ArxivAgent

load_dotenv()
import pandas as pd
from query_llm import construct_openai_message, call_openai

############################### pdf reader, under construction ###############################
# from langchain_community.document_loaders import PyPDFLoader
# loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
# pages = loader.load_and_split()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION", None)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", None)
SEARCHAPI_API_KEY = os.getenv("SEARCHAPI_API_KEY", None)


BING_SUBSCRIPTION_KEY = os.getenv("BING_SUBSCRIPTION_KEY", None)
BING_SEARCH_URL = os.getenv("BING_SEARCH_URL", None)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", None)

OPENAI_GPT_MODEL = "gpt-4-1106-preview"
# OPENAI_GPT_MODEL = "gpt-4-0125-preview"
# OPENAI_GPT_MODEL = "gpt-3.5-turbo-1106"
# OPENAI_GPT_MODEL = "gpt-4"

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

from parse_answers_gpt import system_prompt as parse_answers_prompt
from query_llm import Model


top_llm = ChatOpenAI(
    temperature=0.1,
    model=OPENAI_GPT_MODEL,
    openai_api_key=OPENAI_API_KEY,
    openai_organization=OPENAI_ORGANIZATION,
    streaming=False,
    callbacks=[StreamingStdOutCallbackHandler()],
)

bottom_callback_handler = StreamingStdOutCallbackHandler()

bottom_llm = ChatOpenAI(
    temperature=0,
    model=OPENAI_GPT_MODEL,
    openai_api_key=OPENAI_API_KEY,
    openai_organization=OPENAI_ORGANIZATION,
    max_retries=5,
    streaming=True,
    callbacks=[bottom_callback_handler],
)

google = GoogleSearchAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
# wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
pubmed = PubmedQueryRun()
# google_serper = load_tools(["google-serper"], llm=bottom_llm)
google_trend = api_wrapper = GoogleTrendsAPIWrapper()
# searchapi = load_tools(["searchapi"], llm=bottom_llm)
from langchain_community.utilities import SearchApiAPIWrapper

search = SearchApiAPIWrapper()

serpapi = load_tools(["serpapi"], llm=bottom_llm)
# tavily = TavilySearchResults()
bing = BingSearchAPIWrapper()

tools = load_tools(["llm-math"], llm=bottom_llm)
yahoo = YahooFinanceNewsTool()
tools += [PythonREPLTool()]
tools += [
    Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=google.run,
    ),
    # pubmed,
    # arxiv,
    # wikipedia,
    # google_serper,
    # google,
    # Tool(
    #     name="Intermediate Answer",
    #     func=search.run,
    #     description="useful for when you need to ask with search",
    # ),
    # Tool(
    #     name="Intermediate Answer",
    #     func=bing.run,
    #     description="useful for when you need to ask with search",
    # ),
    # yahooerpapi,
    # tavily
]

instructions = re.sub(
    r"\s+",
    " ",
    """You are a forecasting-focused agent with capabilities to access and analyze a wide array of data for predictive insights. 
        You can consult historical data, financial reports, weather information, and social media trends through a combination of external data sources such as financial databases, weather APIs, social media analytics platforms, and a Python REPL, which you can use to execute Python code for data analysis and model predictions. 
        If you encounter errors in your code, debug and retry until you achieve accurate predictions. Use the output of your code to inform your forecasts, ensuring they are as precise and reliable as possible. 
        In situations where the query is unclear or lacks specifics, engage with the user to refine their request for more targeted in
        sights. 
        While you do not have direct control over the external data sources, you act as a central intelligence, orchestrating multiple assistant agents to gather, process, and analyze data needed for forecasting. It's crucial to provide complete context in your inputs to enable the assistant agents to effectively contribute to the task at hand.
        Make an educated guess instead of saying I cannot provide a forecast. Please output the 80% confidence internval in the format of [lower bound, upper bound] if it's asking for the confidence internval. Please output the probability if it's asking the chance of event happening.
    """,
).replace("\n", " ")


ensemble_instructions = re.sub(
    r"\s+",
    " ",
    """In this chat, you are a superforecaster that has a strong track record of accurate forecasts of the future. 
        As an experienced forecaster, you evaluate past data and trends carefully and aim to predict future events as accurately as you can, even though you cannot know the answer.
        This means you put probabilities on outcomes that you are uncertain about (ranging from 0 to 100%).
        You aim to provide as accurate predictions as you can, ensuring that they are consistent with how you predict the future to be. You also outline your reasons for this forecasting. 
        In your reasons, you will carefully consider the reasons for and against your probability estimate, you will make use of comparison classes of similar events and probabilities and take into account base rates and past events as well as other forecasts and predictions. 
        In your reasons, you will also consider different perspectives. Once you have written your reasons, ensure that they directly inform your forecast.
        Then, you will provide me with a number between 0 and 100 (up to 2 decimal places) that is your best prediction of the event. Take a deep breath and work on this problem step-by-step.
        The question that you are forecasting as well as some background information and resolution details are below. Read them carefully before making your prediction.
        Please use python calculator tool to double check if your reasoning is mathematically reasonable.
    """,
).replace("\n", " ")

query3_instructions = re.sub(
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

base_prompt = hub.pull("langchain-ai/react-agent-template")
# prompt = base_prompt.partial(instructions=instructions)
prompt = base_prompt.partial(instructions=query3_instructions)

conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

agent = create_react_agent(top_llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=conversational_memory,
)


def answer_extractor(response):
    EVALUATOR_MODEL = "gpt-4-0314"
    # message_content = """
    # You are tasked with extracting the final forecasting answer from a given model response. Use the guidelines below to determine your output format based on the type of answer provided:

    # For a numeric confidence interval: If the response includes an "X% confidence interval", extract the numeric range and output it as a Python list [lower_bound, upper_bound] in integer format. Convert all monetary units to their smallest denominations (e.g., millions to exact units).

    # For a binary answer: If the response is a simple yes or no prediction, output True for "yes" and False for "no".
    # Other formats: If the response does not fit the above categories, output the answer as directly stated in the response.

    # Example Input: "The 80% confidence interval for the gross of 'Mean Girls' (the 2024 release) in the US on the weekend of Jan 26-28 is estimated to be between $3.2M and $14.4M."

    # Expected Output: [3200000, 14400000]
    # Remember to exact units and only include number.

    # Instructions: Analyze the response carefully before outputting the value. Output only the value, not the reasoning or any additional information. If the response doesn't include any number, output nan.

    # Response:
    # """ + response
    message_content = (
        """
    You are tasked with extracting the final forecasting answer from a given model response. Use the guidelines below to determine your output format based on the type of answer provided:

    For a numeric confidence interval: If the response includes an "X% confidence interval", extract the numeric range and output it as a Python list [lower_bound, upper_bound] in integer format. Convert all monetary units to their smallest denominations (e.g., millions to exact units).
    Example Input: "The 80% confidence interval for the gross of 'Mean Girls' (the 2024 release) in the US on the weekend of Jan 26-28 is estimated to be between $3.2M and $14.4M."
    Expected Output: [3200000, 14400000]

    For a binary answer: If the response only include the probability of the event happen, output that number of probability, which should within 0-1 range. 
    Example Input: ""Mean Girls" (the 2024 release) has 60% probability to be the highest-grossing movie in the US on the weekend of Jan 26-28?"
    Expected Output: 0.6

    Tips:  
    1. Analyze the response carefully before outputting the value.
    2. Output only the value, not the reasoning or any additional information. 
    3. Remember to exact units and only include number. 
    4. If the response doesn't include any number, output nan.

    Response: 
    """
        + response
    )
    message_content = message_content + response
    request_body = construct_openai_message(
        message_content,
        temperature=0.5,
        max_tokens=1000,
        model=EVALUATOR_MODEL,
    )

    answer = call_openai(request_body, EVALUATOR_MODEL)
    return answer


############################### Prompting Pipeline Start ###############################
# csv_path = "ensemble_question.csv"
# csv_path = "filtered_questions.csv"
csv_path = "prompt2/react.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.to_csv(csv_path, index=False)
else:
    print("File not found. Creating a new one with basic structure.")

# c = 0
# Iterate over the DataFrame rows

if "React Answer" in df:
    answers = df['React Answer']
else:
    answers = ['' for _ in range(len(df))]
if "Model Output" in df:
    responses = df["Model Output"]
else:
    responses = [np.nan for _ in range(len(df))]

for index, row in tqdm(list(df.iterrows())):
    try:
        if not np.isnan(row['React Answer']): continue
    except TypeError:
        ...
    # if c > 10:
    #     break

    # prompt = row["Question"]
    prompt = str(row["question"])
    try:
        if not np.isnan(row["question_description"]):
            ...
    except TypeError:
        prompt += " " + str(row["question_description"])
    print(f"prompt: {prompt}")
    # trial and error loop, try 3 times
    for i in range(3):
        try:
            response = agent_executor.invoke(
                {
                    "input": prompt,
                }
            )["output"]
            break
        except:
            continue

    print(f"response: {response}")
    responses[index] = response
    
    request_body = construct_openai_message(response, max_tokens=5, system_prompt=parse_answers_prompt, model=Model.GPT4.value)
    try:
        parsed_ans = call_openai(request_body, Model.GPT4.value)
    except Exception as e:
        parsed_ans = 'Error'
        print(e)
    answers[index] = parsed_ans
    # try:
    #     answer = answer_extractor(response)
    # except:
    #     answer = "Error Out " + response
    # answers[index] = answer
    # c += 1

    df["React Answer"] = answers
    df["Model Output"] = responses
    df.to_csv(csv_path, index=False)
