The dataset used is in homework.csv and manifold_market.csv. To evaluate the dataset, run either prompting_gpt.py or prompting_react.py. The generated csv is then proceed with process_data.py and compute_brier.py.

`conda create -n forecast python=3.11`
do `pip install -e .` within the folder

Create `.env` file and fill in the API key in the file. 
`OPENAI_API_KEY=
OPENAI_ORGANIZATION=
MP_API_KEY=
BING_SUBSCRIPTION_KEY=
BING_SEARCH_URL=

GOOGLE_API_KEY=
GOOGLE_CSE_ID=
SERPAPI_API_KEY=
SEARCHAPI_API_KEY=
`
