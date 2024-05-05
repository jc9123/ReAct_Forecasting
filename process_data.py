import requests
import datetime
import pandas as pd
from tqdm import tqdm


def fetch_market_info(url):
    slug = url.split('/')[-1]
    base = "https://manifold.markets/api/v0/slug/"
    response = requests.get(f"{base}/{slug}")
    if response.status_code == 200:
        to_include = ['id', 'probability', 'question', 'textDescription', 'createdTime', 'closeTime']
        response = response.json()
        ret = {k: response[k] for k in to_include}
        return ret
    else:
        print("Get error", url)
        return None

def fetch_market_info_2(mid):
    base = "https://manifold.markets/api/v0/market/"
    response = requests.get(f"{base}/{mid}")
    if response.status_code == 200:
        to_include = ['id', 'answers', 'resolution']
        response = response.json()
        return {k: (response[k] if k in response else "Not Yet Resolved") for k in to_include}
    else:
        print("Get error", mid)
        return None

def fetch_market_data(market_ids):
    base = "https://api.manifold.markets/v0/market/"
    result = {}

    for market_id in market_ids:
        url = f"{base}{market_id}"
        response = requests.get(url)
        if response.status_code == 200:
            result[market_id] = response.json()
            return result
        else:
            print("Get error", url)
            return None
            
    
    
def filter_resolution_data():
    path = 'queries.csv'
    data = pd.read_csv(path)
    question_urls = data['url'].tolist()
    # market_ids = [url.split('/')[-1] for url in market_url]
    market_ids = []
    open_times = []
    close_times = []
    timestamp_oks = []
    questions = []
    descriptions = []
    prob = []
    ##End before 2024, May, 2
    resolution_target = datetime.datetime(2024, 5, 2).timestamp() * 1000
    for url in tqdm(question_urls):
        market_info = fetch_market_info(url)
        if market_info is None:
            market_id = open_time = resolution_time = question = description = probability = None
        else:
            market_id = market_info['id']
            open_time = market_info['createdTime']
            resolution_time = market_info['closeTime']
            question = market_info['question']
            description = market_info['textDescription']
            probability = market_info['probability']
        market_ids.append(market_id)
        open_times.append(open_time)
        close_times.append(resolution_time)
        timestamp_oks.append(resolution_time <= resolution_target)
        questions.append(question)
        descriptions.append(description)
        prob.append(probability)

        # print(probability)
    
    buf = pd.DataFrame({'id' : market_ids,
                        'open_time': open_times, 
                        'close_time': close_times, 
                        # 'timestamp_ok': timestamp_oks,
                        'question' : questions,
                        'description' : descriptions,
                        'probability' : prob})
    
    data = pd.concat([data, buf], axis=1)
    data.to_csv("queries2_N.csv")

def get_answer():
    path = 'queries2.csv'
    data = pd.read_csv(path)
    ids = data['id'].tolist()
    resolution = []
    descriptions = []
    ##End before 2024, May, 2
    resolution_target = datetime.datetime(2024, 5, 2).timestamp() * 1000
    for id in tqdm(ids):
        market_info = fetch_market_info_2(id)
        print(market_info)
        resolution.append(market_info['resolution'])
    buf = pd.DataFrame({'resolution': resolution})
    data = pd.concat([data, buf], axis=1)
    data.to_csv("queries3.csv")


def join_tbl():
    df1 = pd.read_csv('merged_q_clean.csv')
    df2 = pd.read_csv('answer_desc.csv')
    result = pd.merge(df1, df2, on='Question', how='inner')
    result.to_csv('merged_q_descript.csv', index=False)

def filter():
    df = pd.read_csv('merged_temp.csv')
    filtered_df = df[(df['resolution'].isin(['YES', 'NO']))]
    filtered_df.to_csv('merged_clean_F.csv', index=False)

def add_numerical():
    df = pd.read_csv('merged_clean.csv')
    mapping = {'YES': 1, 'NO': 0}
    df['resolution numerical'] = df['resolution'].map(mapping)
    df.to_csv('merged_clean2.csv', index=False)

def calculate_brier_score_from_csv():
    df = pd.read_csv('human.csv')
    brier_score = ((df['resolution numerical'] - df['probability']) ** 2)
    df['brier_score'] = brier_score
    df.to_csv('brier_score_human.csv', index=False)
    brier_score = list(brier_score)
    average = lambda numbers: sum(numbers) / len(numbers) if numbers else 0
    a = average(brier_score)
    print(a)


def filter_col():
    df = pd.read_csv('filtered_q.csv')
    filtered_df = df[(df['Gpt_Response'] == 'Yes')]
    filtered_df.to_csv('merged_q_clean.csv', index=False)
 

if __name__ == "__main__":
    # filter_resolution_data()
    # join_tbl()
    # join_tbl()
    # calculate_brier_score_from_csv()
    # filter_col()
    # join_tbl()
    filter()