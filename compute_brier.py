import pandas as pd

filenames = {
    'prompt1/gpt4_resolution': {
        'original': 'GPT Answer',
        'new': 'GPT4_prompt1'
    }, 
    'prompt1/gpt35_resolution': {
        'original': 'GPT Answer',
        'new': 'GPT35_prompt1'
    },
    'prompt2/gpt4': {
        'original': 'GPT Answer',
        'new': 'GPT4_prompt2'
    }, 
    'prompt2/gpt35': {
        'original': 'GPT Answer',
        'new': 'GPT35_prompt2'
    },
    'prompt2/react': {
        'original': 'React Answer',
        'new': 'react_prompt2'
    },
}
meta_df = pd.read_csv('dataset_filtered_with_resolution.csv')

for filename, col_names in filenames.items():
    df = pd.read_csv(filename + '.csv')
    resolutions = meta_df['resolution'] == 'YES'
    meta_df[col_names['new']] = (df[col_names['original']] - resolutions) ** 2
    print(filename, 'num refusals:', meta_df[col_names['new']].isna().sum())
    print(filename, 'mean brier:', meta_df[col_names['new']].mean())
    print(filename, 'accuracy:', ((meta_df[col_names['new']] > 0.5) == resolutions).mean())

meta_df.to_csv('brier_score.csv')