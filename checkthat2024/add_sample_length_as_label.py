import pandas as pd
import os
import numpy as np

def determine_quantiles(train_file_path):
    df_train = pd.read_csv(train_file_path, sep='\t')
    if 'Text' in df_train.columns:
        lengths = df_train['Text'].apply(lambda x: len(x.split())).tolist()
        first_quantile = np.percentile(lengths, 33.333)
        second_quantile = np.percentile(lengths, 66.666)
        print(f"First quantile (33.333%): {first_quantile}")
        print(f"Second quantile (66.666%): {second_quantile}")
        return first_quantile, second_quantile
    return None, None

def process_files(file_paths, first_quantile, second_quantile, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at {output_dir}")

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep='\t')
            if 'Text' in df.columns:
                df['Text'] = df['Text'].apply(lambda text: append_label(text, first_quantile, second_quantile))
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, sep='\t', index=False)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process or save file {file_path}: {e}")

def append_label(text, first_quantile, second_quantile):
    num_tokens = len(text.split())
    if num_tokens < first_quantile:
        return text + ' sss'
    elif num_tokens <= second_quantile:
        return text + ' mmm'
    else:
        return text + ' lll'

input_dir = 'C:/Users/ardi_/Documents/ZHAW/BA/CheckThat2024/data_f/CT24_checkworthy_english'
output_dir = 'C:/Users/ardi_/Documents/ZHAW/BA/CheckThat2024/data_f_lll_quantiles/CT24_checkworthy_english'

# Training data path
train_file_path = os.path.join(input_dir, 'CT24_checkworthy_english_train.tsv')

# All dataset paths including training
file_paths = [
    train_file_path,
    os.path.join(input_dir, 'CT24_checkworthy_english_test.tsv'),
    os.path.join(input_dir, 'CT24_checkworthy_english_dev-test.tsv'),
    os.path.join(input_dir, 'CT24_checkworthy_english_dev.tsv')
]

# Determine quantiles from the training dataset
first_quantile, second_quantile = determine_quantiles(train_file_path)

# Process all datasets using the determined quantiles
if first_quantile is not None and second_quantile is not None:
    process_files(file_paths, first_quantile, second_quantile, output_dir)
else:
    print("Quantiles could not be determined. Processing is aborted.")
