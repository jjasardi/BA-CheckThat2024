import numpy as np

def determine_quantiles_from_array(texts):
    # Calculate lengths of each text
    lengths = [len(text.split()) for text in texts]
    
    # Compute quantiles
    first_quantile = np.percentile(lengths, 33.333)
    second_quantile = np.percentile(lengths, 66.666)
    
    print(f"First quantile (33.333%): {first_quantile}")
    print(f"Second quantile (66.666%): {second_quantile}")
    
    return first_quantile, second_quantile

def append_label(text, first_quantile, second_quantile):
    num_tokens = len(text.split())
    if num_tokens < first_quantile:
        return text + ' sss'
    elif num_tokens <= second_quantile:
        return text + ' mmm'
    else:
        return text + ' lll'

def process_texts(texts):
    first_quantile, second_quantile = determine_quantiles_from_array(texts)
    # Apply labels to each text based on its length
    modified_texts = [append_label(text, first_quantile, second_quantile) for text in texts]
    return modified_texts