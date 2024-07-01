import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

def load_data(filepath, chunksize=100):
    """ Generator to load data in chunks from a TSV file. """
    return pd.read_csv(filepath, sep='\t', chunksize=chunksize)

def translate(texts, model_name):
    """ Translate a list of texts using the specified MarianMT model. """
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    tokenized_texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    try:
        translated = model.generate(**tokenized_texts)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return texts  # Return original texts if an error occurs

def back_translate(texts, src_lang="en", tgt_lang="fr"):
    """ Perform back translation on a list of texts. """
    model_name_to_tgt = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model_name_to_src = f'Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}'
    
    translated_to_tgt = translate(texts, model_name_to_tgt)
    back_translated_texts = translate(translated_to_tgt, model_name_to_src)

    return back_translated_texts

# Path settings
input_path = '/home/ubuntu/BA_24/BA-CheckThat2024/data_f/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv'
output_path = '/home/ubuntu/BA_24/BA-CheckThat2024/data_f_all/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv'

# Initialize an empty DataFrame for all results
final_data = pd.DataFrame()

# Process the data in chunks
for chunk in load_data(input_path, chunksize=100):
    # Perform back translation on the text column of the dataframe
    chunk['Text'] = back_translate(chunk['Text'].tolist())

    # Append processed chunk to the final DataFrame
    final_data = pd.concat([final_data, chunk], ignore_index=True)

# Save the complete DataFrame to a new TSV file
final_data.to_csv(output_path, sep='\t', index=False)
