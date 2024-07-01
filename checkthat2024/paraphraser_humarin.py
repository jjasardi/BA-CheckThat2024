import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

train_df = pd.read_csv("/home/ubuntu/BA_24/BA-CheckThat2024/data_f_all/CT24_checkworthy_english/CT24_checkworthy_english_dev.tsv", sep="\t")

output_dir = "/home/ubuntu/BA_24/BA-CheckThat2024/data_f_all_paraphrased_sr/CT24_checkworthy_english"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the highest sentence ID from the original training data
max_sentence_id = train_df["Sentence_id"].max()

augmented_data = []

max_samples = 52

for index, row in train_df.iterrows():
    if row["class_label"] == "Yes":
        # Paraphrase the sentence
        paraphrased_sentences = paraphrase(row["Text"])
        # Append the paraphrased sentences to the augmented data list
        for paraphrased_sentence in paraphrased_sentences:
            max_sentence_id += 1
            augmented_data.append([max_sentence_id, paraphrased_sentence, row["class_label"]])
    if len(augmented_data) >= max_samples:
        break

new_df = pd.DataFrame(augmented_data, columns=["Sentence_id", "Text", "class_label"])
augmented_df = pd.concat([train_df, new_df], ignore_index=True)

shuffled_augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
shuffled_augmented_df.to_csv(os.path.join(output_dir, 'CT24_checkworthy_english_dev.tsv'), sep="\t", index=False)
