import json
from datasets import load_dataset
import os
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



def main():
    # evaluate the best model on test
    best_model = "mT5_FT_chrf/checkpoint-14500"
    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(best_model)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(best_model)
    # read df without index
    test_df = pd.read_csv("../dataset/inputs_multilingual_test.tsv", sep="\t")
    toxic_sentences = test_df["toxic_sentence"].tolist()
    # predict with the model with batch size 8
    neutral_sentences = []
    for i in tqdm(range(0, len(toxic_sentences), 8)):
        batch = toxic_sentences[i:i+8]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model.generate(**inputs)
        neutral_sentences.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    test_df["neutral_sentence"] = neutral_sentences
    # make neutral_sentence column the second column
    test_df = test_df[["toxic_sentence", "neutral_sentence", "lang"]]
    test_df.to_csv("outputs_multilingual_test_mt5.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

