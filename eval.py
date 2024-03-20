from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
import argparse
from tqdm import tqdm

def eval_polite_score(polite_scores, tokenizer, model, sentences):
    polite_label = 0
    inputs = tokenizer(
        *sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)
    with torch.no_grad():
        try:
            logits = model(**inputs).logits
            preds = torch.softmax(logits, -1)[:,polite_label]
            preds = preds.cpu().numpy()
        except:
            preds = [0] * len(inputs)
    print(f"politeness: {preds}")
    polite_scores.extend(preds)

def compute_cosine_sim(embds1, embds2):
    sim_matrix = np.dot(embds1, embds2.T)
    norms1 = np.linalg.norm(embds1, axis=1)
    norms2 = np.linalg.norm(embds2, axis=1)
    score = sim_matrix / (np.outer(norms1, norms2) + 1e-9)
    return score

def eval_sim_score(all_sim_scores, sts_model, source_sentences, detox_sentences):
    source_len = len(source_sentences)
    detox_len = len(detox_sentences)
    all_texts = detox_sentences + source_sentences
    embds = sts_model.encode(all_texts)
    detox_embds = embds[:detox_len]
    source_embds = embds[detox_len:]
    scores = []
    sim_scores = compute_cosine_sim(detox_embds, source_embds)
    for i in range(detox_len):
        sim_score = max(sim_scores[i,i], 0)
        scores.append(sim_score)
    print(f"similarity: {scores}")
    all_sim_scores.extend(scores)

def eval_chrf_score(chrf_scores, chrf_metric, source_sentences, detox_sentences):
    scores = []
    for src, detox in zip(source_sentences, detox_sentences):
        chrf_score = chrf_metric.compute(predictions=[detox], references=[src], word_order=2)
        chrf_score = chrf_score['score']/100
        scores.append(chrf_score)
    print(f"chrf: {scores}")
    chrf_scores.extend(scores)

def main(args):
    instances = []
    with open(args.eval_file, 'r') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            instances.append(line)
    
    model = AutoModelForSequenceClassification.from_pretrained('textdetox/xlmr-large-toxicity-classifier', num_labels=2).to('cuda')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('textdetox/xlmr-large-toxicity-classifier')
    sts_model = SentenceTransformer('sentence-transformers/LaBSE')
    chrf = evaluate.load("chrf")

    polite_scores = []
    sim_scores = []
    chrf_scores = []

    og_sents = [row[0] for row in instances]
    pred_sents = [row[1] for row in instances]
    for i in tqdm(range(0, len(instances), args.batch_size)):
        eval_chrf_score(chrf_scores, chrf, og_sents[i:i+args.batch_size], pred_sents[i:i+args.batch_size])
        eval_polite_score(polite_scores, tokenizer, model, [pred_sents[i:i+args.batch_size]])
        eval_sim_score(sim_scores, sts_model, og_sents[i:i+args.batch_size], pred_sents[i:i+args.batch_size])
    
    agg_scores = np.array(chrf_scores)*np.array(sim_scores)*np.array(polite_scores)
    overall_total = sum(agg_scores)/len(agg_scores)
    overall_chrf = sum(chrf_scores)/len(chrf_scores)
    overall_sim = sum(sim_scores)/len(sim_scores)
    overall_polite = sum(polite_scores)/len(polite_scores)

    print(f"total: {overall_total}, chrf: {overall_chrf}, similarity: {overall_sim}, polite: {overall_polite}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    main(args)
