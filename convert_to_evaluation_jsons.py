import csv
import json


def convert_to_evaluation_jsons(args):
    filename = args.infile
    #filename = './test_detoxified_bloomz_reordered.tsv'
    preds = []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            assert len(row) == 3
            preds.append(row[1])
    preds_dict = []
    pred_eval_out = args.outfile
    idx = 0
    for src, pred, lang in zip(sources, preds, langs):
        pred_dict = {
            'id': idx,
            'text': pred
        }
        preds_dict.append(pred_dict)
        idx += 1

    with open(args.outfile, 'w') as file:
        for item in preds_dict:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()
    convert_to_evaluation_jsons(args)