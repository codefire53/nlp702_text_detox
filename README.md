# Evaluation
Sometimes since the competition's leaderboard is not reliable (red: the scorer unable to process the file due to encoding error but they dont provide the clear error of the cause), we have to evaluate the prediction by ourselves. To do so, you need to run this command to convert the prediction tsv file into the compliant json.
```
python convert_to_evaluation_jsons.py --infile ./dataset/bloomz_dev.tsv --outfile ./dataset/bloomz_pred_dev.json 
```
Then to do evaluation on validation set (please just differentiate the prediction and o argument here)
```
python evaluate_pred.py --prediction ./dataset/bloomz_pred_dev.json  -i ./dataset/source_dev.json -g ./multidetox_gt_dev.json -o ./eval_files/bloomz_dev.txt
```
Since we dont have ground truth for test set, alternatively we compare the chrf score between the source sentences their respective predictions since we hope that there will be larger overlaps between the ground truths and the source/toxic sentences. Thus for test set, you can run
```
python evaluate_pred.py --prediction ./dataset/bloomz_pred_test.json  -i ./dataset/source_test.json -g ./dataset/source_test.json -o ./eval_files/bloomz_test.txt
```