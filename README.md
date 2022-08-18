# epiTCR
A highly sensitive predictor for TCR-peptide binding 


## Requirements
```text
python >= 3.x
numpy 1.22.4
scikit-learn 1.1.2
```
Other requirements, see env_requirements.txt file.

## Model Training
The main module for training is `epiTCR.py`. You can train the epiTCR with mhc model running:

```commandline
python3 epiTCR.py --trainfile data/split-data/with-mhc/train/train.csv --testfile data/split-data/with-mhc/test/test01.csv --chain cem
```
where:
- `--trainfile` is a csv file with TCR, epitipe and MHC (full length) columns. See example file in data/split-data/without-mhc/train/train.csv as an example.
- `--testfile` is a csv file with TCR, epitipe and MHC (full length) columns. See example file in data/split-data/with-mhc/test/test01.csv as an example.
- `--chain` specify the chain(s) to use (ce, cem). You can select ce (cdr3b+epitope), cem (cdr3b+epitope+mhc). Default: ce

- All other cmd parameters are similar to the training process. 

This will print the predictions on the standard output or on a file (that can be specified with the option --outfile).

Both training and test set should be a comma-separated CSV files. The files should have the following columns (with headers): CDR3b, epitope, MHC (full-length), binder (the binder coulmn is not required in the test file). 


## Binding Prediction

You can predict using the `predict.py` module.
It is quite similar to training, you can predict the epiTCR with mhc model running:
```commandline
python3 predict.py --testfile data/split-data/with-mhc/test/test01.csv --model_file models/rdforest-model.pickle --chain cem
```
where:
- `--testfile` is a csv file with TCR, epitipe and MHC (full length) columns. See example file in data/split-data/with-mhc/test/test01.csv
- `--model_file` specify the full path of the file with trained model, should be a pickle files.
- `--chain` specify the chain(s) to use (ce, cem). You can select ce (cdr3b+epitope), cem (cdr3b+epitope+mhc). Default: ce

## Output file 
epiTCR with mhc outputs a table with 4 columns: CDR3b sequences, epitopes sequences, MHC full length, and predict for each pair of TCR/epitope. The example output file is under test/output/output_prediction.csv


## References

