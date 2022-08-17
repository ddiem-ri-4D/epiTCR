# epiTCR
A highly sensitive predictor for TCR-peptide binding 


## Requirements
```text
python >= 3.x
numpy 1.22.4
scikit-learn 1.1.2
```

## Model Training
The main module for training is `epiTCR.py`. You can train the epiTCR without mhc models running

```commandline
python3 epiTCR.py --trainfile data/split-data/without-mhc/train/train.csv --testfile data/split-data/without-mhc/test/test01.csv 
```
This will print the predictions on the standard output or on a file (that can be specified with the option --outfile).

Both training and test set should be a comma-separated CSV files. The files should have the following columns (with headers): CDR3b, epitope, MHC (full-length), binder (the binder coulmn is not required in the test file). See data/split-data/without-mhc/train/train.csv and data/split-data/without-mhc/test/test01.csv as an example.


## Binding Prediction

You can predict using the `predict.py` module.
It is quite similar to training, run:
```commandline
python3 predict.py --testfile data/split-data/without-mhc/test/test01.csv 
```

## Output file example
epiTCR without mhc outputs a table with 4 columns: CDR3b sequences, epitopes sequences, MHC full length, and predict for each pair of TCR/epitope.

