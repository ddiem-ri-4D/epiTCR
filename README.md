# epiTCR
epiTCR is a highly sensitive predictor for TCR-peptide binding. epiTCR uses TCR CDR3b sequences and peptide sequences as input. Additionally, users can also provide full length MHC to the tool. The output produces the predicted binding probability. 


## Requirements
```text
python >= 3.0.0
numpy 1.22.4
scikit-learn 1.1.2
```
For other requirements, please see the env_requirements.txt file.

## Training
The main module for re-trained is `epiTCR.py`. Users can re-trained the epiTCR model (with or without MHC) by running:

```commandline
python3 epiTCR.py --trainfile data/split-data/with-mhc/train/train.csv --testfile data/split-data/with-mhc/test/test01.csv --chain cem
```

This will print the predictions on the standard output or on a file (that can be specified with the option --outfile).

given that:
- `--trainfile` is a comma-separated file (CSV) containing columns for TCR, epitipe, binder, and/or full length MHC (reported by IMGT). See [example training data](data/split-data/with-mhc/train/train.csv.zip).
- `--testfile` is a CSV file containing columns for TCR, epitipe and/or full length MHC (reported by IMGT). See [example test file](data/split-data/with-mhc/test/test01.csv.zip).
- `--chain` specifies the chain(s) to use (ce, cem). Available options for this parameter are `ce` (cdr3b+epitope) and `cem` (cdr3b+epitope+mhc). Default as `ce`.



## Predictions using the pre-built model

The module for prediction is `predict.py`. Similar to the training command, users predict TCR-epitope or TCR-pMHC by running:

```commandline
python3 predict.py --testfile data/split-data/with-mhc/test/test01.csv --model_file models/rdforest-model.pickle --chain cem
```

given that:
- `--testfile` is a CSV file containing columns for TCR, epitipe and/or full length MHC reported by IMGT. See [example input file](data/split-data/with-mhc/test/test01.csv.zip).
- `--model_file` specifies the full path of the file with trained model, should be a pickle files. Default model as `models/rdforest-model.pickle`.
- `--chain` specifies the chain(s) to use (ce, cem). Options for this parameter are `ce` (cdr3b+epitope) and `cem` (cdr3b+epitope+mhc). Default as `ce`.

## Prediction output  

The prediction will be printed out on the standard output or on a file (which can be specified with the option `--outfile`).

epiTCR prediction output contains a table with four columns: the CDR3b sequences, epitope sequences, (full length MHC,) and the binding probability for the corresponding complexes. The example output file is [here](test/output/output_prediction.csv).

## Contact

For more questions or feedback, please simply post an [Issue](https://github.com/ddiem-ri-4D/epiTCR/issues/new). 

## References

