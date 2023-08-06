# epiTCR

epiTCR is a highly sensitive predictor for TCR-peptide binding. epiTCR uses TCR CDR3b sequences and peptide sequences as input. Additionally, users can also provide full length MHC to the tool. The output produces the predicted binding probability. 

This repository contains the code and the data to train [epiTCR](https://doi.org/10.1093/bioinformatics/btad284) model.


## Requirements

```text
python >= 3.0.0
numpy 1.22.4
scikit-learn 1.1.2
```
For other requirements, please see the `env_requirements.txt` file ([here](env_requirements.txt)).

## Run epiTCR
Users can run epiTCR in two modes: (i) train a new model and make prediction using the newly trained model, or (ii) make prediction using our pre-trained model.

**Train a new model and make prediction**

The main module of epiTCR is `epiTCR.py`. Users can train the epiTCR model (with or without MHC) and give prediction on their data by running:

```commandline
python3 epiTCR.py --trainfile data/splitData/withMHC/train/train.csv --testfile data/splitData/withMHC/test/test01.csv --chain cem
```
given that:
- `--trainfile` is a comma-separated file (CSV) containing columns for TCR, epitipe, binder, and/or full length MHC (reported by IMGT). See [example training data](data/splitData/withMHC/train/train.csv.zip).
- `--testfile` is a CSV file containing columns for TCR, epitope and/or full length MHC (reported by IMGT). See [example test file](data/splitData/withMHC/test/test01.csv).
- `--chain` specifies the chain(s) to use (ce, cem). Available options for this parameter are `ce` (cdr3b+epitope) and `cem` (cdr3b+epitope+mhc). Default as `ce`.

The prediction output is printed out on the standard output (std) or on a file (that can be specified using the option --outfile). For more information, view the section *Prediction output* below. 

**Run prediction using the pre-trained model**

Users can also apply our pre-trained model to directly make prediction on their data using the module `predict.py`. TCR-epitope or TCR-pMHC binding prediction can be run with:

```commandline
python3 predict.py --testfile data/splitData/withMHC/test/test01.csv --modelfile models/rdforestWithMHCModel.pickle --chain cem
```
given that:
- `--testfile` is a CSV file containing columns for TCR, epitipe and/or full length MHC reported by IMGT. See [example input file](data/splitData/withMHC/test/test01.csv).
- `--modelfile` specifies the full path of the file with trained model, should be a pickle files. Default model as `models/rdforestWithMHCModel.pickle`.
- `--chain` specifies the chain(s) to use (ce, cem). Options for this parameter are `ce` (cdr3b+epitope) and `cem` (cdr3b+epitope+mhc). Default as `ce`.

## Prediction output  

epiTCR prediction output contains a table with four columns: the CDR3b sequences, epitope sequences, (full length MHC,) and the binding probability for the corresponding complexes. The example output file is [here](data/test/output/output_prediction.csv).

## Contact

For more questions or feedback, please simply post an [Issue](https://github.com/ddiem-ri-4D/epiTCR/issues/new). 

## Citation
Please cite this paper if it helps your research:
```bibtex
@article{10.1093/bioinformatics/btad284,
    author = {Pham, My-Diem Nguyen and Nguyen, Thanh-Nhan and Tran, Le Son and Nguyen, Que-Tran Bui and Nguyen, Thien-Phuc Hoang and Pham, Thi Mong Quynh and Nguyen, Hoai-Nghia and Giang, Hoa and Phan, Minh-Duy and Nguyen, Vy},
    title = "{epiTCR: a highly sensitive predictor for TCR–peptide binding}",
    journal = {Bioinformatics},
    volume = {39},
    number = {5},
    pages = {btad284},
    year = {2023},
    month = {04},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btad284},
    url = {https://doi.org/10.1093/bioinformatics/btad284},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/39/5/btad284/50204900/btad284.pdf},
}
```

## References
