## Project Organization

    ├── README.md                           <- The top-level README for developers using this project.
    ├── data
    │   ├── convert-data                    
    │   │   │
    │   │   ├── 7-peptides                  <- Representation of input data of 7 highly FP peptides into onehot encoding.
    │   │   │
    │   │   ├── with-mhc                    <- Representation of input data with mhc into onehot encoding.
    │   │   │
    │   │   └── without-mhc                 <- Representation of input data without mhc into onehot encoding.
    │   ├── final-data                      <- Directory wheret rained models, model predictions, summaries and
    │   ├── output-performance              <- Directory wheret rained models, model predictions, summaries and
    │   ├── pred-tools-data                 <- Directory wheret rained models, model predictions, summaries and
    │   │   │
    │   │   ├── atm-tcr                     <- Contains a small number of pre-trained models.
    │   │   │
    │   │   ├── imrex                       <- Intermediate data that has been transformed.
    │   │   │
    │   │   ├── nettcr                      <- Intermediate data that has been transformed.
    │   │   │ 
    │   │   └── pmtnet                      <- The original, immutable data dump. See README for more info.
    │   ├── predict-data                    <- Directory wheret rained models, model predictions, summaries and
    │   │   │
    │   │   ├── 7-peptides                  <- Contains a small number of pre-trained models.
    │   │   │
    │   │   ├── full-testset                <- Intermediate data that has been transformed.
    │   │   │
    │   │   ├── with-mhc                    <- Intermediate data that has been transformed.
    │   │   │
    │   │   └── without-mhc                 <- The original, immutable data dump. See README for more info.
    │   ├── random-sample-data              <- Directory wheret rained models, model predictions, summaries and
    │   ├── similarity-score                <- Directory wheret rained models, model predictions, summaries and
    │   │   │
    │   │   ├── cdr3b                       <- Contains a small number of pre-trained models.   
    │   │   │
    │   │   └── epitope                     <- The original, immutable data dump. See README for more info.
    │   ├── split-data                      <- Scripts to process data and train or evaluate models.
    │   │   │
    │   │   ├── 7-peptides                  <- Bash scripts to download VDJdb data. Used by `Makefile`.
    │   │   │
    │   │   ├── with-mhc                    <- Python scripts to preprocess raw data. Used by `Makefile`.
    │   │   │                         
    │   │   └── without-mhc                 <- Scripts to create exploratory and results oriented 
    │   └──supplementary-data               <- The original, immutable data dump. See README for more info.
    │
    ├── models                              <- Jupyter notebooks. Used for generation of additional figures and
    ├── secondary-analysis                  <- Jupyter notebooks. Used for generation of additional figures and
    ├── src
    │   ├── benchmark                       <- Scripts to process data and train or evaluate models.
    │   │   │
    │   │   ├── compare-roc-auc-tools.ipynb <- Bash scripts to download VDJdb data. Used by `Makefile`.
    │   │   │
    │   │   ├── compare-with-mhc.ipynb      <- Bash scripts to download VDJdb data. Used by `Makefile`.
    │   │   │
    │   │   └── runtime.ipynb               <- Scripts to create exploratory and results oriented 
    │   ├── data-for-runtime                <- Directory wheret rained models, model predictions, summaries and
    │   │   │
    │   │   ├── imrex                       <- Contains a small number of pre-trained models.
    │   │   │
    │   │   ├── nettcr                      <- Intermediate data that has been transformed.
    │   │   │
    │   │   └── pmtnet                      <- The original, immutable data dump. See README for more info.
    │   ├── model-for-7-highly-fp-peptides  <- Directory wheret rained models, model predictions, summaries and
    │   ├── modules                         <- Directory wheret rained models, model predictions, summaries and
    │   ├── predict_fulltest                <- Scripts to process data and train or evaluate models.
    │   │   │
    │   │   ├── with-mhc.ipynb              <- Bash scripts to download VDJdb data. Used by `Makefile`.
    │   │   │
    │   │   └── without-mhc.ipynb           <- Scripts to create exploratory and results oriented 
    │   ├── roc-auc-tools                   <- Scripts to process data and train or evaluate models.
    │   ├── similarity-sequences            <- Scripts to process data and train or evaluate models.
    │   │   │
    │   │   ├── similarity-cdr3b.ipynb      <- Bash scripts to download VDJdb data. Used by `Makefile`.
    │   │   │
    │   │   └── similarity-epitope.ipynb    <- Scripts to create exploratory and results oriented 
    │   ├── split-data                      <- Scripts to process data and train or evaluate models.
    │   │   │
    │   │   ├── split-data-mhc.ipynb        <- Bash scripts to download VDJdb data. Used by `Makefile`.
    │   │   │
    │   │   └── split-data.ipynb            <- Scripts to create exploratory and results oriented 
    │   └── hla2fulllength.ipynb            <- Generated graphics and figures to be used in reporting.
    ├── test
    │   └── output                          <- Generated graphics and figures to be used in reporting.
    ├── requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g.
    │                                      generated with `pip freeze > requirements.txt`.
    │                                      Usage: `pip install -r requirements.txt`.
    │
    ├── epiTCR.py                          <- makes project pip installable (pip install -e .) so src can be imported.
    │
    └── predict.py                         <- tox file with settings for running pytest.
