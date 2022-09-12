## Project Organization

This document explains details about epiTCR project organization, ie. description about all folders and files of the project. This is for further exploration on epiTCR.

    ├── README.md                                       <- Instruction for instant use of epiTCR
    ├── data
    │   ├── convert-data                                <- One hot encoded data
    │   │   │
    │   │   ├── 7-peptides                              <- One hot encoded data for seven dominant peptides
    │   │   │
    │   │   ├── with-mhc                                <- One hot encoded data with MHC
    │   │   │
    │   │   └── without-mhc                             <- One hot encoded  without MHC
    │   │      
    │   ├── final-data                                  <- Preprocessed data
    │   │   │
    │   │   ├── final-without-HLA.csv                   <- Preprocessed data without HLA   
    │   │   │
    │   │   └── final-with-HLA.csv                      <- Preprocessed data with HLA in full length
    |   |
    │   ├── pred-tools-data                             
    │   │   │
    │   │   ├── atm-tcr                                 <- ATM-TCR prediction with trained and re-train models on 15 testsets
    │   │   │
    │   │   ├── imrex                                   <- Imrex prediction with trained model on 15 testsets
    │   │   │                                           
    │   │   ├── nettcr                                  <- NetTCR prediction with trained and re-trained models on 15 testsets
    │   │   │                                           
    │   │   └── pmtnet                                  <- pMTnet prediction with trained model on 9 testsets 
    │   │
    │   ├── predict-data                                <- Prediction of epiTCR
    │   │   │                                           
    │   │   ├── 7-peptides                              <- Prediction on observations related to seven dominant peptides without MHC
    │   │   │                                           
    │   │   ├── full-testset                            <- Prediction on all test sets (with MHC and without MHC)
    │   │   │                                           
    │   │   ├── with-mhc                                <- Prediction on 9 testsets with MHC
    │   │   │                                           
    │   │   └── without-mhc                             <- Prediction on 15 testsets without MHC
    │   │
    │   ├── random-sample-data                          <- Randomly generated data for runtime benchmark
    │   │
    │   ├── similarity-score                            <- Data for sequence similarity analysis
    │   │   │                                           
    │   │   ├── cdr3b                                   <- Similarity on CDR3b sequences
    │   │   │                                           
    │   │   └── epitope                                 <- Similarity on epitope
    │   │
    │   ├── split-data                                  <- Observations splitted for different analysis
    │   │   │                                           
    │   │   ├── 7-peptides                              <- Observations for seven dominant peptides
    │   │   │                                           
    │   │   ├── with-mhc                                <- Observations splitted into nine test sets with MHC
    │   │   │                                           
    │   │   └── without-mhc                             <- Observations splitted into 15 test sets without MHC
    │   │
    │   ├── supplementary-data                          <- Supplementary data for conversion of HLA typing to full length sequence following IMGT data
    │   │
    │   └── test                                            
    │       └── output                                  <- Prediction output 
    │
    ├── models                                          <- epiTCR trained models
    │
    ├── secondary-analysis                              <- Secondary analyses for the manuscript, incl. similarity, dominant peptides, and neoantigens
    │
    ├── src                                             
    │   ├── benchmark                                   
    │   │   │                                           
    │   │   ├── compare-roc-auc-tools.ipynb             <- Python scripts for epiTCR benchmark
    │   │   │                                           
    │   │   ├── compare-with-mhc.ipynb                  <- Python scripts for epiTCR-MHC benchmark
    │   │   │                                           
    │   │   └── runtime.ipynb                           <- Python scripts for runtime benchmark
    │   │
    │   ├── model-for-7-highly-fp-peptides              <- Python scripts for training individual models for seven dominant peptides
    │   │
    │   ├── modules                                     <- Libraries for model training, evaluation and prediction
    │   │   │
    │   │   ├── model.py                                <- Libary for training models
    │   │   │         
    │   │   ├── plot.py                                 <- Library for visualization in model evaluation
    │   │   │                                           
    │   │   ├── processor.py                            <- Library for data representation
    │   │   │                                                                             
    │   │   └── utils.py                                <- Library for sequence processing
    │   │
    │   ├── predict_fulltest                            
    │   │   │                                           
    │   │   ├── with-mhc.ipynb                          <- Scripts for full prediction on observations with MHC
    │   │   │                                           
    │   │   └── without-mhc.ipynb                       <- Scripts for full prediction on observations without MHC 
    │   │
    │   ├── roc-auc-tools                               <- Scripts for performance evaluation
    │   │
    │   ├── similarity-sequences                        <- Scripts for data processing based on sequence similarity
    │   │   │                                           
    │   │   ├── similarity-cdr3b.ipynb                  <- Scripts for data processing based on CDR3 similarity
    │   │   │                                           
    │   │   └── similarity-epitope.ipynb                <- Scripts for data processing based on epitope similarity
    │   │
    │   ├── split-data                                 
    │   │   │                                           
    │   │   ├── split-data-mhc.ipynb                    <- Scripts for data generation into nine test sets with MHC
    │   │   │                                           
    │   │   └── split-data.ipynb                        <- Scripts for data generation into 15 test sets without MHC
    │   │
    │   └── hla2fulllength.ipynb                        <- Scripts for conversion from HLA typing to full length sequence following IGMT 
    │
    ├── env_requirements.txt                            <- Requirements file for reproducing the analysis environment
    │
    ├── epiTCR.py                                       <- Main module of epiTCR
    │                                                   
    └── predict.py                                      <- Module for epiTCR pre-trained model
