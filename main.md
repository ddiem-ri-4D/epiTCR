## Project Organization

    ├── README.md                                       <- The top-level README for users on epiTCR
    ├── data
    │   ├── convert-data                    
    │   │   │
    │   │   ├── 7-peptides                              <- One hot encoded input for 7 highly FP peptides into onehot encoding
    │   │   │
    │   │   ├── with-mhc                                <- One hot encoded input with MHC
    │   │   │
    │   │   └── without-mhc                             <- One hot encoded input without MHC
    │   │      
    │   ├── final-data                      
    │   │   │
    │   │   ├── final-with-HLA-without-full-length.csv  <- Preprocessed data with HLA without full length (for pMTnet)   
    │   │   │
    │   │   └── final-with-HLA.csv                      <- Preprocessed data with HLA in full length (for epiTCR-MHC)
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
    │   ├── random-sample-data                          <- Generate random sample data to compare running times tools.
    │   ├── similarity-score                            
    │   │   │                                           
    │   │   ├── cdr3b                                   <- Similarity score data with CDR3 for similarity ranges from 50-100%.
    │   │   │                                           
    │   │   └── epitope                                 <- Similarity score data with epitope for similarity ranges from 50-100%.
    │   ├── split-data                                  
    │   │   │                                           
    │   │   ├── 7-peptides                              <- Split raw data into 7 highly FP peptides data without MHC.
    │   │   │                                           
    │   │   ├── with-mhc                                <- Split raw data into 9 test sets with MHC.
    │   │   │                                           
    │   │   └── without-mhc                             <- Split raw data into 15 test sets without MHC.
    │   ├── supplementary-data                          <- Supplementary data for convert HLA Alele into protein sequence (365 aa).
    │   └── test                                            
    │       └── output                                  <- Prediction output file contains a table with four columns: the CDR3b sequences, epitope sequences, (full length MHC,) and the binding probability for the corresponding complexes
    │
    ├── models                                          <- Contains pre-trained model files.
    ├── secondary-analysis                              <- 
    ├── src                                             
    │   ├── benchmark                                   
    │   │   │                                           
    │   │   ├── compare-roc-auc-tools.ipynb             <- Python scripts for compare ROC AUC tools without MHC.
    │   │   │                                           
    │   │   ├── compare-with-mhc.ipynb                  <- Python scripts for compare ROC AUC tools with MHC.
    │   │   │                                           
    │   │   └── runtime.ipynb                           <- Python scripts for compare ROC AUC tools for running time without MHC and with MHC.
    │   ├── model-for-7-highly-fp-peptides              <- Python scripts for 7 highly FP peptides models.
    │   ├── modules                                     <- Directory wheret rained models, model predictions, summaries and
    │   │   ├── model.py                                <- Python scripts include functions for training models.
    │   │   │         
    │   │   ├── plot.py                                 <- Python scripts include functions for visualization results training models.
    │   │   │                                           
    │   │   ├── processor.py                            <- Python scripts to preprocess raw data.
    │   │   │                                                                             
    │   │   └── utils.py                                <- Python scripts include functions that process sequences.
    │   ├── predict_fulltest                           
    │   │   │                                           
    │   │   ├── with-mhc.ipynb                          <- Python scripts for full testset with MHC.
    │   │   │                                           
    │   │   └── without-mhc.ipynb                       <- Python scripts for full testset with MHC 
    │   ├── roc-auc-tools                               <- Python scripts for performance evaluation ROC AUC tools.
    │   ├── similarity-sequences                        <- Python Scripts to process data and train or evaluate models.
    │   │   │                                           
    │   │   ├── similarity-cdr3b.ipynb                  <- Python scripts with CDR3 for similarity ranges from 50-100%.
    │   │   │                                           
    │   │   └── similarity-epitope.ipynb                <- Python scripts with epitope for similarity ranges from 50-100%.
    │   ├── split-data                                 
    │   │   │                                           
    │   │   ├── split-data-mhc.ipynb                    <- Python scripts to split raw data into 9 test sets with MHC.
    │   │   │                                           
    │   │   └── split-data.ipynb                        <- Python scripts to split raw data into 15 test sets without MHC.
    │   └── hla2fulllength.ipynb                        <- Python scripts convert HLA Alele into protein sequence (365 aa).
    ├── env_requirements.txt                            <- The requirements file for reproducing the analysis environment, e.g.
    │                                                      generated with `pip freeze > env_requirements.txt`.
    │                                                      Usage: `pip install -r env_requirements.txt`.
    │
    ├── epiTCR.py                                       <- Python scripts to train the epiTCR model (with or without MHC) and give prediction.
    │                                                   
    └── predict.py                                      <- Python scripts pre-trained model to directly make prediction.
