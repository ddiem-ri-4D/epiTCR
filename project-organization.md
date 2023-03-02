## Project Organization

This document explains details about epiTCR project organization, ie. description about all folders and files of the project. This is for further exploration on epiTCR.

    ├── README.md                                       <- Instruction for instant use of epiTCR
    ├── data
    │   ├── finalData                                   <- Preprocessed data
    │   │   │
    │   │   ├── finalWithHLAConverted.csv               <- Preprocessed data with HLA  & Converted to pseudo sequence
    │   │   │
    │   │   └── finalWithoutHLA.csv                     <- Preprocessed data without HLA 
    |   |
    │   ├── get10Subsampling                            <- set 10 time samplings on epiTCR, ATMTCR and NetTCR data
    │   │
    │   ├── hlaConvertPeudoSeq                                                                     
    │   │   └── HLAWithPseudoSeq.csv                    <- Converted to pseudo sequence with HLA 
    |   |
    │   ├── nonOverlapingPeptide                             
    │   │   │
    │   │   ├── withMHC                                 <- Preprocessed unseen data with MLC
    │   │   │                                           
    │   │   └── withoutMHC                              <- Preprocessed unseen data with MLC
    │   │
    │   ├── pred7DominantPeptide                        <- 7 Dominant peptide prediction on full data   
    │   │
    │   ├── predepiTCRData                             
    │   │   │
    │   │   ├── fullTestset                             <- epiTCR prediction with full testset
    │   │   │
    │   │   ├── withMHC                                 <- epiTCR prediction with trained model on 9 testsets
    │   │   │                                                                                     
    │   │   └── withoutMHC                              <- epiTCR prediction with trained model on 15 testsets 
    │   │
    │   ├── predToolsData                             
    │   │   │
    │   │   ├── ATMTCR                                  <- ATM-TCR prediction with trained and re-train models on 15 testsets
    │   │   │
    │   │   ├── Imrex                                   <- Imrex prediction with trained model on 15 testsets
    │   │   │                                           
    │   │   ├── NetTCR                                  <- NetTCR prediction with trained and re-trained models on 15 testsets
    │   │   │                                           
    │   │   ├── pMTnet                                  <- pMTnet prediction with trained model on 9 testsets 
    |   │   │  
    |   |   └── outputPerformance                            
    |   │      │
    |   │      ├── ATMTCR                               <- ATM-TCR performance metrics with trained and re-train models on 15 testsets
    |   │      │
    |   │      ├── Imrex                                <- Imrex performance metrics with trained model on 15 testsets
    |   │      │                                        
    |   │      ├── NetTCR                               <- NetTCR predperformance metricsiction with trained and re-trained models on 15 testsets   
    │   │      │                              
    |   │      └── pMTnet                               <- pMTnet performance metrics with trained model on 9 testsets 
    │   │
    │   ├── randomSampleData                            
    │   │   │
    │   │   ├── epiTCR                                  <- epiTCR get randomly generated data for runtime benchmark
    │   │   │
    │   │   ├── Imrex                                   <- Imrex get randomly generated data for runtime benchmark
    │   │   │                                           
    │   │   ├── NetTCR                                  <- NetTCR get randomly generated data for runtime benchmark
    │   │   │                                           
    │   │   └── pMTnet                                  <- pMTnet get randomly generated data for runtime benchmark
    │   │
    │   ├── runTimeData                                 <- prepare data for runtime benchmark
    │   │  
    │   ├── setDataPredict10Subs                        <- prepare data for 10-time subsampling benchmark
    │   │
    │   ├── similarityScore                                                                                                   
    │   │   └── matrixLevenSimiEpi.csv                  <- Matrix similarity of epitope sequences using measure LV
    │   │
    │   ├── splitData                                   <- Observations splitted for different analysis
    │   │   │                                                                                     
    │   │   ├── withMHC                                 <- Observations splitted into nine test sets with MHC
    │   │   │                                           
    │   │   └── withoutMHC                              <- Observations splitted into 15 test sets without MHC
    │   │
    │   └── test                                            
    │       └── output                                  <- Prediction output 
    │
    ├── models                                          <- epiTCR trained models
    │
    ├── secondary-analysis                              <- Secondary analyses for the manuscript, incl. similarity, dominant peptides, and neoantigens
    │
    ├── src      
    │   ├── get10Subsampling                            <- python scripts for get 10 time-subsampling; benchmark with epiTCR, ATMTCR and NetTCR data
    │   │                                     
    │   ├── benchmark                                   
    │   │   │                                           
    │   │   ├── compareROCAUCTools.py                   <- Python scripts for epiTCR benchmark
    │   │   │                                           
    │   │   ├── compareWithMHC.py                       <- Python scripts for epiTCR-MHC benchmark
    │   │   │                                           
    │   │   └── runTime.py                              <- Python scripts for runtime benchmark
    │   │
    │   ├── calSimilarityByLV                           <- Calculate similarity of epitope sequences using measure LV
    │   │
    │   ├── hlaConvertPseudoSeq                         <- Convert hla to pseudo sequence
    │   │
    │   ├── model7DominantPeptide                       <- Python scripts for training individual models for seven dominant peptides
    │   │
    │   ├── modelWithUnseenData                         
    │   │   │
    │   │   ├── epiTCR2.py                              <- Main module of epiTCR on unseen data
    │   │   │                                                                             
    │   │   └── peptide2.py                             <- LModule for epiTCR pre-trained model on unseen data
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
    │   ├── predFulltest                            
    │   │   │                                           
    │   │   ├── withMHC.py                              <- Scripts for full prediction on observations with MHC
    │   │   │                                           
    │   │   └── withoutMHC.py                           <- Scripts for full prediction on observations without MHC 
    │   │
    │   ├── rocAUCTools                                 <- Scripts for performance evaluation
    │   │
    │   └── splitData                                
    │       │                                           
    │       ├── splitDataWithMHC.py                     <- Scripts for data generation into nine test sets with MHC
    │       │                                           
    │       └── splitDataWithoutMHC.py                  <- Scripts for data generation into 15 test sets without MHC
    │
    ├── env_requirements.txt                            <- Requirements file for reproducing the analysis environment
    │
    ├── epiTCR.py                                       <- Main module of epiTCR
    │                                                   
    └── predict.py                                      <- Module for epiTCR pre-trained model
