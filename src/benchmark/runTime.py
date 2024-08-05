import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to read and sample datasets
def read_and_sample(file_paths, sample_sizes, output_path_prefix, columns=None):
    datasets = [pd.read_csv(file) for file in file_paths]
    data = pd.concat(datasets, axis=0).reset_index(drop=True)
    
    for size in sample_sizes:
        sampled_data = data.sample(n=size, random_state=42).reset_index(drop=True)
        sampled_data.to_csv(f'{output_path_prefix}{size}.csv', index=False)

# epiTCR data preparation
epiTCR_file_paths = [f"../../data/splitData/withoutMHC/test/test{i:02d}.csv" for i in range(1, 16)]
epiTCR_mhc_file_paths = [f"../../data/splitData/withoutMHC/test/test{i:02d}.csv" for i in range(1, 10)]
sample_sizes = [10000, 50000, 200000, 500000, 1000000]

read_and_sample(epiTCR_file_paths, sample_sizes, "../../data/randomSampleData/epiTCR/withoutMHC/dataTest")
read_and_sample(epiTCR_mhc_file_paths, sample_sizes[:-1], "../../data/randomSampleData/epiTCR/withMHC/dataTest")

# pMTnet data preparation
pMTnet_file_paths = [f"../../data/runTimeData/pMTnet/test{i:02d}.csv" for i in range(1, 10)]
normalized_pMTnet_file_paths = [f"../../data/runTimeData/pMTnet/normalizeData/test{i:02d}.csv" for i in range(1, 10)]

def read_and_sample_pMTnet(file_paths, normalized_file_paths, sample_sizes, output_path_prefix):
    y_data = [pd.read_csv(file).iloc[:, 3:] for file in file_paths]
    X_data = [pd.read_csv(file) for file in normalized_file_paths]
    data = pd.concat([pd.concat([X, y], axis=1) for X, y in zip(X_data, y_data)], axis=0).reset_index(drop=True)
    
    for size in sample_sizes:
        sampled_data = data.sample(n=size, random_state=42).reset_index(drop=True)
        X_sampled = sampled_data.iloc[:, :3]
        X_sampled.to_csv(f'{output_path_prefix}{size}.csv', index=False)

read_and_sample_pMTnet(pMTnet_file_paths, normalized_pMTnet_file_paths, sample_sizes[:-1], "../../data/randomSampleData/pMTnet/dataTestpMTnet")

# Imrex data preparation
imrex_file_paths = [f"../../data/runTimeData/Imrex/ptest{i:02d}.csv" for i in range(1, 16)]
read_and_sample(imrex_file_paths, sample_sizes, "../../data/randomSampleData/Imrex/dataTestImrex")

# NetTCR data preparation
nettcr_file_paths = [f"../../data/runTimeData/NetTCR/test{i:02d}.csv" for i in range(1, 16)]
read_and_sample(nettcr_file_paths, sample_sizes, "../../data/randomSampleData/NetTCR/dataTestNetTCR")

# Benchmark Running Time Plot
tmp_dataset = pd.DataFrame(([["  10000",  18.817,   7.649,   2.662,  2489.677,   1.000,    1.431], 
                             ["  50000",  40.415,  17.154,  10.220, 12295.409,   4.509,    7.159], 
                             [" 200000", 132.958,  40.154,  22.604, 51264.141,  17.975,   27.638], 
                             [" 500000", 330.146,  94.591,  54.236,132951.125,  48.843,   70.597], 
                             ["1000000", 660.439, 179.598, 101.045,262892.017,  94.023,  142.194]]), 
                             columns=['count_sample', 'ATM-TCR', 'Imrex', 'NetTCR',
                                      'pMTnet', 'epiTCR','epiTCR_mhc'])

tmp_dataset[["ATM-TCR_log", "Imrex_log", "NetTCR_log", "pMTnet_log", "epiTCR_log", "epiTCR_mhc_log"]] = np.log10(tmp_dataset[["ATM-TCR", "Imrex", "NetTCR", "pMTnet", "epiTCR", "epiTCR_mhc"]])

# Plotting
def plot_runtime(data, columns, labels, output_path):
    for col, label in zip(columns, labels):
        plt.plot(data['count_sample'], data[col], label=label, marker='o')
    plt.xlabel('Dataset size')
    plt.ylabel('Runtime (log10(s))')
    plt.legend()
    plt.savefig(output_path + ".png")
    plt.savefig(output_path + ".pdf")
    plt.show()

plot_runtime(tmp_dataset, ["ATM-TCR_log", "Imrex_log", "NetTCR_log", "epiTCR_log"], ['ATM-TCR', 'Imrex', 'NetTCR', 'epiTCR'], "../../analysis/figures/benchmarkRunTime")
plot_runtime(tmp_dataset, ["pMTnet_log", "epiTCR_mhc_log"], ['pMTnet', 'epiTCR'], "../../analysis/figures/benchmarkRunTimeWithMHC")
