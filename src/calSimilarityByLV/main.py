import pandas as pd
from thefuzz import process, fuzz
import modules.processor as Processor

# Read the training and test data
data_train = pd.read_csv("../../data/nonOverlapingPeptide/withoutMHC/train/train.csv")
data_test = pd.read_csv("../../data/nonOverlapingPeptide/withoutMHC/test/test.csv")

# Combine training and test data
data_full = pd.concat([data_train, data_test], axis=0)

# Get unique epitopes
unique_epitopes = data_full['epitope'].unique().tolist()
length = len(unique_epitopes)

# Compute similarity scores
score_sort_epitope = [(epitope,) + match
                      for epitope in unique_epitopes 
                      for match in process.extract(epitope, unique_epitopes, scorer=fuzz.token_sort_ratio, limit=4120)]

# Create a DataFrame with similarity scores
similarity_sort = pd.DataFrame(score_sort_epitope, columns=['epitope', 'match_sort', 'score_sort'])
print(similarity_sort.head(10))

# Split the DataFrame by position
split_dataframes = Processor.splitDataframeByPosition(similarity_sort, length)

# Process each split DataFrame
processed_dfs = []
for i in range(length):
    temp_df = split_dataframes[i].reset_index().pivot('epitope', 'match_sort', 'score_sort').reset_index().rename_axis(None, axis=1)
    processed_dfs.append(temp_df)

# Concatenate processed DataFrames and sort by 'epitope'
result_df = pd.concat(processed_dfs)
sorted_result_df = result_df.sort_values('epitope').reset_index(drop=True)
print(sorted_result_df.tail(5))

# Save the resulting DataFrame to a CSV file
sorted_result_df.to_csv("../../data/similarityScore/matrixLevenSimiEpi.csv", index=False)
