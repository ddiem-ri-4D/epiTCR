import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("../../data/finalData/finalWithoutHLA.csv")

# Drop duplicates and select relevant columns
df = df.drop_duplicates()
df = df[["CDR3b", "epitope", "binder"]]
df = df.drop_duplicates()

# Split data into features and labels
X = df[['CDR3b', 'epitope']]
y = df[['binder']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.88, random_state=42)

# Further split the test set into smaller test sets
X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.14, random_state=42)
X_test3, X_test4, y_test3, y_test4 = train_test_split(X_test1, y_test1, test_size=0.165, random_state=42)
X_test5, X_test6, y_test5, y_test6 = train_test_split(X_test3, y_test3, test_size=0.2, random_state=42)
X_test7, X_test8, y_test7, y_test8 = train_test_split(X_test5, y_test5, test_size=0.2, random_state=42)
X_test9, X_test10, y_test9, y_test10 = train_test_split(X_test7, y_test7, test_size=0.2, random_state=42)
X_test11, X_test12, y_test11, y_test12 = train_test_split(X_test9, y_test9, test_size=0.2, random_state=42)

# Save full test and training sets
test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
test.to_csv("../../data/splitData/withMHC/test/fulltest.csv", index=False)

train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)

# Split training data into two parts
train_1 = train[train["binder"] == 1]
train_1_1 = train_1.iloc[:6684, :]
train_1_2 = train_1.iloc[6684:, :]

train_0 = train[train["binder"] == 0]
train_0_2 = train_0.iloc[0:62950, :]
train_0_1 = train_0.iloc[62950:, :]

train = pd.concat([train_1_1, train_0_1], axis=0).reset_index(drop=True)
trainp = pd.concat([train_1_2, train_0_2], axis=0).reset_index(drop=True)

train.to_csv("../../data/splitData/withoutMHC/train/train.csv", index=False)
trainp.to_csv("../../data/splitData/withoutMHC/test/test15.csv", index=False)

full_test = pd.concat([test, trainp], axis=0).reset_index(drop=True)
full_test.to_csv("../../data/splitData/withoutMHC/test/full_test.csv", index=False)

# Function to create balanced test sets and save them
def create_balanced_test_set(X, y, sample_size, file_prefix):
    test_set = pd.concat([X, y], axis=1).reset_index(drop=True)
    test_set_1 = test_set[test_set["binder"] == 1]
    test_set_1_1 = test_set_1.iloc[:sample_size, :]
    test_set_1_2 = test_set_1.iloc[sample_size:, :]
    test_set_0 = test_set[test_set["binder"] == 0]
    test_set_0_1 = test_set_0.sample(n=sample_size * 10, random_state=1)
    test_set_0_2 = test_set_0.sample(n=sample_size * 10, random_state=2)
    test_set = pd.concat([test_set_1_1, test_set_0_1], axis=0).reset_index(drop=True)
    test_set_p = pd.concat([test_set_1_2, test_set_0_2], axis=0).reset_index(drop=True)
    test_set.to_csv(f"../../data/splitData/withoutMHC/test/{file_prefix}.csv", index=False)
    test_set_p.to_csv(f"../../data/splitData/withoutMHC/test/{file_prefix}_p.csv", index=False)

# Create and save balanced test sets
create_balanced_test_set(X_test2, y_test2, 6628, "test01")
create_balanced_test_set(X_test4, y_test4, 6695, "test03")
create_balanced_test_set(X_test6, y_test6, 6728, "test05")
create_balanced_test_set(X_test8, y_test8, 6667, "test07")
create_balanced_test_set(X_test10, y_test10, 6652, "test09")
create_balanced_test_set(X_test12, y_test12, 6688, "test11")
create_balanced_test_set(X_test11, y_test11, 6738, "test13")
