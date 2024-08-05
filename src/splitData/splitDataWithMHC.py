import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("../../data/finalData/finalWithHLA.csv")

# Select relevant columns
df = df[["CDR3b", "epitope", "binder", "HLA"]]
X = df[["CDR3b", "epitope", "HLA"]]
y = df["binder"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# Further split the test set into smaller test sets
X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.111, random_state=42)
X_test3, X_test4, y_test3, y_test4 = train_test_split(X_test1, y_test1, test_size=0.125, random_state=42)
X_test5, X_test6, y_test5, y_test6 = train_test_split(X_test3, y_test3, test_size=0.145, random_state=42)
X_test7, X_test8, y_test7, y_test8 = train_test_split(X_test5, y_test5, test_size=0.17, random_state=42)
X_test9, X_test10, y_test9, y_test10 = train_test_split(X_test7, y_test7, test_size=0.205, random_state=42)
X_test11, X_test12, y_test11, y_test12 = train_test_split(X_test9, y_test9, test_size=0.25, random_state=42)
X_test13, X_test14, y_test13, y_test14 = train_test_split(X_test11, y_test11, test_size=0.33, random_state=42)
X_test15, X_test16, y_test15, y_test16 = train_test_split(X_test13, y_test13, test_size=0.5, random_state=42)

# Save full test and training sets
test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
test.to_csv("../../data/splitData/withMHC/test/fulltest.csv", index=False)

train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
train.to_csv("../../data/splitData/withoutMHC/train/train.csv", index=False)

# Function to create balanced test sets
def create_balanced_test_set(X, y, sample_size, filename):
    test_set = pd.concat([X, y], axis=1).reset_index(drop=True)
    test_set_binders = test_set[test_set["binder"] == 1]
    test_set_non_binders = test_set[test_set["binder"] == 0].sample(n=sample_size, random_state=1)
    balanced_test_set = pd.concat([test_set_non_binders, test_set_binders], axis=0).reset_index(drop=True)
    balanced_test_set.to_csv(filename, index=False)

# Create and save balanced test sets
create_balanced_test_set(X_test2, y_test2, 6682 * 10, "../../data/splitData/withoutMHC/test/test01.csv")
create_balanced_test_set(X_test4, y_test4, 6587 * 10, "../../data/splitData/withoutMHC/test/test02.csv")
create_balanced_test_set(X_test6, y_test6, 6721 * 10, "../../data/splitData/withoutMHC/test/test03.csv")
create_balanced_test_set(X_test8, y_test8, 6728 * 10, "../../data/splitData/withoutMHC/test/test04.csv")
create_balanced_test_set(X_test10, y_test10, 6843 * 10, "../../data/splitData/withoutMHC/test/test05.csv")
create_balanced_test_set(X_test12, y_test12, 6685 * 10, "../../data/splitData/withoutMHC/test/test06.csv")
create_balanced_test_set(X_test14, y_test14, 6387 * 10, "../../data/splitData/withoutMHC/test/test07.csv")
create_balanced_test_set(X_test16, y_test16, 6584 * 10, "../../data/splitData/withoutMHC/test/test08.csv")
create_balanced_test_set(X_test15, y_test15, 6580 * 10, "../../data/splitData/withoutMHC/test/test09.csv")
