import pandas as pd
from thefuzz import process, fuzz
import modules.processor as Processor
import pandas as pd
from thefuzz import fuzz
from thefuzz import process


DATA_TRAIN = pd.read_csv("../../data/nonOverlapingPeptide/withoutMHC/train/train.csv")
DATA_TEST = pd.read_csv("../../data/nonOverlapingPeptide/withoutMHC/test/test.csv")

DATA_FULL = pd.concat([DATA_TRAIN, DATA_TEST], axis = 0)


uni_epitope = DATA_FULL['epitope'].unique().tolist()
LENGHT = len(uni_epitope)

score_sort_epitope = [(x,) + i
                     for x in uni_epitope 
                     for i in process.extract(x, uni_epitope, scorer=fuzz.token_sort_ratio, limit=4120)]

similarity_sort = pd.DataFrame(score_sort_epitope, columns=['epitope','match_sort','score_sort'])
similarity_sort.head(10)

sp_dataframe = Processor.splitDataframeByPosition(similarity_sort, LENGHT)

res = []
for i in range(LENGHT):
    temp = sp_dataframe[i].reset_index().pivot('epitope', 'match_sort', 'score_sort').\
                           reset_index().rename_axis(None, axis=1)
    res.append(temp)
pdList = [res[i] for i in range(LENGHT)]  
new_df = pd.concat(pdList)
sort_new_df = new_df.sort_values('epitope').reset_index(drop=True)
print(sort_new_df.tail(5))

sort_new_df.to_csv("../../data/similarityScore/matrixLevenSimiEpi.csv", index=False)