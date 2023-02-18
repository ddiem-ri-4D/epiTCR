library(data.table)
library(ggplot2)
library(dplyr)

dominant <- c('NLVPMVATV', 'KLGGALQAK', 'GILGFVFTL', 'TPRVTGGGAM', 'GLCTLVAML', 'AVFDRKSDAK',
              'ELAGIGILTV')

train <- fread('/Users/vy/Desktop/GS/TCR-epitope-prediction/train.csv', data.table =  F)
test01 <- fread('/Users/vy/Desktop/06032022_predict_15testsets/predict_test15.csv', data.table = F)
test01.dominant <- test01[test01$epitope %in% dominant,]
calculate_metrics(test01.dominant$predict_proba, test01.dominant$binder, 0.5)

intrain <- test01[test01$epitope %in% train$epitope,]
intrain.ndominant <- intrain[!intrain$epitope %in% dominant,]

intrain.dominant <- intrain[intrain$epitope %in% dominant,]
train.ndominant <- train[train$epitope %in% intrain.ndominant$epitope,]
write.table(train.ndominant, '/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/alternative_train.csv', quote = F, row.names = F, col.names = T, append = F, sep = ',')

#-- epitope not in train
ntrain <- test01[!test01$epitope %in% train$epitope,]
table(ntrain$cat)

#---
train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.pos)
alter.train.dominant.pos <- dplyr::sample_n(train.dominant.pos, 690)
alter.train.dominant <- rbind(alter.train.dominant.pos, train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/alternative_train_v2.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
alter.train <- rbind(train.dominant.neg, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/alternative_train_v3.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.pos)
alter.train.dominant.pos <- dplyr::sample_n(train.dominant.pos, 1277) # 0.03
alter.train.dominant <- rbind(alter.train.dominant.pos, train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/alternative_train_v4.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.pos)
alter.train.dominant.pos <- dplyr::sample_n(train.dominant.pos, 1703) # 0.04
alter.train.dominant <- rbind(alter.train.dominant.pos, train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/alternative_train_v5.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.pos)
alter.train.dominant.pos <- dplyr::sample_n(train.dominant.pos, 2128) # 0.05
alter.train.dominant <- rbind(alter.train.dominant.pos, train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/alternative_train_v6.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.neg)
alter.train.dominant.neg <- dplyr::sample_n(train.dominant.neg, 37765) # 0.06
alter.train.dominant <- rbind(train.dominant.pos, alter.train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/alternative_train_v7.csv', quote = F, sep = ',', row.names = F, append = F)
