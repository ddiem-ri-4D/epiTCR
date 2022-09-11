library(data.table)
library(ggplot2)
library(rquery)

calculate_metrics <- function(prob, label, p_thres) {
  auc <- acc <- sens <- spec <- NA
  pred <- ifelse(prob >= p_thres, 1, 0)
  if (length(unique(label)) == 2) {
    pr <- ROCR::prediction(pred, label)
    pe <- ROCR::performance(pr, "tpr", "fpr")
    roc.data <- data.frame(fpr=unlist(pe@x.values), tpr=unlist(pe@y.values))
    sens <- roc.data[2, 2]
    spec <- 1 - roc.data[2, 1]
    roc_obj <- pROC::roc(label, prob, levels = c(0, 1), direction = "<")
    auc <- pROC::auc(roc_obj)
  } else if (all(label == 1)) {
    sens <- length(pred[pred == 1])/length(label[label == 1])
  }
  acc <- length(pred[pred == label])/length(label)
  res <- list('auc' = auc, 'acc' = acc, 'sens' = sens, 'spec' = spec)
  return(res)
}

training <- fread('/Users/vy/Desktop/GS/TCR-epitope-prediction/train.csv', data.table = F)
training$type <- 'training'

all.wohla <- fread('/Users/vy/Desktop/GS/TCR-datasets/filtered/final-without-HLA.csv', data.table = F)
all.wohla$type <- 'original'

all_pred <- fread('/Users/vy/Downloads/prediction_out_full_testdata_nonmhc.csv', data.table = F)
dim(unique(all_pred))
dim(all_pred)
all_pred.merged <- merge(all_pred, training, by = c('CDR3b', 'epitope', 'binder'), all.x = T, all.y = T)
dim(unique(all_pred.merged))

all_pred.merged <- natural_join(all_pred.merged, all.wohla, by = c('CDR3b', 'epitope', 'binder'), jointype = 'FULL')

not.in.filtered <- all_pred.merged[is.na(all_pred.merged$type),]
filtered.not.in <- all_pred.merged[is.na(all_pred.merged$predict_proba) & all_pred.merged$type != 'training',]

all_pred.merged <- all_pred.merged[all_pred.merged$type != 'training',]
dim(all_pred.merged)
dim(training)

all_pred.merged <- all_pred.merged[, c(1:5)]
neo <- fread('/Users/vy/Downloads/Table S5.csv', data.table = F)
neo <- neo[neo$cancer != 'NA',]
neo <- neo[, c(1:7)]
neo.pred <- merge(neo, all_pred.merged, by = c('epitope'), all.x = F, all.y = F)
dim(neo.pred)

neo.pred.roc <- unique(neo.pred[, c(1, 8:11)])
write.table(neo.pred.roc, 'neo-pred.csv', quote = F, row.names = F, sep = ',')
calculate_metrics(neo.pred.roc$predict_proba, neo.pred.roc$binder, 0.5)

library(pROC)
roc.obj <- roc(binder ~ predict_proba, data = neo.pred.roc)
ggroc(roc.obj, legacy.axes = TRUE, color = 'red') + xlab("1- Specificity") + ylab("Sensitivity") +
  cowplot::theme_cowplot()

perf <- c(unlist(calculate_metrics(neo.pred.roc$predict_proba, neo.pred.roc$binder, 0.5)), 'overall')
for (group in unique(neo.pred$cancer.type.lev2)) {
  print(group)
  group.pred <- unique(neo.pred[neo.pred$cancer.type.lev2 == group, c('epitope', 'CDR3b', 'binder', 'binder_pred', 'predict_proba')])
  print(dim(group.pred))
  perf <- rbind(perf, c(unlist(calculate_metrics(group.pred$predict_proba, group.pred$binder, 0.5)), group))
}

perf <- as.data.frame(perf)
perf$auc <- as.numeric(perf$auc)
perf$acc <- as.numeric(perf$acc)
perf$sens <- as.numeric(perf$sens)
perf$spec <- as.numeric(perf$spec)

perf$V5 <- plyr::mapvalues(perf$V5, from = unique(perf$V5), to = c('Overall', 'Melanoma', 'Other', 'Breast cancer'))
perf$V5 <- factor(perf$V5, levels = c('Overall', 'Melanoma', 'Breast cancer', 'Other'))
perf.melt <- melt(perf, id.vars = c('V5'))
perf.melt$variable <- plyr::mapvalues(perf.melt$variable, from = unique(perf.melt$variable), to = c('AUC', 'Accuracy', 'Sensitivity', 'Specificity'))
ggplot(perf.melt, aes(x = V5, y = value, fill = V5)) +
  geom_bar(stat = 'identity') + facet_wrap(~ variable) +
  cowplot::theme_cowplot() + theme(axis.text.x=element_blank(),
                                   axis.ticks.x=element_blank()) + xlab('Groups') + ylab('Metrics') +
  guides(fill=guide_legend(title="Groups")) +
  geom_text(aes(label = round(value, 3)), stat = "identity", vjust = 0.5, hjust = 1, size = 4, colour = "white", angle = 90)
