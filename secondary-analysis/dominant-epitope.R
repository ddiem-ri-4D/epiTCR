library(data.table)
library(ggplot2)
library(dplyr)

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

train <- fread('/Users/vy/Desktop/GS/TCR-epitope-prediction/train.csv', data.table =  F)
test01 <- fread('/Users/vy/Desktop/GS/TCR-epitope-prediction/predict_proba_withoutMHC/test01_predict_proba.csv', data.table = F)
test01.dominant <- test01[test01$epitope %in% dominant,]
calculate_metrics(test01.dominant$predict_proba, test01.dominant$binder, 0.5)

p_thres <- 0.5
test01$pred <- ifelse(test01$predict_proba >= p_thres, 1, 0)
test01$cat <- ifelse(test01$binder == 1 & test01$pred == 0, 'FN',
                     ifelse(test01$binder == 0 & test01$pred == 1, 'FP',
                            ifelse(test01$binder == 1 & test01$pred == 1, 'TP', 'TN')))

test01.FP <- test01[test01$cat == 'FP',]
test01.FP$group <- ifelse(test01.FP$epitope %in% dominant, 'Dominant peptides', 'Other peptides')
fp <- as.data.frame(table(test01.FP$group)) %>% mutate(perc = Freq / sum(Freq)) %>%
  mutate(labels = round(perc, 4) * 100)
ggplot(fp, aes(x="", y=perc, fill=Var1)) +
  geom_col() +
  geom_label(aes(label = as.character(paste0(labels, '%'))), color = c("white", "white"),
             position = position_stack(vjust = 0.5),
             show.legend = FALSE) +
  guides(fill = guide_legend(title = "False positives")) +
  coord_polar("y", start=0) +
  theme_void() + guides()

test01$group <- ifelse(test01$epitope %in% dominant, 'Dominant peptides', 'Other peptides')
full <- as.data.frame(table(test01$group)) %>% mutate(perc = Freq / sum(Freq)) %>%
  mutate(labels = round(perc, 4) * 100)
ggplot(full, aes(x="", y=perc, fill=Var1)) +
  geom_col() +
  geom_label(aes(label = as.character(paste0(labels, '%'))), color = c("white", "white"),
             position = position_stack(vjust = 0.5),
             show.legend = FALSE) +
  guides(fill = guide_legend(title = "Contribution to test set")) +
  coord_polar("y", start=0) +
  theme_void() + guides()

intrain <- test01[test01$epitope %in% train$epitope,]
table(intrain$cat)
calculate_metrics(intrain$predict_proba, intrain$binder, 0.5)

dominant <- c('NLVPMVATV', 'KLGGALQAK', 'GILGFVFTL', 'TPRVTGGGAM', 'GLCTLVAML', 'AVFDRKSDAK',
              'ELAGIGILTV')
intrain.ndominant <- intrain[!intrain$epitope %in% dominant,]
calculate_metrics(intrain.ndominant$predict_proba, intrain.ndominant$binder, 0.5)

intrain.dominant <- intrain[intrain$epitope %in% dominant,]
calculate_metrics(intrain.dominant$predict_proba, intrain.dominant$binder, 0.5)

train.ndominant <- train[train$epitope %in% intrain.ndominant$epitope,]
table(train.ndominant$binder)
write.table(train.ndominant, '/Users/vy/Desktop/GS/TCR-epitope-prediction/alternative_train.csv', quote = F, row.names = F, col.names = T, append = F, sep = ',')

#-- epitope not in train
ntrain <- test01[!test01$epitope %in% train$epitope,]
table(ntrain$cat)

#--- generate other training sets
train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.pos)
alter.train.dominant.pos <- dplyr::sample_n(train.dominant.pos, 690)
alter.train.dominant <- rbind(alter.train.dominant.pos, train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/Desktop/GS/TCR-epitope-prediction/alternative_train_v2.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
alter.train <- rbind(train.dominant.neg, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/Desktop/GS/TCR-epitope-prediction/alternative_train_v3.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.pos)
alter.train.dominant.pos <- dplyr::sample_n(train.dominant.pos, 1277) # 0.03
alter.train.dominant <- rbind(alter.train.dominant.pos, train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/Desktop/GS/TCR-epitope-prediction/alternative_train_v4.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.pos)
alter.train.dominant.pos <- dplyr::sample_n(train.dominant.pos, 1703) # 0.04
alter.train.dominant <- rbind(alter.train.dominant.pos, train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/Desktop/GS/TCR-epitope-prediction/alternative_train_v5.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.pos)
alter.train.dominant.pos <- dplyr::sample_n(train.dominant.pos, 2128) # 0.05
alter.train.dominant <- rbind(alter.train.dominant.pos, train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/Desktop/GS/TCR-epitope-prediction/alternative_train_v6.csv', quote = F, sep = ',', row.names = F, append = F)

train.dominant <- train[train$epitope %in% dominant,]
train.dominant.pos <- train.dominant[train.dominant$binder == 1,]
train.dominant.neg <- train.dominant[train.dominant$binder == 0,]
dim(train.dominant.neg)
alter.train.dominant.neg <- dplyr::sample_n(train.dominant.neg, 37765) # 0.06
alter.train.dominant <- rbind(train.dominant.pos, alter.train.dominant.neg)
dim(alter.train.dominant)
alter.train <- rbind(alter.train.dominant, train.ndominant)
dim(alter.train)
write.table(alter.train, '/Users/vy/Desktop/GS/TCR-epitope-prediction/alternative_train_v7.csv', quote = F, sep = ',', row.names = F, append = F)

#--- intepreting prediction result from different trainings
dominant <- c('NLVPMVATV', 'KLGGALQAK', 'GILGFVFTL', 'TPRVTGGGAM', 'GLCTLVAML', 'AVFDRKSDAK',
              'ELAGIGILTV')
files <- list.files('/Users/vy/Desktop/GS/TCR-epitope-prediction/predict_proba_withoutMHC', pattern = '^predict_proba', full.names = T)
test01 <- fread('/Users/vy/Desktop/GS/TCR-epitope-prediction/predict_proba_withoutMHC/test01_predict_proba.csv', data.table = F)
p_thres <- 0.5

perf <- NULL
for (file in files) {
  version <- gsub('.*predict_proba_', '', file)
  version <- gsub('.csv', '', version)
  pred <- fread(file, data.table = F)
  test01 <- cbind(test01, pred$predict_proba)
  pred <- cbind(test01[, c(1:3)], pred$predict_proba)
  colnames(test01)[ncol(test01)] <- paste0('predict_proba', '_', version)
  colnames(pred)[4] <- 'predict_proba'

  overall.perf <- calculate_metrics(pred$predict_proba, pred$binder, 0.5)

  alter.intrain <- pred[pred$epitope %in% train$epitope,]
  alter.intrain.perf <- calculate_metrics(alter.intrain$predict_proba, alter.intrain$binder, 0.5)

  alter.intrain.ndominant <- alter.intrain[!alter.intrain$epitope %in% dominant,]
  intrain.ndominant.perf <- calculate_metrics(alter.intrain.ndominant$predict_proba, alter.intrain.ndominant$binder, 0.5)

  alter.intrain.dominant <- alter.intrain[alter.intrain$epitope %in% dominant,]
  intrain.dominant.perf <- calculate_metrics(alter.intrain.dominant$predict_proba, alter.intrain.dominant$binder, 0.5)

  alter.ndominant <- pred[!pred$epitope %in% dominant,]
  ndominant.perf <- calculate_metrics(alter.ndominant$predict_proba, alter.ndominant$binder, 0.5)

  if (is.null(perf))
    perf <- cbind(version, 'overall', overall.perf$auc, overall.perf$acc,
                  overall.perf$sens, overall.perf$spec)
  else
    perf <- rbind(perf, cbind(version, 'overall', overall.perf$auc, overall.perf$acc,
                              overall.perf$sens, overall.perf$spec))

  perf <- rbind(perf, cbind(version, 'intrain', alter.intrain.perf$auc,
                            alter.intrain.perf$acc, alter.intrain.perf$sens,
                            alter.intrain.perf$spec))

  perf <- rbind(perf, cbind(version, 'intrain-non-dominant', intrain.ndominant.perf$auc,
                            intrain.ndominant.perf$acc, intrain.ndominant.perf$sens,
                            intrain.ndominant.perf$spec))

  perf <- rbind(perf, cbind(version, 'intrain-dominant', intrain.dominant.perf$auc,
                            intrain.dominant.perf$acc, intrain.dominant.perf$sens,
                            intrain.dominant.perf$spec))

  perf <- rbind(perf, cbind(version, 'non-dominant', ndominant.perf$auc,
                            ndominant.perf$acc, ndominant.perf$sens,
                            ndominant.perf$spec))
}

colnames(perf) <- c('version', 'group', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity')
perf <- as.data.frame(perf)
perf$dominant.proportion.train <- plyr::mapvalues(perf$version, from = unique(perf$version),
                                                  to = c(2266/44834, NaN, 690/44834, 0, 1277/44834,
                                                         1703/44834, 2128/44834, 2266/37765))
perf <- melt(perf, id.vars = c('version', 'group', 'dominant.proportion.train'))
perf$value <- as.numeric(perf$value)
perf$dominant.proportion.train <- round(as.numeric(perf$dominant.proportion.train), 3)
perf <- perf[order(perf$dominant.proportion.train, decreasing = F),]
perf$version <- factor(perf$version, levels = unique(perf$version))

perf$group[perf$group == 'overall'] <- 'Overall'
perf$group[perf$group == 'intrain-dominant'] <- 'Dominant peptides'
perf$group[perf$group == 'non-dominant'] <- 'Other peptides'
perf$group <- factor(perf$group, levels = unique(perf$group))

library(pROC)
roc.obj1 <- roc(binder ~ predict_proba_original + predict_proba_v2 +
                  predict_proba_v4 + predict_proba_v5 +
                  predict_proba_v6 + predict_proba_v7, data = test01)
ggroc(roc.obj1, legacy.axes = TRUE) + xlab("FPR") + ylab("TPR") +
  scale_color_manual(values = c('black', 'darkviolet', 'blue', 'darkgreen',
                                'orange', 'red'),
                     labels = unique(perf$dominant.proportion.train[perf$dominant.proportion.train > 0])) +
  cowplot::theme_cowplot() + guides(color = guide_legend(title ="Try"))

test01.dominant <- test01[test01$epitope %in% dominant,]
roc.obj1.dominant <- roc(binder ~ predict_proba_original + predict_proba_v2 +
                  predict_proba_v4 + predict_proba_v5 +
                  predict_proba_v6 + predict_proba_v7, data = test01.dominant)
perf.dominant.spec <- perf[perf$group == 'Dominant peptides' & perf$variable == 'Specificity'
                           & !is.nan(perf$dominant.proportion.train)
                           & perf$dominant.proportion.train > 0,]
fpr <- 1 - perf.dominant.spec$value

h_segment_data = data.frame(
  x = c(rep(0, 6)), y = unique(perf[perf$group == 'Dominant peptides'
                                    & perf$variable == 'Sensitivity'
                                    & !is.nan(perf$dominant.proportion.train)
                                    & perf$dominant.proportion.train > 0, 'value']),
  xend = fpr, yend = unique(perf[perf$group == 'Dominant peptides'
                                 & perf$variable == 'Sensitivity'
                                 & !is.nan(perf$dominant.proportion.train)
                                 & perf$dominant.proportion.train > 0, 'value'])
)

v_segment_data = data.frame(
  x = fpr, y = 0,
  xend = fpr, yend = unique(perf[perf$group == 'Dominant peptides'
                                 & perf$variable == 'Sensitivity'
                                 & !is.nan(perf$dominant.proportion.train)
                                 & perf$dominant.proportion.train > 0, 'value'])
)

ggroc(roc.obj1.dominant, legacy.axes = TRUE) + xlab("1 - Specificity") + ylab("Sensitivity") +
  scale_color_manual(values = c('black', 'darkviolet', 'blue', 'darkgreen',
                                'orange', 'red'),
                     labels = unique(perf$dominant.proportion.train[perf$dominant.proportion.train > 0])) +
  geom_segment(data = h_segment_data, aes(x = x, y = y, xend = xend, yend = yend),
             color = c('black', 'darkviolet', 'blue', 'darkgreen',
                       'orange', 'red'), linetype = 'longdash') +
  geom_segment(data = v_segment_data, aes(x = x, y = y, xend = xend, yend = yend),
               color = c('black', 'darkviolet', 'blue', 'darkgreen',
                         'orange', 'red'), linetype = 'longdash') +
  cowplot::theme_cowplot() +
  guides(color = guide_legend(title ="Positive/negative\nproportion in\ntraining set"))

# AUC
ggplot(perf[perf$variable == 'AUC' & perf$group %in% c('Overall', 'Dominant peptides', 'Other peptides') & perf$dominant.proportion.train != 'NaN' & perf$dominant.proportion.train > 0,],
       aes(x = version, y = value, fill = group)) +
  geom_bar(aes(x = version, y = value, fill = group), stat = 'identity', position = position_dodge(width = 0.9)) +
  list(geom_step(aes(x = version, y = dominant.proportion.train * y.ratio), size = 1.5,
                 color = 'black', group = 1, position = position_nudge(x = -0.5)),
       # add one more line on the right
       geom_segment(mapping = aes(x = 5.49, y = dominant.proportion.train[17] * y.ratio, xend = 6.49,
                                  yend = dominant.proportion.train[17] * y.ratio), color = 'black', size = 1.5)) +
  cowplot::theme_cowplot() + xlab('Trainings corresponding to proportions') + ylab('AUC') +
  theme(axis.text.x=element_blank(),
        axis.ticks.x= element_blank()) +
  scale_y_continuous(sec.axis = sec_axis( trans=~./ y.ratio, name="Positive/negative proportion of 7 dominant peptides in training set")) +
  guides(fill = guide_legend(title = 'Groups'))

# Acc, sens, spec
ggplot(perf[perf$variable %in% c('Accuracy', 'Sensitivity', 'Specificity') & perf$group %in% c('Overall', 'Dominant peptides', 'Other peptides')
            & perf$dominant.proportion.train != 'NaN' & perf$dominant.proportion.train > 0,],
       aes(x = factor(dominant.proportion.train), y = value, color = variable, group = new_group)) +
  geom_line(aes(linetype = group)) +
  scale_linetype_manual(values = c('solid', 'dashed', 'dotted')) +
  geom_point() +
  #theme(legend.position = c(.9, .75)) +
  cowplot::theme_cowplot() + xlab('Positive/negative proportion of 7 dominant peptides in training set') + ylab('Metrics') +
  guides(color = guide_legend(title = 'Metrics'), linetype = guide_legend(title = 'Groups')) +
  theme(legend.position = c(.65, .55), legend.box.background = element_rect(colour = "black"),
        legend.margin=margin(5,5,5,5), legend.title=element_text(size=12),
        legend.key.size = unit(16, 'pt'), legend.text = element_text(size=11),
        legend.spacing.y=unit(0.1,"cm"), axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12))
