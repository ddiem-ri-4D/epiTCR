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

dominant <- c('NLVPMVATV', 'KLGGALQAK', 'GILGFVFTL', 'TPRVTGGGAM', 'GLCTLVAML', 'AVFDRKSDAK',
              'ELAGIGILTV')
files <- list.files('/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-training-pred/epiTCR/', pattern = 'alternative-training-pred-test01', full.names = T)
train <- fread('/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/train.csv', data.table =  F)
test01 <- fread('/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-trainings/test01.csv', data.table = F)
p_thres <- 0.5

perf <- NULL
for (file in files) {
  version <- gsub('.*alternative-training-pred-test01-', '', file)
  version <- gsub('.csv', '', version)
  pred <- fread(file, data.table = F)
  test01 <- merge(test01, pred[, 1:4], by = c('CDR3b', 'epitope', 'binder'))
  colnames(test01)[ncol(test01)] <- paste0('predict_proba', '_', version)

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
                                                  to = c(2266/44834, NaN, 690/44834,
                                                         0, 1277/44834,
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
roc.obj1 <- roc(binder ~ predict_proba_v2 + predict_proba_v4 +
                  predict_proba_v5 + predict_proba_v6 +
                  predict_proba_original + predict_proba_v7, data = test01)
ggroc(roc.obj1, legacy.axes = TRUE) + xlab("FPR") + ylab("TPR") +
  scale_color_manual(values = c('black', 'darkviolet', 'blue', 'darkgreen',
                                       'orange', 'red'),
                                       labels = unique(perf$dominant.proportion.train[perf$dominant.proportion.train > 0])) +
  cowplot::theme_cowplot() + guides(color = guide_legend(title ="Try"))

test01.dominant <- test01[test01$epitope %in% dominant,]
roc.obj1.dominant <- roc(binder ~ predict_proba_v2 + predict_proba_v4 +
                           predict_proba_v5 + predict_proba_v6 +
                           predict_proba_original + predict_proba_v7, data = test01.dominant)
ggroc(roc.obj1.dominant, legacy.axes = TRUE) + xlab("FPR") + ylab("TPR") +
  scale_color_manual(values = c('black', 'darkviolet', 'blue', 'darkgreen',
                                       'orange', 'red'),
                                       labels = unique(perf$dominant.proportion.train[perf$dominant.proportion.train > 0])) +
  cowplot::theme_cowplot() + guides(color = guide_legend(title ="Try"))

perf.dominant.spec <- perf[perf$group == 'Dominant peptides' & perf$variable == 'Specificity'
                           & !is.nan(perf$dominant.proportion.train)
                           & perf$dominant.proportion.train > 0,]
fpr <- 1 - perf.dominant.spec$value

h_segment_data = data.frame(
  x = c(rep(0, 6)), y = perf[perf$group == 'Dominant peptides'
                             & perf$variable == 'Sensitivity'
                             & !is.nan(perf$dominant.proportion.train)
                             & perf$dominant.proportion.train > 0, 'value'],
  xend = fpr, yend = perf[perf$group == 'Dominant peptides'
                          & perf$variable == 'Sensitivity'
                          & !is.nan(perf$dominant.proportion.train)
                          & perf$dominant.proportion.train > 0, 'value']
)

v_segment_data = data.frame(
  x = fpr, y = 0,
  xend = fpr, yend = perf[perf$group == 'Dominant peptides'
                          & perf$variable == 'Sensitivity'
                          & !is.nan(perf$dominant.proportion.train)
                          & perf$dominant.proportion.train > 0, 'value']
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

# AUC - full tries
y.ratio = 10

# selected-tries
perf$new_group <- paste0(perf$group, '_', perf$variable)

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

# AUC selected-tries
ggplot(perf[perf$variable == 'AUC' & perf$group %in% c('Overall', 'Dominant peptides', 'Other peptides') & perf$dominant.proportion.train != 'NaN' & perf$dominant.proportion.train > 0,],
       aes(x = version, y = value, fill = group)) +
  geom_bar(aes(x = version, y = value, fill = group), stat = 'identity', position = position_dodge(width = 0.9)) +
  list(geom_step(aes(x = version, y = dominant.proportion.train * y.ratio), size = 1.5,
                 color = 'black', group = 1, position = position_nudge(x = -0.5)),
       # add one more line on the right
       geom_segment(mapping = aes(x = 5.49, y = dominant.proportion.train[17] * y.ratio, xend = 6.49,
                                  yend = dominant.proportion.train[17] * y.ratio), color = 'black', size = 1.5)) +
  #geom_text(aes(label = round(value, 2)), stat = "identity", vjust = 1.5, size = 3, colour = "white") +
  cowplot::theme_cowplot() + xlab('Trainings corresponding to proportions') + ylab('AUC') +
  theme(axis.text.x=element_blank(),
        axis.ticks.x= element_blank()) +
  scale_y_continuous(sec.axis = sec_axis( trans=~./ y.ratio, name="Positive/negative proportion of 7 dominant peptides in training set")) +
  guides(fill = guide_legend(title = 'Groups'))

#---
test01 <- fread('/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-training-pred/epiTCR/alternative-training-pred-test01-original.csv', data.table = F)
files <- list.files('/Users/vy/epiTCR/data/pred7DominantPeptide/alternative-training-pred/NetTCR', full.names = T)

perf <- NULL
for (file in files) {
  version <- gsub('.*predict_itest01_', '', file)
  version <- gsub('.csv', '', version)
  pred <- fread(file, data.table = F)
  if (version == 'original') {
    colnames(pred)[3] <- 'predict_proba'
    pred <- pred[, -4]
  } else colnames(pred)[4] <- 'predict_proba'
  test01 <- merge(test01, pred, by.x = c('CDR3b', 'epitope', 'binder'), by.y = c('CDR3b', 'peptide', 'binder'))
  colnames(test01)[ncol(test01)] <- paste0('predict_proba', '_', version)

  overall.perf <- calculate_metrics(pred$predict_proba, pred$binder, 0.5)

  alter.intrain <- pred[pred$peptide %in% train$epitope,]
  alter.intrain.perf <- calculate_metrics(alter.intrain$predict_proba, alter.intrain$binder, 0.5)

  alter.intrain.ndominant <- alter.intrain[!alter.intrain$peptide %in% dominant,]
  intrain.ndominant.perf <- calculate_metrics(alter.intrain.ndominant$predict_proba, alter.intrain.ndominant$binder, 0.5)

  alter.intrain.dominant <- alter.intrain[alter.intrain$peptide %in% dominant,]
  intrain.dominant.perf <- calculate_metrics(alter.intrain.dominant$predict_proba, alter.intrain.dominant$binder, 0.5)

  alter.ndominant <- pred[!pred$peptide %in% dominant,]
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
                                                  to = c(2266/44834, NaN, 690/44834,
                                                         0, 1277/44834,
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
roc.obj1 <- roc(binder ~ predict_proba_v2 + predict_proba_v4 +
                  predict_proba_v5 + predict_proba_v6 +
                  predict_proba_original + predict_proba_v7, data = test01)
ggroc(roc.obj1, legacy.axes = TRUE) + xlab("FPR") + ylab("TPR") +
  scale_color_manual(values = c('black', 'darkviolet', 'blue', 'darkgreen',
                                       'orange', 'red'),
                                       labels = unique(perf$dominant.proportion.train[perf$dominant.proportion.train > 0])) +
  cowplot::theme_cowplot() + guides(color = guide_legend(title ="Try"))

y.ratio = 10

# selected-tries
perf$new_group <- paste0(perf$group, '_', perf$variable)

ggplot(perf[perf$variable %in% c('Accuracy', 'Sensitivity', 'Specificity') & perf$group %in% c('Overall', 'Dominant peptides', 'Other peptides')
            & perf$dominant.proportion.train != 'NaN' & perf$dominant.proportion.train > 0,],
       aes(x = factor(dominant.proportion.train), y = value, color = variable, group = new_group)) +
  geom_line(aes(linetype = group)) +
  scale_linetype_manual(values = c('solid', 'dashed', 'dotted')) +
  geom_point() +
  #theme(legend.position = c(.9, .75)) +
  cowplot::theme_cowplot() + xlab('Positive/negative proportion of 7 dominant peptides in training set') + ylab('Metrics') +
  guides(color = guide_legend(title = 'Metrics'), linetype = guide_legend(title = 'Groups')) +
  theme(legend.position = c(.55, .35), legend.box.background = element_rect(colour = "black"),
        legend.margin=margin(5,5,5,5), legend.title=element_text(size=12),
        legend.key.size = unit(16, 'pt'), legend.text = element_text(size=11),
        legend.spacing.y=unit(0.1,"cm"), axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12))

# AUC selected-tries
ggplot(perf[perf$variable == 'AUC' & perf$group %in% c('Overall', 'Dominant peptides', 'Other peptides') & perf$dominant.proportion.train != 'NaN' & perf$dominant.proportion.train > 0,],
       aes(x = version, y = value, fill = group)) +
  geom_bar(aes(x = version, y = value, fill = group), stat = 'identity', position = position_dodge(width = 0.9)) +
  list(geom_step(aes(x = version, y = dominant.proportion.train * y.ratio), size = 1.5,
                 color = 'black', group = 1, position = position_nudge(x = -0.5)),
       # add one more line on the right
       geom_segment(mapping = aes(x = 5.49, y = dominant.proportion.train[17] * y.ratio, xend = 6.49,
                                  yend = dominant.proportion.train[17] * y.ratio), color = 'black', size = 1.5)) +
  #geom_text(aes(label = round(value, 2)), stat = "identity", vjust = 1.5, size = 3, colour = "white") +
  cowplot::theme_cowplot() + xlab('Trainings corresponding to proportions') + ylab('AUC') +
  theme(axis.text.x=element_blank(),
        axis.ticks.x= element_blank()) +
  scale_y_continuous(sec.axis = sec_axis( trans=~./ y.ratio, name="Positive/negative proportion of 7 dominant peptides in training set")) +
  guides(fill = guide_legend(title = 'Groups'))

