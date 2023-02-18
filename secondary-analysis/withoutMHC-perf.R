library(data.table)
library(dplyr)
library(ggplot2)
library(viridis)

round2 = function(x, digits) {
  posneg = sign(x)
  z = abs(x)*10^digits
  z = z + 0.5 + sqrt(.Machine$double.eps)
  z = trunc(z)
  z = z/10^digits
  z*posneg
}

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

# label shuffling
files <- list.files('/Users/vy/epiTCR/data/predepiTCRData/withoutMHC', full.names = T)
test01 <- fread(files[1], data.table = F)
dim(test01)
test01$shuffled_binder <- sample(test01$binder)
calculate_metrics(test01$predict_proba, test01$shuffled_binder, 0.5)

roc.obj1 <- roc(binder ~ predict_proba, data = test01)
roc.obj2 <- roc(shuffled_binder ~ predict_proba, data = test01)
ggroc(list(roc.obj1, roc.obj2), legacy.axes = TRUE) + xlab("FPR") + ylab("TPR") +
  scale_color_manual(values = c('red', 'black'),
                     labels = c('True label', 'Shuffled label')) +
  cowplot::theme_cowplot() + guides(color = guide_legend(title ="Prediction on:"))

# #calculate epiTCR perf, no need to rerun
# files <- list.files('/Users/vy/epiTCR/data/predepiTCRData/withoutMHC', full.names = T)
# files
#
# perf <- NULL
# for (file in files) {
#   dat <- fread(file, data.table = F)
#   test.no <- gsub('.*test', '', file)
#   test.no <- gsub('_predict_proba.csv', '', test.no)
#
#   perf.test <- calculate_metrics(dat$predict_proba, dat$binder, 0.5)
#   if (is.null(perf)) perf <- c(perf.test$acc, perf.test$sens, perf.test$spec, perf.test$auc, test.no)
#   else perf <- rbind(perf, c(perf.test$acc, perf.test$sens, perf.test$spec, perf.test$auc, test.no))
# }
# colnames(perf) <- c('acc', 'sens', 'spec', 'auc', 'test')
# write.table(perf,
#             '/Users/vy/epiTCR/data/predToolsData/performance_output/epiTCR_output_performance_without_mhc.csv',
#             quote = F, row.names = F, sep = ',')

# without MHC
files <- list.files('/Users/vy/epiTCR/data/predToolsData/performance_output', full.names = T)
files <- files[!grepl('with_|pMTnet|mlibtcr', files)]

all = NULL
for (file in files) {
  dat <- fread(file, data.table = F)
  dat$test <- rownames(dat)
  file.short <- gsub('.*/', '', file)
  dat$Tools <- ifelse(grepl('train', file.short), '-retrained', '')
  dat$Tools <- paste0(gsub('_.*', '', file.short), dat$Tools)
  if (is.null(all)) all <- dat
  else all <- rbind(all, dat)
}

all$Tools <- plyr::mapvalues(all$Tools, from = unique(all$Tools), to = c('ATM-TCR', 'ATM-TCR*', 'epiTCR', 'Imrex','NetTCR', 'NetTCR*'))
all$Tools <- factor(all$Tools, levels = rev(c('epiTCR', 'NetTCR', 'NetTCR*', 'Imrex', 'ATM-TCR', 'ATM-TCR*')))
all$test <- as.numeric(all$test)
all$auc <- round2(all$auc, 2)
all$acc <- round2(all$acc, 2)
all$sens <- round2(all$sens, 2)
all$spec <- round2(all$spec, 2)

all.long <- melt(all, id.vars = c('Tools', 'test'))
all.long$variable <- factor(all.long$variable, levels = c('auc', 'acc', 'sens', 'spec'))
levels(all.long$variable) <- c('AUC', 'Accuracy', 'Sensitivity', 'Specificity')
ggplot(all.long, aes(x = test, y = Tools, fill = value)) +
  geom_tile() +
  scale_fill_gradientn(colors = c('burlywood1', 'orange', 'orangered2', 'firebrick4'),
                       values = scales::rescale(sort(all.long$value))) +
  geom_text(aes(test, Tools, label = value), colour = "black", size = 3, check_overlap = TRUE) +
  facet_wrap(~ variable, ncol = 1) +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        panel.background = element_rect(fill = "white"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
  ) + xlab('') + ylab('') + guides(fill = "none") + scale_x_continuous(expand = c(0,0))
