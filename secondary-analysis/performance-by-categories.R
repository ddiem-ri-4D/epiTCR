library(data.table)
library(ggplot2)

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

dominant <- c('NLVPMVATV', 'KLGGALQAK', 'GILGFVFTL', 'TPRVTGGGAM', 'GLCTLVAML', 'AVFDRKSDAK', 'ELAGIGILTV')

binding.train <- fread('/Users/vy/epiTCR/data/categories/full-training-with-categories.csv', data.table = F)
train.dominant <- binding.train[binding.train$epitope %in% dominant,]

binding.pred <- fread('/Users/vy/epiTCR/data/categories/full-prediction-with-categories.csv', data.table = F)
virus.perf <- calculate_metrics(binding.pred$predict_proba[binding.pred$Pathogen_source == 'virus'],
                                binding.pred$binder[binding.pred$Pathogen_source == 'virus'], 0.5)

human.perf <- calculate_metrics(binding.pred$predict_proba[binding.pred$Pathogen_source == 'human'],
                                binding.pred$binder[binding.pred$Pathogen_source == 'human'], 0.5)

bacteria.perf <- calculate_metrics(binding.pred$predict_proba[binding.pred$Pathogen_source == 'bacteria'],
                                   binding.pred$binder[binding.pred$Pathogen_source == 'bacteria'], 0.5)

perf.cat <- data.frame(category = c(rep('Virus', 4), rep('Human', 4), rep('Bacteria', 4)),
                       metrics = c('AUC', 'Accuracy', 'Sensitivity', 'Specificity',
                                   'AUC', 'Accuracy', 'Sensitivity', 'Specificity',
                                   'AUC', 'Accuracy', 'Sensitivity', 'Specificity'),
                       values = c(virus.perf$auc, virus.perf$acc, virus.perf$sens, virus.perf$spec,
                                  human.perf$auc, human.perf$acc, human.perf$sens, human.perf$spec,
                                  bacteria.perf$auc, bacteria.perf$acc, bacteria.perf$sens, bacteria.perf$spec))
perf.cat$metrics <- factor(perf.cat$metrics, levels = c('AUC', 'Accuracy', 'Sensitivity', 'Specificity'))
ggplot(perf.cat[perf.cat$category != 'Bacteria',], aes(x = category, y = values, fill = category)) +
  geom_bar(stat = 'identity') + facet_wrap(~ metrics) +
  cowplot::theme_cowplot() + theme(axis.text.x=element_blank(),
                                   axis.ticks.x=element_blank()) + xlab('') + ylab('Metrics') +
  guides(fill=guide_legend(title="Categories")) +
  geom_text(aes(label = round(values, 3)), stat = "identity", vjust = 1.5, hjust = 0.5, size = 4, colour = "white", angle = 0)
ld <- layer_data(last_plot())

binding.pred$major_pathogen <- binding.pred$Pathogen_source
binding.pred$major_pathogen[!binding.pred$major_pathogen %in% c('virus', 'human')] <- 'unknown/other'
binding.train$major_pathogen <- binding.train$Pathogen_source
binding.train$major_pathogen[!binding.train$major_pathogen %in% c('virus', 'human')] <- 'unknown/other'

data.summary <- rbind(table(binding.train$binder, binding.train$major_pathogen) %>%
                        data.frame() %>% mutate(dataset = 'training') %>% group_by(Var1) %>%
                        mutate(sum = sum(Freq)) %>% mutate(perc = Freq/sum),
                      table(binding.pred$binder, binding.pred$major_pathogen) %>%
                        data.frame() %>% mutate(dataset = 'test') %>% group_by(Var1) %>%
                        mutate(sum = sum(Freq)) %>% mutate(perc = Freq/sum))
data.summary$Var1 <- plyr::mapvalues(data.summary$Var1, from = unique(data.summary$Var1), to = c('Non-binding', 'Binding'))
data.summary$Var2 <- factor(data.summary$Var2, levels = c('unknown/other', 'human', 'virus'))
data.summary$dataset <- factor(data.summary$dataset, levels = c('training', 'test'))
ggplot(data.summary, aes(x = Var1, y = Freq, fill = Var2)) +
  geom_bar(position= 'fill', stat = 'identity') +
  facet_wrap(~ dataset) + scale_fill_manual(values = c('gray', '#F8766D', '#00BFC4'),
                                            labels = c('Unknown/other', 'Human', 'Virus')) +
  cowplot::theme_cowplot() + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
                                   #legend.position = 'bottom', legend.direction = 'vertical'
  ) +
  xlab('') + ylab('Proportion') +
  guides(fill=guide_legend(title="Categories"))

data.summary <- rbind(table(binding.train$binder, binding.train$database_division) %>%
                        data.frame() %>% mutate(dataset = 'training') %>%
                        group_by(Var1) %>% mutate(sum = sum(Freq)) %>%
                        mutate(perc = Freq/sum),
                      table(binding.pred$binder, binding.pred$database_division) %>%
                        data.frame() %>% mutate(dataset = 'test') %>%
                        group_by(Var1) %>% mutate(sum = sum(Freq)) %>%
                        mutate(perc = Freq/sum))
data.summary$Var1 <- plyr::mapvalues(data.summary$Var1, from = unique(data.summary$Var1), to = c('Non-binding', 'Binding'))
data.summary$dataset <- factor(data.summary$dataset, levels = c('training', 'test'))
ggplot(data.summary[data.summary$perc > 0,], aes(x = Var1, y = Freq, fill = Var2)) +
  geom_bar(position= 'fill', stat = 'identity') +
  facet_wrap(~ dataset) +
  cowplot::theme_cowplot() + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
                                   legend.position = 'bottom', legend.direction = 'vertical') +
  geom_text(aes(label = round(perc, 2)), position = position_fill(vjust = 0.5)) +
  xlab('') + ylab('Proportion') +
  guides(fill=guide_legend(title="Categories"))

data.summary <- rbind(table(train.dominant$binder, train.dominant$database_division) %>%
                        data.frame() %>% mutate(dataset = 'training') %>%
                        group_by(Var1) %>% mutate(sum = sum(Freq)) %>%
                        mutate(perc = Freq/sum),
                      table(binding.pred[binding.pred$epitope %in% dominant,]$binder, binding.pred[binding.pred$epitope %in% dominant,]$database_division) %>%
                        data.frame() %>% mutate(dataset = 'test') %>%
                        group_by(Var1) %>% mutate(sum = sum(Freq)) %>%
                        mutate(perc = Freq/sum))
data.summary$Var1 <- plyr::mapvalues(data.summary$Var1, from = unique(data.summary$Var1), to = c('Non-binding', 'Binding'))
data.summary$dataset <- factor(data.summary$dataset, levels = c('training', 'test'))
ggplot(data.summary[data.summary$perc > 0,], aes(x = Var1, y = perc, fill = Var2)) +
  geom_bar(position= 'fill', stat = 'identity') +
  facet_wrap(~ dataset) +
  cowplot::theme_cowplot() + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
                                   legend.position = 'bottom', legend.direction = 'vertical') +
  geom_text(aes(label = round(perc, 2)), position = position_fill(vjust = 0.5)) +
  scale_fill_discrete(labels = c('dominant antigen-specific association database', 'dominant antigen-specific validated database')) +
  xlab('') + ylab('Proportion') +
  guides(fill=guide_legend(title="Categories"))
ld <- layer_data(last_plot())

pred.validated <- binding.pred[grepl('validated', binding.pred$database_division),]
pred.assoc <- binding.pred[!grepl('validated', binding.pred$database_division),]
perf.validated <- calculate_metrics(pred.validated$predict_proba, pred.validated$binder, 0.5)
perf.assoc <- calculate_metrics(pred.assoc$predict_proba, pred.assoc$binder, 0.5)
perf.all <- calculate_metrics(binding.pred$predict_proba, binding.pred$binder, 0.5)

perf.cat <- data.frame(category = c(rep('all data', 4),
                                    rep('antigen-specific validated database', 4),
                                    rep('antigen-specific association database', 4)),
                       performance = c('AUC', 'Accuracy', 'Sensitivity', 'Specificity',
                                       'AUC', 'Accuracy', 'Sensitivity', 'Specificity',
                                       'AUC', 'Accuracy', 'Sensitivity', 'Specificity'),
                       values = c(perf.all$auc, perf.all$acc, perf.all$sens, perf.all$spec,
                                  perf.validated$auc, perf.validated$acc, perf.validated$sens, perf.validated$spec,
                                  perf.assoc$auc, perf.assoc$acc, perf.assoc$sens, perf.assoc$spec))
perf.cat$performance <- factor(perf.cat$performance, levels = c('AUC', 'Accuracy', 'Sensitivity', 'Specificity'))
perf.cat$category <- factor(perf.cat$category,
                            levels = c('all data', 'antigen-specific validated database', 'antigen-specific association database'))
ggplot(perf.cat, aes(x = category, y = values, fill = category)) +
  geom_bar(stat = 'identity') + facet_wrap(~ performance) +
  cowplot::theme_cowplot() + theme(axis.text.x=element_blank(),
                                   axis.ticks.x=element_blank()) + xlab('') + ylab('Performance') +
  guides(fill=guide_legend(title="Groups")) +
  geom_text(aes(label = round(values, 3)), stat = "identity", vjust = 1.5, hjust = 0.5, size = 4, colour = "white") +
  scale_fill_manual(values = c('gray', '#00BFC4', '#F8766D')) +
  theme(axis.text.x = element_blank(),
        legend.position = 'bottom', legend.direction = 'vertical')
