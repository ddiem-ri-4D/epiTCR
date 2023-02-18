library(gplots)
library(data.table)
library(dplyr)
library(factoextra)

dat <- fread('/Users/vy/Downloads/matrix_leven_simi_epi.csv', data.table = F)
rownames(dat) <- dat$epitope
dat$epitope <- NULL

pred <- fread('/Users/vy/epiTCR/data/nonOverlapingPeptide/withoutMHC/test/full_pred.csv', data.table = F)
train.obs <- fread('/Users/vy/epiTCR/data/nonOverlapingPeptide/withoutMHC/train/train.csv')
any(train.obs$epitope %in% pred$epitope)

dat.test <- dat[unique(pred$epitope), unique(pred$epitope)]
dat.train <- dat[unique(train.obs$epitope), unique(train.obs$epitope)]

row.dist <- dist(as.matrix(dat.test))
row.hc <- hclust(row.dist, method = 'ward.D2')
row.dend <- as.dendrogram(row.hc)

clusters <- cutree(row.hc, 6)
prediction <- table(pred$epitope, pred$binder_pred) %>% as.data.frame.matrix()
colnames(prediction) <- c('neg_pred', 'pos_pred')

plot.dat <- NULL
for (i in c(1:6)) {
  print(i)
  set.seed(333)
  clust.peps <- names(clusters[clusters == i])
  train.test.simi <- data.frame(t(dat[clust.peps, colnames(dat.train)]))

  train.test.simi <- train.test.simi %/% 10
  train.test.simi <- train.test.simi[, unlist(sapply(train.test.simi,
                                                     function(x) any(x == 5) & any(x == 6) & any(x == 7) &
                                                       any(x == 8) & any(x == 9))), drop = F]
  if (ncol(train.test.simi) == 0)
    next
  clust.pep <- sample(colnames(train.test.simi), size = 1)
  train.test.simi <- train.test.simi[, clust.pep, drop = F]
  train.test.simi$bin <- train.test.simi[, 1]
  train.obs$bin <- plyr::mapvalues(train.obs$epitope, from = rownames(train.test.simi), to = train.test.simi$bin)
  train.calc <- train.obs %>% group_by(bin) %>% mutate(total_obs_bin = n()) %>% group_by(bin, cat) %>%
    mutate(total_obs_bin_cat = n(), perc_train = total_obs_bin_cat/total_obs_bin) %>% select(bin, cat, perc_train) %>% unique()
  train.calc.template <- data.frame(cbind(bin = rep(unique(train.calc$bin), each = 3),
                                          cat = rep(c('mix', 'pos', 'neg'), length(unique(train.calc$bin))),
                                          perc_train = 0))
  train.calc <- rquery::natural_join(train.calc, train.calc.template, by = c('bin', 'cat'), jointype = 'FULL')
  train.calc$cat <- factor(train.calc$cat, levels = c('mix', 'pos', 'neg'))

  prediction.pep <- prediction[clust.pep,]
  prediction.pep <- cbind(prediction.pep,
                          perc_test = c(0,
                                        prediction.pep$pos_pred/(prediction.pep$neg_pred + prediction.pep$pos_pred),
                                        prediction.pep$neg_pred/(prediction.pep$neg_pred + prediction.pep$pos_pred)))
  prediction.pep <- cbind(prediction.pep, cat = c('mix', 'pos', 'neg'))
  prediction.pep <- prediction.pep[order(prediction.pep$cat),]

  calc <- cbind(train.calc, perc_test = rep(prediction.pep$perc_test, length(unique(train.calc$bin))))
  calc$perc_train <- as.numeric(calc$perc_train)
  calc$perc_test <- as.numeric(calc$perc_test)
  plot.dat.i <- calc %>% group_by(bin) %>% mutate(rmse = sqrt(mean((perc_train - perc_test)^2)))

  if (is.null(plot.dat.i)) plot.dat <- cbind(clust = i, plot.dat.i)
  else plot.dat <- rbind(plot.dat, cbind(clust = i, plot.dat.i))
}
plot.dat$clust <- factor(plot.dat$clust)
plot.dat$bin <- as.numeric(plot.dat$bin)
plot.dat <- plot.dat[plot.dat$bin >= 5,]
plot.dat$bin <- plyr::mapvalues(plot.dat$bin, from = unique(plot.dat$bin), to = c('50-59', '60-69', '70-79', '80-89', '90-99'))
ggplot(plot.dat, aes(x = bin, y = rmse, color = clust)) +
  geom_point() +
  geom_line(aes(group = clust)) +
  facet_wrap(vars(clust), nrow = 2) +
  cowplot::theme_cowplot() +
  guides(color=guide_legend(title="Peptides")) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  xlab('Similarity bins (%)') + ylab('Root Mean Square Error (RMSE)')

# mean AUC
# average perf run by epiTCR, NetTCR, and ATM-TCR on 10 times subsampling full test set
plot.dat <- data.frame(Tools = c('epiTCR', 'NetTCR', 'ATM-TCR'), AUC = c(0.75, 0.75, 0.31))
ggplot(plot.dat, aes(x = Tools, y = AUC, color = Tools, fill = Tools)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(label = AUC), stat = "identity", vjust = 2, hjust = 0.5, size = 5, colour = "white", angle = 0) +
  cowplot::theme_cowplot() + theme(axis.ticks.x=element_blank()) + xlab('') +
  theme(legend.position = "none")
