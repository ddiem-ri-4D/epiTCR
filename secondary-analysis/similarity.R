library(data.table)
library(ggplot2)
library(rquery)

round2 = function(x, digits) {
  posneg = sign(x)
  z = abs(x)*10^digits
  z = z + 0.5 + sqrt(.Machine$double.eps)
  z = trunc(z)
  z = z/10^digits
  z*posneg
}
training <- fread('/Users/vy/Desktop/GS/TCR-epitope-prediction/train.csv', data.table = F)
training$type <- 'training'

range <- c('50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100')

#--- CDR3b
mlib.count <- NULL
perf <- NULL
mlib.files <- list.files('/Users/vy/Desktop/GS/TCR-epitope-prediction/similarity_fulltest/CDR3b/', full.names = T)
prediction <- NULL
prediction <- list()
for (file in mlib.files) {
  temp <- fread(file, data.table = F)
  prediction[[file]] <- temp
  if (is.null(mlib.count)) mlib.count <- nrow(temp)
  else mlib.count <- c(mlib.count, nrow(temp))

  if (is.null(perf)) perf <- calculate_metrics(temp$predict_proba, temp$binder, 0.5)
  else perf <- rbind(perf, calculate_metrics(temp$predict_proba, temp$binder, 0.5))
}
perf <- as.data.frame(perf)
perf <- lapply(perf, as.numeric)
perf <- as.data.frame(perf)
perf$count <- as.numeric(mlib.count)
colnames(perf) <- c('AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Count')
perf$Range <- range
perf$count.rela <- perf$Count/sum(perf$Count)

perf.melt <- melt(perf, id.vars = c('Range', 'Count', 'count.rela'))

y.ratio = 2.5
ggplot(perf.melt[perf.melt$variable == 'Specificity',], aes(x = Range, y = value)) + # changed depending on metrics
  geom_bar(stat = 'identity', fill = 'darkblue') +
  list(geom_step(aes(x = Range, y = count.rela * y.ratio), size = 1.5,
                 color = 'red', group = 1, position = position_nudge(x = -0.5)),
       # add one more line on the right
       geom_segment(mapping = aes(x = 9.45, y = count.rela[10] * y.ratio, xend = 10.49, yend = count.rela[10] * y.ratio), color = 'red', size = 1.5)) +
  geom_text(aes(label = round(value, 2)), stat = "identity", vjust = 1.5, size = 4, colour = "white") +
  cowplot::theme_cowplot() +
  theme(axis.text.x=element_text(angle = 45, vjust = 1, hjust = 1),
        axis.ticks.x=element_blank()) + xlab('Similarity ranges on CDR3β') + ylab('Specificity') + # changed depending on metrics
  scale_y_continuous(sec.axis = sec_axis( trans=~./ y.ratio, name="Data proportion"), limit = c(0, 1))

#--- epitope
mlib.count <- NULL
perf <- NULL
mlib.files <- list.files('/Users/vy/Desktop/GS/TCR-epitope-prediction/similarity_fulltest/epitope/', full.names = T)
prediction <- NULL
prediction <- list()
for (file in mlib.files) {
  temp <- fread(file, data.table = F)
  prediction[[file]] <- temp
  if (is.null(mlib.count)) mlib.count <- nrow(temp)
  else mlib.count <- c(mlib.count, nrow(temp))

  if (is.null(perf)) perf <- calculate_metrics(temp$predict_proba, temp$binder, 0.5)
  else perf <- rbind(perf, calculate_metrics(temp$predict_proba, temp$binder, 0.5))
}
perf <- as.data.frame(perf)
perf <- lapply(perf, as.numeric)
perf <- as.data.frame(perf)
perf$count <- as.numeric(mlib.count)
colnames(perf) <- c('AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Count')
perf$Range <- range
perf$count.rela <- perf$Count/sum(perf$Count)

perf.melt <- melt(perf, id.vars = c('Range', 'Count', 'count.rela'))

y.ratio = 2
ggplot(perf.melt[perf.melt$variable == 'Specificity',], aes(x = Range, y = value)) + # changed depending on metrics
  geom_bar(stat = 'identity', fill = 'darkblue') +
  list(geom_step(aes(x = Range, y = count.rela * y.ratio), size = 1.5,
                 color = 'red', group = 1, position = position_nudge(x = -0.5)),
       # add one more line on the right
       geom_segment(mapping = aes(x = 9.45, y = count.rela[10] * y.ratio, xend = 10.49, yend = count.rela[10] * y.ratio), color = 'red', size = 1.5)) +
  geom_text(aes(label = round(value, 2)), stat = "identity", vjust = 1.5, size = 4, colour = "white") +
  cowplot::theme_cowplot() +
  theme(axis.text.x=element_text(angle = 45, vjust = 1, hjust = 1),
        axis.ticks.x=element_blank()) + xlab('Similarity ranges on epitope') + ylab('Specificity') + # changed depending on metrics
  scale_y_continuous(sec.axis = sec_axis( trans=~./ y.ratio, name="Data proportion"), limit = c(0, 1))
