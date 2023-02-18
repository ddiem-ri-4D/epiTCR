# get all data
fulldat <- fread('/Users/vy/epiTCR/data/finalData/finalwithoutHLA.csv', data.table = F)
pos.pep <- unique(fulldat$epitope[fulldat$binder == 1])
neg.pep <- unique(fulldat$epitope[fulldat$binder == 0])

generated <- fread('/Users/vy/epiTCR/data/nonOverlapingPeptide/withoutMHC/generated_data_300000.csv', data.table = F) #%>% sample_n(., size = 300000)
length(unique(generated$epitope))
neg.pep <- unique(c(neg.pep, unique(generated$epitope)))
length(neg.pep[neg.pep %in% pos.pep]) # 44 wt peptides having positive interactions

fulldat <- merge(fulldat, generated[, -4], all = T)

mix <- neg.pep[neg.pep %in% pos.pep]

# pos.only
pos.pep <- pos.pep[!pos.pep %in% mix]
pos.train.pep <- sample(pos.pep, size = 1251)
pos.test.pep <- pos.pep[!pos.pep %in% pos.train.pep]

pos.train.obs <- fulldat[fulldat$epitope %in% pos.train.pep,]
pos.test.obs <- fulldat[fulldat$epitope %in% pos.test.pep,]

# mix
mix.train.pep <- sample(mix, size = 70)
mix.test.pep <- mix[!mix %in% mix.train.pep]

mix.train.obs <- fulldat[fulldat$epitope %in% mix.train.pep,]
train.mix.balance <- NULL
for (pep in mix.train.pep) {
  train.pos <- mix.train.obs[mix.train.obs$epitope == pep & mix.train.obs$binder == 1,] #%>% sample_n(., nrow(pos.train.obs) %/% length(pos.train.pep))
  train.neg <- mix.train.obs[mix.train.obs$epitope == pep & mix.train.obs$binder == 0,] %>% sample_n(., nrow(train.pos))
  if (is.null(train.mix.balance)) train.mix.balance <- train.pos
  else train.mix.balance <- rbind(train.mix.balance, train.pos)
  train.mix.balance <- rbind(train.mix.balance, train.neg)
}
train.mix.balance <- sample_n(train.mix.balance, nrow(pos.train.obs))
dim(train.mix.balance)

mix.test.obs <- fulldat[fulldat$epitope %in% mix.test.pep,]
test.mix.balance <- NULL
for (pep in mix.test.pep) {
  test.pos <- mix.test.obs[mix.test.obs$epitope == pep & mix.test.obs$binder == 1,]
  test.neg <- mix.test.obs[mix.test.obs$epitope == pep & mix.test.obs$binder == 0,] %>% sample_n(., nrow(test.pos))
  if (is.null(test.mix.balance)) test.mix.balance <- test.pos
  else test.mix.balance <- rbind(test.mix.balance, test.pos)
  test.mix.balance <- rbind(test.mix.balance, test.neg)
}

test.mix.10x <- NULL
for (pep in mix.test.pep) {
  test.pos <- mix.test.obs[mix.test.obs$epitope == pep & mix.test.obs$binder == 1,]
  test.neg <- mix.test.obs[mix.test.obs$epitope == pep & mix.test.obs$binder == 0,] %>%
    sample_n(., min(nrow(test.pos) * 10, nrow(.)))
  if (is.null(test.mix.10x)) test.mix.10x <- test.pos
  else test.mix.10x <- rbind(test.mix.10x, test.pos)
  test.mix.10x <- rbind(test.mix.10x, test.neg)
}

# neg.only
neg.pep <- neg.pep[!neg.pep %in% mix]

# add neg peptides
neg.train.pep <- sample(neg.pep, size = 1251)
neg.test.pep <- neg.pep[!neg.pep %in% neg.train.pep]
neg.test.pep.sample <- sample(neg.test.pep, size = length(pos.test.pep))

neg.train.obs <- fulldat[fulldat$epitope %in% neg.train.pep & fulldat$binder == 0,]
neg.train.obs <- sample_n(neg.train.obs, size = nrow(pos.train.obs))
length(unique(neg.train.obs$epitope))

neg.test.obs.full <- fulldat[fulldat$epitope %in% neg.test.pep & fulldat$binder == 0, ]
neg.test.obs.sample <- fulldat[fulldat$epitope %in% neg.test.pep.sample & fulldat$binder == 0, ]
neg.test.obs.sample <- sample_n(neg.test.obs.sample, size = nrow(pos.test.obs))

neg.test.obs.10x <- fulldat[fulldat$epitope %in% neg.test.pep & fulldat$binder == 0, ] %>%
  sample_n(., size = min(nrow(.), nrow(pos.test.obs) * 10))

train.obs <- rbind(cbind(train.mix.balance, cat = 'mix'),
                  cbind(pos.train.obs, cat = 'pos'),
                  cbind(neg.train.obs, cat = 'neg'))
write.table(train.obs, '/Users/vy/epiTCR/data/nonOverlapingPeptide/withoutMHC/train/train.csv', quote = F, row.names = F, sep = ',')

test.obs.full <- rbind(cbind(mix.test.obs, cat = 'mix'),
                       cbind(pos.test.obs, cat = 'pos'),
                       cbind(neg.test.obs.full, cat = 'neg'))
write.table(test.obs.full, '/Users/vy/epiTCR/data/nonOverlapingPeptide/withoutMHC/test/test.csv', quote = F, row.names = F, sep = ',')

