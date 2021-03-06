## create a new data frame with all the variables, cuz some packages doesn't take dplyr %>%
nhanes_data1 <- nhanes_data %>% mutate(BMXBMI2 = BMXBMI^2)
nhanes_category <- nhanes_data1 %>% select(-c(bornUS, BMXHT, BMXWT, BMXWAIST, BMXBMI, BMXBMI2))
nhanes_first <- nhanes_data1 %>% select(-c(bornUS, BMXHT, BMXWT, BMXWAIST, BMI_cat, BMXBMI2))
nhanes_second <- nhanes_data1 %>% select(-c(BMXHT, BMXWT, BMXWAIST, BMI_cat))
nhanes_finalmodel1 <- nhanes_data1 %>% select(-c(highchol, INDFMPIR, marital, cancer, alcoholic, race, hypertension, mi, smoker, BMXWT, BMXHT, BMXWAIST, BMI_cat))


dataset = nhanes_second


set.seed(86)
library(glmnet)
N = nrow(dataset)
train.index = sample(1:N, round(4*N/5))
test.index = - train.index

x_vars = model.matrix(dead ~., dataset[train.index,])[, -12]
y_var = dataset$dead[train.index]
x.test = model.matrix(dead ~., dataset[test.index,])[, -12]
y.test = dataset[-train.index, "dead"]

## glm

## fit a model with only categorical BMIs
lr.all <- glm(dead~., data = dataset,
                       family = "binomial", subset = train.index)
#lr.all <- glm(dead~ educ + diabetic + stroke + female + BMXBMI*age, data = dataset,
#              family = "binomial", subset = train.index)

lr.II <- glm(dead ~ educ + diabetic + stroke + female + bornUS + BMXBMI*age, data = dataset,
              family = "binomial", subset = train.index)
lr.I <- glm(dead ~ age + educ + diabetic + stroke + female + bornUS + BMXBMI + BMXBMI2, data = dataset,
              family = "binomial", subset = train.index)

lr.mod <- step(lr.all)

summary(lr.mod)
summary(lr.I)
summary(lr.II)
# prediction
lr.pred = predict(lr.mod, newdata = dataset[test.index, ], type = "response")
lr.pred1 = predict(lr.I, newdata = dataset[test.index, ], type = "response")
lr.pred2 = predict(lr.II, newdata = dataset[test.index, ], type = "response")

### SUBGROUP ANALYSIS
# Not born in US
not.bornUS <- nhanes_data %>% subset(bornUS == 0)
us.model1 <- glm(dead ~ age + educ + diabetic + stroke + female + BMXBMI + I(BMXBMI^2), data = not.bornUS, family = "binomial")
us.model2 <- glm(dead ~ age + educ + diabetic + stroke + female + BMXBMI + I(BMXBMI*age), data = not.bornUS, family = "binomial")

# Never smokers
non.smoking <- nhanes_data %>% subset(smoker == 0)
smoke.model1 <- glm(dead ~ age + educ + diabetic + stroke + female + BMXBMI + I(BMXBMI^2), data = non.smoking, family = "binomial")
smoke.model2 <- glm(dead ~ age + educ + diabetic + stroke + female + BMXBMI + I(BMXBMI*age), data = non.smoking, family = "binomial")

### Ridge

ridge.cv = cv.glmnet(x_vars, y_var, alpha = 0, family = "binomial", type.measure = "class")
coef(ridge.cv)
plot(ridge.cv)

# optimal lambda
ridge.lam = ridge.cv$lambda.1se

# plot optimal model
ridge.mod = glmnet(x_vars, y_var, alpha = 0, family = "binomial")

plot(ridge.mod, xvar="lambda", label = TRUE)
abline(v=log(ridge.lam), lty=2)

# prediction
ridge.pred = predict(ridge.mod, newx = x.test, s = ridge.lam, type = "response", exact = TRUE)


## REMARKS:
  
##  Note that for 2-class logistic regression, we need to specify family=“binomial”. Also for the cross-validation to choose the best lambda, we need to use misclassification error rate (instead of MSE) as the criterion. This is done by type.measure=“class”.

## When predicting, we want to have the exact predicted probabilities, so use type=“response”, exact=TRUE.

## When choosing the optimal lambda, we have two candidates: lambda.min that minimizes the cross validated misclassification error, or lambda.1se, the error at which is one standard error above the minimum error. The choice is subjective here: if we want higher prediction performance, we choose lambda.min; if we want higher regularization and hence more parsimonious model, we choose lambda.1se.

## Lasso

# cross-validation
set.seed(1)
lasso.cv = cv.glmnet(x_vars, y_var, alpha = 1, family = "binomial", type.measure = "class")
coef(lasso.cv)
plot(lasso.cv)

# optimal lambda
lasso.lam <- lasso.cv$lambda.1se  # lasso.cv$lambda.min

# plot optimal model
lasso.mod <- glmnet(x_vars, y_var, alpha = 1, family = "binomial")
plot(lasso.mod, xvar = "lambda", label = TRUE)
abline(v = log(lasso.lam), lty=2)

# prediction
lasso.pred <- predict(lasso.mod, newx = x.test, s = lasso.lam, type = "response", exact = TRUE)


## Elastic Net

# candidates for alpha
alphas <- seq(0, 1, 0.05)

# random partition for 10-fold cross-validation
K = 10
n = nrow(x_vars)
fold = rep(0, n)
set.seed(1)
shuffled.index = sample(n, n, replace = FALSE)
fold[shuffled.index] = rep(1:K, length.out = n)
table(fold)

# cross-validation to find the best alpha-lambda combination
en.cv.error = data.frame(alpha = alphas)
for (i in 1:length(alphas)){
  en.cv = cv.glmnet(x_vars, y_var, alpha = alphas[i], foldid = fold, family = "binomial", type.measure = "class")
  en.cv.error[i, "lambda"] = en.cv$lambda.1se
  en.cv.error[i, "error"] = min(en.cv$cvm) + en.cv$cvsd[which.min(en.cv$cvm)]
}

# optimal lambda and alpha
en.lam = en.cv.error[which.min(en.cv.error$error), "lambda"]
en.alpha = en.cv.error[which.min(en.cv.error$error), "alpha"]

# plot optimal alpha
plot(en.cv.error$alpha, en.cv.error$error, type = "l")
abline(v = en.alpha, lty = 2)

# plot the optimal model
en.mod = glmnet(x_vars, y_var, alpha = en.alpha, family = "binomial")
plot(en.mod, xvar = "lambda", label = TRUE)
abline(v = log(en.lam), lty = 2)

# prediction
en.pred = predict(en.mod, newx = x.test, s = en.lam, type = "response", exact = TRUE)

library("ROCR")

prediction.lr <- prediction(lr.pred, y.test)
prediction.lr1 <- prediction(lr.pred1, y.test)
prediction.lr2 <- prediction(lr.pred2, y.test)
prediction.ridge <- prediction(ridge.pred, y.test)
prediction.lasso <- prediction(lasso.pred, y.test)
prediction.en <- prediction(en.pred, y.test)

# misclassification rate
err.lr <- performance(prediction.lr, measure = "err")
err.lr1 <- performance(prediction.lr1, measure = "err")
err.lr2 <- performance(prediction.lr2, measure = "err")
err.ridge <- performance(prediction.ridge, measure = "err")
err.lasso <- performance(prediction.lasso, measure = "err")
err.en <- performance(prediction.en, measure = "err")
plot(err.lr, ylim=c(0.1, 0.5))
plot(err.lr1, ylim=c(0.1, 0.5))
plot(err.lr2, add=TRUE, col="brown")
legend(0.55, 0.49, legend = c("Model I", "Model II"), col = c("black", "brown"), lty = 1:1, cex = 0.8)
title("Predictive Accuracy Model I vs Model II")

plot(err.ridge, add=TRUE, col="blue")
plot(err.lasso, add=TRUE, col="red")
plot(err.en, add=TRUE, col="green")
legend(0.55, 0.49, legend = c("Linear", "Ridge", "Lasso", "Elastic Net"), col = c("black", "blue", "red", "green"), lty = 1:1, cex = 0.8)
title("Regularization Methods Prediction Performance")

# ROC plot
ROC.lr <- performance(prediction.lr, measure = "tpr", x.measure = "fpr")
ROC.lr1 <- performance(prediction.lr1, measure = "tpr", x.measure = "fpr")
ROC.lr2 <- performance(prediction.lr2, measure = "tpr", x.measure = "fpr")
ROC.ridge <- performance(prediction.ridge, measure = "tpr", x.measure = "fpr")
ROC.lasso <- performance(prediction.lasso, measure = "tpr", x.measure = "fpr")
ROC.en <- performance(prediction.en, measure = "tpr", x.measure = "fpr")
plot(ROC.lr)
plot(ROC.lr1)
abline(a=0, b=1, lty=2) # diagonal line
plot(ROC.lr2, add=TRUE, col="brown")
plot(ROC.ridge, add=TRUE, col="blue")
legend(0.6, 0.35, legend = c("Model I", "Model II"), col = c("black", "brown"), lty = 1:1, cex = 0.8)
title("ROC of Model I and Model II")

plot(ROC.lasso, add=TRUE, col="red")
plot(ROC.en, add=TRUE, col="green")
legend(0.6, 0.35, legend = c("Linear", "Ridge", "Lasso", "Elastic Net"), col = c("black", "blue", "red", "green"), lty = 1:1, cex = 0.8)
title("ROC of Regularization Methods")


# AUC
as.numeric(performance(prediction.lr, "auc")@y.values)
as.numeric(performance(prediction.lr1, "auc")@y.values)
as.numeric(performance(prediction.lr2, "auc")@y.values)
as.numeric(performance(prediction.ridge, "auc")@y.values)
as.numeric(performance(prediction.lasso, "auc")@y.values)
as.numeric(performance(prediction.en, "auc")@y.values)

## compare coefficients
coef(lr.mod)
coef(ridge.mod, s = ridge.lam)
coef(lasso.mod, s = lasso.lam)
coef(en.mod, s = en.lam)


######## SVM
library(caret)
train_control1 <- trainControl(method = "cv", number = 10, selectionFunction = "best", classProbs = TRUE)
mydata1 <- dataset[train.index, ]
mydata1$dead <- ifelse(mydata1$dead == "1", "Dead", "Alive")
mydata1$dead <- as.factor(mydata1$dead)

# train the model
set.seed(123)
SVM <- train(dead~., data = mydata1, trControl = train_control1, method="svmRadial") 

# report results
print(SVM)

# use Caret's modelLookup function to check the list of parameters can be tuned.
modelLookup("svmRadial")

# grid search for selecting parameters
SVM_grid <- expand.grid(.sigma = c(0.02, 0.03368, 0.05), .C = c(0.20,0.25,0.3))

# train the model
SVM.time11 <- Sys.time() #<= to help you track computation time for A2 Q1
SVM_train <- train(dead~., data=mydata1, trControl=train_control1, method="svmRadial", metric="Accuracy", tuneGrid=SVM_grid) 
SVM.time12 <- Sys.time() #<= to help you track computation time for A2 Q1
print(SVM_train) # see output
SVM.time1 <- SVM.time12 - SVM.time11
SVM.time1

SVM_grid2 <- expand.grid(.sigma = c(0.06, 0.07, 0.1), .C = c(0.18,0.20,0.22))

# train the model
SVM.time31 <- Sys.time() #<= to help you track computation time for A2 Q1
SVM_train2 <- train(dead~., data=mydata1, trControl=train_control1, method="svmRadial", metric="Accuracy", tuneGrid=SVM_grid2) 
SVM.time32 <- Sys.time() #<= to help you track computation time for A2 Q1
print(SVM_train2) # see output
SVM.time3 <- SVM.time32 - SVM.time31
SVM.time3

data2 = dataset[test.index, ]
data2$dead <- ifelse(data2$dead == "1", "Dead", "Alive")
data2$dead <- as.factor(data2$dead)


# Prediction
SVM_train2_prob <- predict(SVM_train2, data2, type = "prob")
# Thresholding
threshold_d1<-seq(0.001,0.999,0.0001)
cost_d1<-rep(0,length(threshold_d1))
accuracy_d1<-rep(0,length(threshold_d1))
kappa_d1<-rep(0,length(threshold_d1))
for(i in 1:length(threshold_d1)){
  pred <- factor(ifelse(SVM_train2_prob[, "Dead"] > threshold_d1[i], "Dead", "Alive"))
  cost_matrix_d <- confusionMatrix(pred, data2$dead)
  cost_d1[i] <- -cost_matrix_d$table[2,1] - cost_matrix_d$table[1,2]
  accuracy_d1[i] <- cost_matrix_d$overall[1]
  kappa_d1[i] <- cost_matrix_d$overall[2]
}
a<-which.max(cost_d1)
a
threshold_d1[a]
accuracy_d1[a]
kappa_d1[a]
cost_d1[a]


# Prediction
SVM_train2_prob <- predict(SVM_train2, data2, type = "prob")
# Thresholding
threshold <- threshold_d1[a]
SVM_train2_pred <- factor(ifelse(SVM_train2_prob[, "Dead"] > threshold, "Dead", "Alive") )
# ConfusionMatrix and other performance metrics
cost_matrix_a<-confusionMatrix(SVM_train2_pred, data2$dead)
cost_matrix_a

SVM_train2_pred = ifelse(SVM_train2_pred == "Alive", 0, 1)


##### GBM

library("gbm")

ds <- c(1, 2, 4, 6, 8)  # candidates for d
lambdas <- c(0.01, 0.005, 0.001, 0.0005)  # candidates for lambda

tune.out <- data.frame()
for (d in ds) {  # iterate over all possible d
for (lambda in lambdas) {  # iterate over all possible lambda
    # **calculate max n.trees by my secret formula**
  for (n in (1:10) * 50 / (lambda * sqrt(d))) {  # iterate over all possible n.trees
      set.seed(321)
      gbm.mod <- gbm(dead ~ ., data = dataset[train.index,], distribution = "bernoulli", n.trees = n, shrinkage = lambda, interaction.depth = d, cv.folds = 10)
      n.opt <- gbm.perf(gbm.mod, method="cv")
      cat("n =", n, " n.opt =", n.opt, "\n")
      if (n.opt / n < 0.95) break
    }
    cv.err <- gbm.mod$cv.error[n.opt]
    pred <- predict(gbm.mod, newdata = dataset[test.index,], n.trees = n.opt)
    test.err <- mean((pred - y.test)^2)
    out <- data.frame(d = d, lambda = lambda, n = n, n.opt = n.opt, cv.err = cv.err, test.err = test.err)
    print(out)
    tune.out <- rbind(tune.out, out)
  }
  
}
# fit a boosting model with optimal parameters
set.seed(321)
bmi.gbm <- gbm(dead ~ ., data = dataset[train.index,], distribution = "bernoulli", n.trees = 2050, shrinkage = 0.01, interaction.depth = 6)
bmi.gbm

# predict
prob.gbm <- predict(bmi.gbm, newdata = dataset[test.index,], n.trees = 2050, type="response")

# misclassification error in test data
pred.gbm <- as.factor(ifelse(prob.gbm > 0.5, "Yes", "No"))
table(pred.gbm, y.test)

# variable importance
summary(bmi.gbm)

# partial plot in gbm
plot(bmi.gbm, i = "BMXBMI", type = "response")
# proved that low BMX (<20) and very high BMI (>40) increase the chances of death

plot(bmi.gbm, i = "age", type = "response")
# higher the age, higher risks of dying (very reasonable haha)

plot(bmi.gbm, i = "educ", type = "response")
# higher the education level, lower the chance of dying

plot(bmi.gbm, i = "female", type = "response")
# female lower chance of dying

plot(bmi.gbm, i="highchol", type="response")

plot(bmi.gbm, i=c("age", "BMXBMI"), type="response")



## compare models
prediction.lr <- prediction(lr.pred, y.test)
prediction.ridge <- prediction(ridge.pred, y.test)
prediction.lasso <- prediction(lasso.pred, y.test)
prediction.en <- prediction(en.pred, y.test)
prediction.svm <- prediction(SVM_train2_prob[,2], y.test)
prediction.gbm <- prediction(prob.gbm, y.test)


# misclassification rate
err.lr <- performance(prediction.lr, measure = "err")
err.ridge <- performance(prediction.ridge, measure = "err")
err.lasso <- performance(prediction.lasso, measure = "err")
err.en <- performance(prediction.en, measure = "err")
err.svm <- performance(prediction.svm, measure = "err")
err.gbm <- performance(prediction.gbm, measure = "err")
plot(err.lr, ylim=c(0.1, 0.5))
#plot(err.ridge, add=TRUE, col="blue")
#plot(err.lasso, add=TRUE, col="red")
#plot(err.en, add=TRUE, col="green")
plot(err.svm, add=TRUE, col="purple")
plot(err.gbm, col="pink", add=TRUE)
legend(0.55, 0.49, legend = c("Linear", "SVM", "GBM"), col = c("black", "purple", "pink"), lty = 1:1, cex = 0.8)
title("GLM vs SVM vs GBM Prediction Performance")

# ROC plot
ROC.lr <- performance(prediction.lr, measure = "tpr", x.measure = "fpr")
ROC.ridge <- performance(prediction.ridge, measure = "tpr", x.measure = "fpr")
ROC.lasso <- performance(prediction.lasso, measure = "tpr", x.measure = "fpr")
ROC.en <- performance(prediction.en, measure = "tpr", x.measure = "fpr")
ROC.svm <- performance(prediction.svm, measure = "tpr", x.measure = "fpr")
ROC.gbm <- performance(prediction.gbm, measure = "tpr", x.measure = "fpr")

plot(ROC.lr)
abline(a=0, b=1, lty=2) # diagonal line
#plot(ROC.ridge, add=TRUE, col="blue")
#plot(ROC.lasso, add=TRUE, col="red")
#plot(ROC.en, add=TRUE, col="green")
plot(ROC.svm, add=TRUE, col="purple")
plot(ROC.gbm, add=TRUE, col="pink")
legend(0.65, 0.4, legend = c("Linear", "SVM", "GBM"), col = c("black", "purple", "pink"), lty = 1:1, cex = 0.8)
title("ROC: GLM vs SVM vs GBM")



# AUC
as.numeric(performance(prediction.lr, "auc")@y.values)
as.numeric(performance(prediction.ridge, "auc")@y.values)
as.numeric(performance(prediction.lasso, "auc")@y.values)
as.numeric(performance(prediction.en, "auc")@y.values)
as.numeric(performance(prediction.svm, "auc")@y.values)
as.numeric(performance(prediction.gbm, "auc")@y.values)


BMI = seq(16,44,0.1)
BMI_Effect = -0.387671*numbers + 0.005892*numbers^2
plot(BMI, BMI_Effect)
title("GLM Model I BMI Effect")

theshold = 0.5
lr.pred1_a <- factor(ifelse(lr.pred1 > threshold, 1, 0) )
lr.pred2_a <- factor(ifelse(lr.pred2 > threshold, 1, 0) )
ridge.pred <- factor(ifelse(ridge.pred > threshold, 1, 0) )
lasso.pred <- factor(ifelse(lasso.pred > threshold, 1, 0) )
en.pred <- factor(ifelse(en.pred > threshold, 1, 0) )
svm.pred <- factor(ifelse(SVM_train2_pred > threshold, 1, 0) )
gbm.pred <- factor(ifelse(prob.gbm > threshold, 1, 0) )

cost_matrix_lr1<-confusionMatrix(lr.pred1_a, as.factor(dataset[test.index,]$dead))
cost_matrix_lr2<-confusionMatrix(lr.pred2_a, as.factor(dataset[test.index,]$dead))
cost_matrix_ridge<-confusionMatrix(ridge.pred, as.factor(dataset[test.index,]$dead))
cost_matrix_lasso<-confusionMatrix(lasso.pred, as.factor(dataset[test.index,]$dead))
cost_matrix_en<-confusionMatrix(en.pred, as.factor(dataset[test.index,]$dead))
cost_matrix_svm<-confusionMatrix(svm.pred, as.factor(dataset[test.index,]$dead))
cost_matrix_gbm<-confusionMatrix(gbm.pred, as.factor(dataset[test.index,]$dead))

cost_matrix_lr1
cost_matrix_lr2
cost_matrix_ridge
cost_matrix_lasso
cost_matrix_en
cost_matrix_svm
cost_matrix_gbm



##################################### Model Diagnostic ########################
########################## GLM diagostic ##############################
glmDiag_sq <- glm.diag(fit_step_sq)
glmDiag_cat <- glm.diag(fit_step2)
glmDiag_sq_df <- data.frame(res = glmDiag_sq$res,
                            rd = glmDiag_sq$rd,
                            rp =glmDiag_sq$rp,
                            cook = glmDiag_sq$cook,
                            h = glmDiag_sq$h)

glmDiag_cat_df <- data.frame(res = glmDiag_cat$res,
                            rd = glmDiag_cat$rd,
                            rp =glmDiag_cat$rp,
                            cook = glmDiag_cat$cook,
                            h = glmDiag_cat$h)

plot(glmDiag_sq$h/(1 - glmDiag_sq$h),glmDiag_sq$cook)

plot(x = NULL)

library(ggplot2)
sq_cook <- ggplot(data = glmDiag_sq_df, mapping = aes(x=cook)) + 
                    geom_histogram(bins =90) + 
                    geom_vline(xintercept = 0.16, linetype="dotted", size = 1) +
                    theme_bw()

cat_cook <- ggplot(data = glmDiag_cat_df, mapping = aes(x=cook)) + 
  geom_histogram(bins =90) + 
  geom_vline(xintercept = 0.143, linetype="dotted", size = 1) +
  theme_bw()

sq_cat_diag <- rbind(glmDiag_sq_df, glmDiag_cat_df)
sq_cat_diag$model <- c(rep("BMI Category", nrow(glmDiag_sq_df)),
                       rep("BMI Squared", nrow(glmDiag_cat_df)))
cat_cook
#abline(v = pf(0.5, df1 = 8, 993))


ggplot(data = sq_cat_diag, mapping = aes(x = cook, fill=model)) + 
  geom_histogram(bins =90, alpha=0.4, position = "identity") + 
  #geom_vline(xintercept = 0.143, linetype="dotted", size = 1, col="red") +
  theme_bw()

ggplot(data = sq_cat_diag, mapping = aes(x=h/(1-h),y = cook, col=model)) + 
  geom_point(alpha = 0.6) + 
  #geom_vline(xintercept = 0.143, linetype="dotted", size = 1, col="red") +
  theme_bw()

library(boot)
glm.diag.plots(fit_step2)
glm.diag.plots(fit_step_sq)
library(jtools)


jtools::plot_summs(fit_step_sq, scale=T,inner_ci_level = .9,plot.distributions = TRUE)
jtools::plot_summs(fit_step2, scale=T,inner_ci_level = .9,plot.distributions = TRUE)


jtools::plot_summs(fit_step_sq, fit_step2, scale=T,inner_ci_level = .9, model.names = c("Model 1", "Model 2"))
jtools::plot_summs(fit_step2, scale=T,inner_ci_level = .9,plot.distributions = TRUE)

summ(fit_step2,vifs = TRUE)
summ(fit_step_sq, vifs = TRUE)


export_summs(fit_step2, fit_step_sq, scale = TRUE, to.file = "docx", file.name = "./test.docx", model.names = c("Model1 (BMI as categorical data)", "Model2 (BMI^2 introduced)"))

