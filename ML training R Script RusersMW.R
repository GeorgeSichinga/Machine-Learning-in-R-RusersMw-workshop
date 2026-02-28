# =============================================================================
# Machine Learning in R - R Users Malawi Workshop
# Facilitator: George Sichinga
# Date: 22 February 2026
# =============================================================================


# -----------------------------------------------------------------------------
# SLIDE 4: Install & Load Key Packages
# -----------------------------------------------------------------------------

# Install all packages needed for this workshop in one go.
# You only need to run install.packages() once on your machine.
install.packages(c(
  "caret",
  "tidymodels",
  "randomForest",
  "rpart",
  "e1071",
  "xgboost",
  "arules",
  "factoextra",
  "ggplot2",
  "dplyr"
))

# caret (Classification And REgression Training) is the main framework
# we will use today. It provides a unified interface for training,
# tuning and evaluating many different ML models.
library(caret)


# -----------------------------------------------------------------------------
# SLIDE 6: Data Splitting - Train / Validation / Test Sets
# -----------------------------------------------------------------------------

library(caret)

# A small agricultural dataset representing Malawi districts.
# rainfall and fertilizer are our predictors; yield is what we want to predict.
data <- data.frame(
  rainfall   = c(850, 780, 920, 810, 870,
                 760, 900, 820, 740, 880),
  fertilizer = c(50, 40, 60, 45, 55,
                 38, 58, 47, 35, 52),
  yield      = c(3.2, 2.8, 3.8, 3.0, 3.5,
                 2.6, 3.7, 3.1, 2.5, 3.4)
)

# set.seed() ensures we get the same random split every time we run the code.
# This is important for reproducibility.
set.seed(42)

# createDataPartition() splits the data while preserving the distribution
# of the outcome variable. p = 0.7 means 70% goes to training.
idx   <- createDataPartition(data$yield, p = 0.7, list = FALSE)
train <- data[idx, ]
test  <- data[-idx, ]

# trainControl() defines how the model will be validated during training.
# method = "cv" means k-fold cross validation; number = 5 means 5 folds.
# In 5-fold CV, the training data is split into 5 equal parts.
# The model trains on 4 parts and validates on the remaining 1, repeated 5 times.
ctrl <- trainControl(
  method = "cv",
  number = 5)

cat("Train rows:", nrow(train), "\n")
cat("Test rows: ", nrow(test),  "\n")


# -----------------------------------------------------------------------------
# SLIDE 10: Linear Regression - Malawi Maize Yield Example
# -----------------------------------------------------------------------------

library(caret)

data <- data.frame(
  rainfall   = c(850, 780, 920, 810, 870, 760, 900),
  fertilizer = c(50,  40,  60,  45,  55,  38,  58),
  yield      = c(3.2, 2.8, 3.8, 3.0, 3.5, 2.6, 3.7)
)

set.seed(42)
idx   <- createDataPartition(data$yield, p = 0.7, list = FALSE)
train <- data[idx, ]
test  <- data[-idx, ]

ctrl <- trainControl(method = "cv", number = 5)

# train() fits the model. The formula yield ~ rainfall + fertilizer tells R
# to predict yield using rainfall and fertilizer as input variables.
# method = "lm" specifies ordinary linear regression.
lr_model <- train(
  yield ~ rainfall + fertilizer,
  data      = train,
  method    = "lm",
  trControl = ctrl
)

# print() shows the cross-validated performance metrics (RMSE, Rsquared, MAE).
# summary() shows the regression coefficients and their significance.
print(lr_model)
summary(lr_model)

# predict() applies the trained model to the test set.
# postResample() computes RMSE, R-squared and MAE on the test predictions.
# Lower RMSE and higher R-squared indicate a better fit.
pred <- predict(lr_model, newdata = test)
postResample(pred, test$yield)


# -----------------------------------------------------------------------------
# SLIDE 12: Logistic Regression - Malaria Risk Example
# -----------------------------------------------------------------------------

library(caret)

# This dataset records patient age, local rainfall and malaria test result.
# Logistic regression is used when the outcome is categorical (Pos / Neg),
# not continuous. It models the probability of belonging to a class.
health <- data.frame(
  age      = c(5, 30, 12, 45, 8, 60, 25, 3, 50, 18,
               35, 22, 14, 55, 40),
  rainfall = c(850, 700, 900, 650, 880, 620, 720,
               870, 640, 800, 730, 810, 750, 610, 760),
  malaria  = factor(c("Pos", "Neg", "Pos", "Neg",
                      "Pos", "Neg", "Neg", "Pos",
                      "Neg", "Pos", "Neg", "Pos",
                      "Neg", "Neg", "Pos"))
)

# relevel() sets "Neg" as the reference class so that the model predicts
# the probability of testing positive relative to negative.
health$malaria <- relevel(health$malaria, ref = "Neg")

set.seed(42)
idx     <- createDataPartition(health$malaria, p = 0.7, list = FALSE)
train_h <- health[idx, ]
test_h  <- health[-idx, ]

# classProbs = TRUE allows the model to output probabilities, not just labels.
# summaryFunction = twoClassSummary enables ROC, Sensitivity and Specificity
# as performance metrics, which are more informative than accuracy alone
# when classes may be imbalanced.
ctrl <- trainControl(
  method          = "cv",
  number          = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary
)

# method = "glm" with family = binomial fits logistic regression.
# metric = "ROC" tells caret to select the best model based on AUC.
lg <- train(
  malaria ~ age + rainfall,
  data      = train_h,
  method    = "glm",
  family    = binomial,
  metric    = "ROC",
  trControl = ctrl,
  control   = glm.control(maxit = 100)
)

pred <- predict(lg, newdata = test_h)
confusionMatrix(pred, test_h$malaria, positive = "Pos")


# -----------------------------------------------------------------------------
# SLIDE 14: Decision Tree
# -----------------------------------------------------------------------------

library(caret)
library(rpart)
library(rpart.plot)

# Decision trees split the data into branches based on predictor values.
# Each split tries to reduce prediction error as much as possible.
# They are easy to interpret and visualise, making them a good starting point.
agri_data <- data.frame(
  rainfall   = c(850, 780, 920, 810, 870, 760, 900),
  fertilizer = c(50,  40,  60,  45,  55,  38,  58),
  yield      = c(3.2, 2.8, 3.8, 3.0, 3.5, 2.6, 3.7)
)

set.seed(42)
idx     <- createDataPartition(agri_data$yield, p = 0.7, list = FALSE)
train_a <- agri_data[idx, ]
test_a  <- agri_data[-idx, ]

# LOOCV (Leave-One-Out Cross-Validation) trains on all observations except one
# and tests on that single observation. This is repeated for every observation.
# It is thorough but computationally expensive on large datasets.
ctrl   <- trainControl(method = "LOOCV")

# cp (complexity parameter) controls how deep the tree grows.
# A smaller cp allows more splits and a more complex tree.
# We search across a range of cp values to find the one with lowest RMSE.
cpGrid <- expand.grid(cp = seq(0.0001, 0.02, by = 0.002))

dt_model <- train(
  yield ~ rainfall + fertilizer,
  data      = train_a,
  method    = "rpart",
  trControl = ctrl,
  tuneGrid  = cpGrid,
  metric    = "RMSE",
  control   = rpart.control(minsplit = 2)
)

# bestTune shows the cp value that produced the lowest cross-validated RMSE.
print(dt_model$bestTune)

# rpart.plot() draws the decision tree so we can see every split and leaf node.
rpart.plot(dt_model$finalModel,
           type  = 2,
           extra = 101,
           main  = "Maize Yield Decision Tree")

# Variable importance tells us which predictors contributed most to the splits.
if (!is.null(dt_model$finalModel$splits)) {
  print(varImp(dt_model))
} else {
  cat("Tree has no splits → no variable importance available\n")
}

pred <- predict(dt_model, test_a)
postResample(pred, test_a$yield)


# -----------------------------------------------------------------------------
# SLIDE 16: Random Forest - Food Security Classification
# -----------------------------------------------------------------------------

library(caret)
library(randomForest)

# Random Forest builds many decision trees on random subsets of the data
# and combines their predictions. This reduces overfitting and improves
# generalisation compared to a single decision tree.
fs <- data.frame(
  rainfall   = c(850, 600, 920, 500, 870, 450,
                 810, 580, 900, 630, 860, 490),
  fertilizer = c(50,  20,  60,  15,  55,  10,
                 45,  18,  58,  22,  52,  12),
  status     = factor(c("Secure", "AtRisk",
                        "Secure", "AtRisk",
                        "Secure", "AtRisk",
                        "Secure", "AtRisk",
                        "Secure", "AtRisk",
                        "Secure", "AtRisk"))
)

fs$status <- relevel(fs$status, ref = "AtRisk")

set.seed(42)
idx      <- createDataPartition(fs$status, p = 0.7, list = FALSE)
train_rf <- fs[idx, ]
test_rf  <- fs[-idx, ]

ctrl   <- trainControl(method = "cv", number = 3)

# mtry is the number of variables randomly considered at each split.
# Tuning mtry helps us find the right balance between bias and variance.
# With only 2 predictors here, we try both possible values (1 and 2).
rfGrid <- expand.grid(mtry = 1:2)

# ntree = 500 means the forest will contain 500 individual trees.
# importance = TRUE allows us to extract variable importance after training.
rf <- train(
  status ~ rainfall + fertilizer,
  data       = train_rf,
  method     = "rf",
  ntree      = 500,
  trControl  = ctrl,
  tuneGrid   = rfGrid,
  importance = TRUE
)

print(rf$bestTune)

# varImpPlot() shows how much each variable contributes to model accuracy.
# MeanDecreaseAccuracy: how much accuracy drops when a variable is removed.
# MeanDecreaseGini: how much a variable reduces impurity across all trees.
varImpPlot(rf$finalModel)

pred_rf <- predict(rf, test_rf)
confusionMatrix(pred_rf, test_rf$status, positive = "AtRisk")


# -----------------------------------------------------------------------------
# SLIDE 20: Dimension Reduction - PCA
# -----------------------------------------------------------------------------

library(factoextra)

# PCA (Principal Component Analysis) reduces many correlated variables into
# a smaller set of uncorrelated components called principal components (PCs).
# Each PC captures as much variance in the data as possible.
# This is useful for visualisation and for removing redundant information.
survey <- data.frame(
  calorie_intake   = c(1850, 1620, 2100, 1700, 2050, 1580, 1950, 1800, 1650),
  diet_diversity   = c(5, 3, 7, 4, 6, 3, 6, 5, 4),
  food_expenditure = c(45, 32, 60, 38, 55, 30, 52, 44, 35),
  stunting_rate    = c(35, 48, 20, 42, 25, 50, 28, 38, 45),
  wasting_rate     = c(8,  15,  4, 12,  5, 17,  6, 10, 14),
  market_access    = c(80, 55, 90, 65, 85, 50, 88, 72, 58)
)

# scale. = TRUE standardises all variables to have mean 0 and standard
# deviation 1 before computing PCs. This is necessary when variables
# are measured on different scales (e.g. calories vs diversity scores).
pca_result <- prcomp(survey, scale. = TRUE)

# summary() shows the proportion of total variance explained by each PC.
# We look for the point where adding more PCs gives diminishing returns.
summary(pca_result)

# The biplot shows both observations (districts) and variables (arrows)
# in the same PC space. Variables pointing in the same direction are
# positively correlated; opposing arrows indicate negative correlation.
fviz_pca_biplot(pca_result, repel = TRUE,
                title   = "PCA: Malawi Food Security",
                col.var = "steelblue",
                col.ind = "darkred")

# The scree plot shows variance explained by each PC.
# A common rule is to keep PCs before the point where the curve levels off.
fviz_eig(pca_result, addlabels = TRUE,
         main = "Variance Explained by PCs")


# -----------------------------------------------------------------------------
# SLIDE 21: Association Rule Mining - Apriori Algorithm
# -----------------------------------------------------------------------------

library(arules)
library(arulesViz)

# Association rule mining finds items that frequently appear together.
# Here each transaction is the set of crops grown by one farm household.
# The goal is to discover crop combinations that commonly co-occur.
crops <- list(
  c("Maize", "Soybean",   "Groundnut"),
  c("Maize", "Tobacco",   "Cassava"),
  c("Maize", "Soybean",   "Cassava"),
  c("Maize", "Groundnut", "Cassava", "Rice"),
  c("Soybean", "Groundnut", "Sunflower"),
  c("Maize", "Soybean",   "Tobacco"),
  c("Maize", "Cassava",   "Rice"),
  c("Groundnut", "Soybean", "Cassava"),
  c("Maize", "Soybean",   "Groundnut", "Rice"),
  c("Maize", "Tobacco",   "Groundnut")
)

# Convert the list of crop sets into a transactions object that arules can use.
trans <- as(crops, "transactions")
summary(trans)

# supp (support): the minimum proportion of transactions containing the rule.
# conf (confidence): how often the rule is correct when the left side occurs.
# minlen = 2 ensures rules have at least one item on each side.
rules <- apriori(
  trans,
  parameter = list(
    supp   = 0.20,
    conf   = 0.60,
    minlen = 2
  )
)

cat("Number of rules generated:", length(rules), "\n")

if (length(rules) > 0) {
  # Sorting by lift ranks rules by how much more likely the right side item
  # is given the left side, compared to it occurring by chance.
  # A lift value greater than 1 indicates a genuine positive association.
  sorted_rules <- sort(rules, by = "lift")
  inspect(head(sorted_rules, 5))
  plot(rules, method = "graph",
       main = "Crop Association Rules")
} else {
  cat("No rules found. Consider lowering support or confidence.\n")
}


# -----------------------------------------------------------------------------
# SLIDE 22: Model Evaluation
# -----------------------------------------------------------------------------

library(caret)

# Model evaluation tells us how well each trained model performs on data
# it has never seen before (the test set). For regression models we use
# RMSE (Root Mean Squared Error), R-squared and MAE (Mean Absolute Error).
# For classification models we use a confusion matrix which shows counts
# of correct and incorrect predictions broken down by class.

cat("\n==============================\n")
cat("Logistic Regression Evaluation\n")
cat("==============================\n")

pred_logit <- predict(lg, newdata = test_h)
pred_logit <- factor(pred_logit, levels = levels(test_h$malaria))
print(confusionMatrix(pred_logit, test_h$malaria, positive = "Pos"))


cat("\n==============================\n")
cat("Random Forest Evaluation\n")
cat("==============================\n")

pred_rf <- predict(rf, newdata = test_rf)
pred_rf  <- factor(pred_rf, levels = levels(test_rf$status))
print(confusionMatrix(pred_rf, test_rf$status, positive = "AtRisk"))


cat("\n==============================\n")
cat("Random Forest Cross-Validation Summary\n")
cat("==============================\n")

# rf$results shows the average performance across all CV folds for each
# value of mtry that was tried during tuning.
# rf$resample shows the individual performance for each fold separately,
# which helps us understand how stable the model is across different data subsets.
print(rf$results)
print(rf$resample)


# =============================================================================
# END OF SCRIPT
# =============================================================================