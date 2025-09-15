
# I. Exploratory Data Analysis
# 1. Correlation analysis
library(ggplot2)
library(reshape2)
library(RColorBrewer)
numeric_vars <- Dataset[sapply(Dataset, is.numeric)]
cor_matrix <- cor(numeric_vars, use = "complete.obs")
melted_cor <- melt(cor_matrix)
ggplot(data = melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  scale_fill_gradient2(low = "#A62C2C", high = "#27548A", mid = "white",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name = "Correlation") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 9, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Heatmap of Numerical Variables",
       x = "", y = "")


# 2. Similarity matrix
library(dplyr)
library(scales)
compute_similarity <- function(df, var) {
  dummies <- model.matrix(~ get(var) - 1, data = df)
  colnames(dummies) <- gsub("get\\(var\\)", "", colnames(dummies))
  dist_mat <- dist(t(dummies), method = "euclidean")
  max_dist <- max(as.matrix(dist_mat))
  sim_mat <- 1 - as.matrix(dist_mat) / max_dist
  diag(sim_mat) <- 1
  return(sim_mat)
}
plot_similarity <- function(sim_mat, title_text, xlab_text, ylab_text) {
  melted_sim <- melt(sim_mat)
  ggplot(melted_sim, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", value)), size = 4) +
    scale_fill_gradient(low = "white", high = "#003366", limits = c(0, 1)) +
    theme_minimal() +
    labs(title = title_text, x = xlab_text, y = ylab_text, fill = "") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid = element_blank(),
          plot.title = element_text(hjust = 0.5))
}
# Contract similarity
sim_contract <- compute_similarity(Dataset, "Contract")
plot_similarity(sim_contract,
                "Similarity Matrix of Contract Types",
                "Contract", "Contract")

# Payment method similarity
sim_payment <- compute_similarity(Dataset, "Payment.Method")
plot_similarity(sim_payment,
                "Similarity Matrix of Payment Methods",
                "Payment Method", "Payment Method")


#II. Data pre-processing
# 1. Capping outliers  
cap_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  x[x < lower] <- lower
  x[x > upper] <- upper
  return(x)
}
# 2. Identify numeric columns, excluding "Number.of.Dependents"
numeric_cols <- sapply(Dataset, is.numeric)
target_cols <- setdiff(names(Dataset)[numeric_cols], "Number.of.Dependents")
for (col in target_cols) {
  Dataset[[col]] <- cap_outliers(Dataset[[col]])
}

# 3. Encoding dataset
Dataset$Contract <- recode(Dataset$Contract,
                      "Month-to-Month" = 1,
                      "One Year" = 2,
                      "Two Year" = 3)
Dataset$Gender <- recode(Dataset$Gender,
                      "Female" = 0,
                      "Male" = 1)
Dataset$Payment.Method <- recode(Dataset$Payment.Method,
                      "Mailed Check" = 1,
                      "Bank Withdrawal" = 2,
                      "Credit Card" = 3)

# 4. SMOTE
library(smotefamily)
Dataset$Churn <- as.factor(Dataset$Churn)
X <- Dataset[, setdiff(names(Dataset), "Churn")]  
y <- Dataset$Churn                        

# Apply SMOTE
smote_result <- SMOTE(X, y, K = 5, dup_size = 2)
Dataset_smote <- data.frame(smote_result$data)
Dataset_smote$Churn <- as.factor(Dataset_smote$class)
Dataset_smote$class <- NULL 

#Rounding decimals of encoded categorical variables after SMOTE
Dataset_smote$Gender <- round(Dataset_smote$Gender)
Dataset_smote$Contract <- round(Dataset_smote$Contract)
Dataset_smote$Payment.Method <-round(Dataset_smote$Payment.Method)

# 5. Assigning the rounded variables as factors
Dataset_smote$Gender <- factor(Dataset_smote$Gender, levels = c(0, 1), 
                               labels = c("Female", "Male"))
Dataset_smote$Contract <- factor(Dataset_smote$Contract, levels = c(1, 2, 3), 
                                 labels = c("Month-to-Month", "One Year", "Two Year"))
Dataset_smote$Payment.Method <- factor(Dataset_smote$Payment.Method,levels = c(1, 2, 3),
                                      labels = c("Mailed Check", "Bank Withdrawal", "Credit Card"))


#III. Data Processing

#1. Logistic regression

#1.1 Model Training
names(Dataset_smote) <- make.names(names(Dataset_smote))
predictors <- setdiff(names(Dataset_smote), "Churn")
formula <- as.formula(paste("Churn ~", paste(predictors, collapse = " + ")))
logit_model <- glm(formula, data = Dataset_smote, family = binomial)
summary(logit_model)
exp(coef(logit_model))

# 1.2 Model evaluation
library(caret)
library(pROC)
library(e1071)
# Predict probabilities (for class = 1)
pred_prob <- predict(logit_model, type = "response")

# Classify using 0.5 threshold
pred_class <- ifelse(pred_prob >= 0.5, 1, 0)
pred_class <- as.factor(pred_class)
actual <- as.factor(Dataset_smote$Churn)

# Confusion matrix (positive class = "1")
conf_matrix <- confusionMatrix(pred_class, actual, positive = "1")
print(conf_matrix)

# Extract key metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1 <- conf_matrix$byClass["F1"]

# AUC
roc_obj <- roc(actual, pred_prob)
auc_value <- auc(roc_obj)
cat("Accuracy: ", round(accuracy, 3), "\n")
cat("Precision (Churn=1): ", round(precision, 3), "\n")
cat("Recall (Churn=1): ", round(recall, 3), "\n")
cat("F1 Score (Churn=1): ", round(f1, 3), "\n")
cat("AUC: ", round(auc_value, 3), "\n")

# 2. Logistic regression model with interaction: Contract * Monthly.Charge

# 2.1 Train Model
logit_model_interact <- glm(
  Churn ~ Contract * Monthly.Charge + 
    Age + Gender + Tenure.in.Months + 
    Avg.Monthly.GB.Download + Avg.Monthly.Long.Distance.Charges +
    Number.of.Dependents + Payment.Method,
  data = Dataset_smote,
  family = binomial
)
summary(logit_model_interact)
exp(coef(logit_model_interact))

# 2.2 Model evaluation

# Predict probabilities
pred_probs <- predict(logit_model_interact, type = "response")

# Classify based on threshold (0.5 default)
pred_class <- ifelse(pred_probs >= 0.5, 1, 0)

# Create confusion matrix
conf_matrix <- confusionMatrix(as.factor(pred_class), 
                               as.factor(Dataset_smote$Churn), positive = "1")
print(conf_matrix)

# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1 <- conf_matrix$byClass["F1"]
cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1, "\n")
roc_obj <- roc(Dataset_smote$Churn, pred_probs)
auc_value <- auc(roc_obj)
cat("AUC: ", auc_value, "\n")

# 3. Decision tree
# 3.1 Model training

library(rpart)
library(rpart.plot)

Dataset_smote$Churn <- as.factor(Dataset_smote$Churn)
dt_model <- rpart(Churn ~ ., data = Dataset_smote, method = "class")
dt_pred <- predict(dt_model, Dataset_smote, type = "class")
conf_matrix <- confusionMatrix(dt_pred, Dataset_smote$Churn, positive = "1")
print(conf_matrix)
rpart.plot(dt_model, extra = 106, type = 3, fallen.leaves = TRUE)

#3.2 Model evaluation
library(pROC)
dt_prob <- predict(dt_model, Dataset_smote, type = "prob")[, "1"]
conf_matrix <- confusionMatrix(dt_pred, Dataset_smote$Churn, positive = "1")
print(conf_matrix)
cm <- as.matrix(conf_matrix$table)
TP <- cm[2,2]
TN <- cm[1,1]
FP <- cm[2,1]
FN <- cm[1,2]
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * precision * recall / (precision + recall)

cat("\nPrecision:", round(precision, 3))
cat("\nRecall:", round(recall, 3))
cat("\nF1 Score:", round(f1_score, 3))

roc_obj <- roc(Dataset_smote$Churn, dt_prob)
auc_value <- auc(roc_obj)
cat("\nAUC:", round(auc_value, 3))


#Ranking importance
importance <- dt_model$variable.importance
importance_norm <- importance / sum(importance)
importance_df <- data.frame(
  Predictor = names(importance_norm),
  Importance = round(importance_norm, 4)
)
importance_df <- importance_df[order(-importance_df$Importance), ]
print(importance_df)

ggplot(importance_df, aes(x = reorder(Predictor, Importance), y = Importance)) +
  geom_col(fill = "#27548A") +
  coord_flip() +
  labs(title = "Predictor Importance in Decision Tree Model",
       x = "Predictor",
       y = "Normalized Importance") +
  theme_minimal()

# 4. Example: Predicting churn probabilities
new_data <- data.frame(
  Age = 24,
  Gender = factor("Male", levels = c("Female", "Male")),
  Contract = factor("Month-to-Month", levels = c("Month-to-Month", "One Year", "Two Year")),
  Monthly.Charge = 10,
  Tenure.in.Months = 7,
  Avg.Monthly.GB.Download = 11,
  Avg.Monthly.Long.Distance.Charges = 0,
  Number.of.Dependents = 0,
  Payment.Method = factor("Credit Card", 
                          levels = c("Mailed Check", "Bank Withdrawal", "Credit Card"))
)

# Predict churn probability using logistic regression model
churn_prob <- predict(logit_model, newdata = new_data, type = "response")
print(churn_prob)

# Predict churn probability using logistic regression model with an interaction term
churn_prob <- predict(logit_model_interact, newdata = new_data, type = "response")
print(churn_prob)

# Predict churn probability using Decision Tree
dt_prob <- predict(dt_model, new_data, type = "prob")
print(dt_prob)
  