# HarvardX PH125.9x: Capstone Project 2 - Own
# R Code: Online Shopper’s Purchasing Intention Model Comparison
# Author: Dr Luis Satch
# Date: 2024-09-28

# Table of Contents
# 1. Install Libraries .................................................. Line 22
# 2. Load Libraries ..................................................... Line 38
# 3. Data Download and Preparation ...................................... Line 54
# 4. Data Exploration ................................................... Line 82
# 5. Visualising the Distribution of the Target Variable (Revenue) ...... Line 99
# 6. Calculating Correlation with Revenue ............................... Line 145
# 7. Feature Selection .................................................. Line 160
# 8. Data Splitting and Scaling ......................................... Line 175
# 9. Model 1: MLP Model ................................................. Line 195
# 10. Hyperparameter Tuning for MLP Model ............................... Line 220
# 11. Model 2: Random Forest Model ...................................... Line 252
# 12. Model Comparison and Feature Importance ........................... Line 263


##################################################################################
# 1. Install Libraries
##################################################################################

# Install necessary libraries if they are not installed
if(!require(httr)) install.packages("httr", dependencies = TRUE)
if(!require(readr)) install.packages("readr", dependencies = TRUE)
if(!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
if(!require(reshape2)) install.packages("reshape2", dependencies = TRUE)
if(!require(psych)) install.packages("psych", dependencies = TRUE)
if(!require(caret)) install.packages("caret", dependencies = TRUE)
if(!require(caTools)) install.packages("caTools", dependencies = TRUE)
if(!require(scales)) install.packages("scales", dependencies = TRUE)
if(!require(nnet)) install.packages("nnet", dependencies = TRUE)
if(!require(randomForest)) install.packages("randomForest", dependencies = TRUE)

##################################################################################
# 2. Load Libraries
##################################################################################

# Load necessary libraries
library(httr)      # For making HTTP requests to access online data
library(readr)     # For reading and writing CSV
library(ggplot2)   # For creating visualisations and data plots
library(reshape2)  # For creating heatmaps
library(psych)     # For calculating point-biserial correlations
library(caret)     # For one-hot encoding and model evaluation (confusion matrix)
library(caTools)   # For splitting data into training and testing sets
library(scales)    # For data scaling/normalisation
library(nnet)      # For creating neural network models (Multilayer Perceptron)
library(randomForest) # For building random forest models (ensemble learning)

##################################################################################
# 3. Data Download and Preparation
##################################################################################

# Clear the environment
rm(list = ls())

# Clear the plots
if(!is.null(dev.list())) dev.off()

# Clear the terminal (this is specific to RStudio)
cat("\014")

# Set the URL and file path
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
file_path <- "online_shoppers_intention.csv"

# Download the dataset if it doesn't exist in the working directory
if(!file.exists(file_path)) {
  GET(url, write_disk(file_path, overwrite = TRUE))
  message("Dataset downloaded successfully.")
} else {
  message("Dataset already exists in the working directory.")
}

# Load the dataset into R as a data frame
online_shoppers_purchasing_intention_dataset <- read_csv(file_path)

##################################################################################
# 4. Data Exploration
##################################################################################

# View the first few rows of the dataset
head(online_shoppers_purchasing_intention_dataset)

# Check for missing values in the dataset
missing_values <- colSums(is.na(online_shoppers_purchasing_intention_dataset))
print(missing_values)

# Check the structure of the dataset
str(online_shoppers_purchasing_intention_dataset)

# Summary statistics for numerical features
summary(online_shoppers_purchasing_intention_dataset)

##################################################################################
# 5. Visualising the Distribution of the Target Variable (Revenue)
##################################################################################

# Plot the distribution of Revenue
ggplot(online_shoppers_purchasing_intention_dataset, aes(x = Revenue)) +
  geom_bar(fill = 'steelblue', width = 0.5) +  # Adjust the width of the bars
  labs(title = "Distribution of Purchases (Revenue)",
       x = "Revenue (Purchase Made)", y = "Count") +
  theme(axis.text.x = element_text(size = 10, angle = 0, hjust = 1),  # Adjust x-axis text size
        plot.title = element_text(hjust = 0.5))  # Center the title

# Bar plot for VisitorType and Revenue
ggplot(online_shoppers_purchasing_intention_dataset, aes(x = VisitorType, fill = Revenue)) +
  geom_bar(position = "dodge") +
  labs(title = "Visitor Type vs. Purchase Decision",
       x = "Visitor Type", y = "Count", fill = "Revenue")

# Histogram for ProductRelated_Duration
ggplot(online_shoppers_purchasing_intention_dataset, aes(x = ProductRelated_Duration)) +
  geom_histogram(bins = 30, fill = 'tomato', color = 'black') +
  labs(title = "Distribution of Time Spent on Product Related Pages",
       x = "ProductRelated_Duration", y = "Frequency")

# Boxplot for BounceRates grouped by Revenue
ggplot(online_shoppers_purchasing_intention_dataset, aes(x = Revenue, y = BounceRates)) +
  geom_boxplot(fill = 'lightblue') +
  labs(title = "Bounce Rates by Purchase Decision",
       x = "Revenue (Purchase Made)", y = "Bounce Rates")

# Calculate correlation matrix for numerical features
correlation_matrix <- cor(online_shoppers_purchasing_intention_dataset[, sapply(online_shoppers_purchasing_intention_dataset, is.numeric)])

# Melt the matrix for plotting
melted_corr <- melt(correlation_matrix)

# Plot the heatmap
ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) + 
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + 
  labs(title = "Correlation Heatmap") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 9, hjust = 1))

##################################################################################
# 6. Calculating Correlation with Revenue
##################################################################################

# Select numerical columns for correlation
numerical_columns <- online_shoppers_purchasing_intention_dataset[, sapply(online_shoppers_purchasing_intention_dataset, is.numeric)]

# Add Revenue as a binary numeric column (1 for TRUE, 0 for FALSE)
numerical_columns$Revenue <- as.numeric(online_shoppers_purchasing_intention_dataset$Revenue)

# Calculate point-biserial correlation
correlation_with_revenue <- corr.test(numerical_columns, use="pairwise", method="pearson")
print(correlation_with_revenue)


##################################################################################
# 7. Feature selection
##################################################################################

# Feature selection: focusing on key features with moderate to strong correlations with Revenue
# Convert VisitorType to dummy variables (one-hot encoding)
online_shoppers_purchasing_intention_dataset$VisitorType <- as.factor(online_shoppers_purchasing_intention_dataset$VisitorType)
dummies <- dummyVars(" ~ VisitorType + Month", data = online_shoppers_purchasing_intention_dataset)
online_shoppers_purchasing_intention_dataset_transformed <- data.frame(predict(dummies, newdata = online_shoppers_purchasing_intention_dataset))

# Combine with the original dataset, keeping selected features
final_data <- cbind(online_shoppers_purchasing_intention_dataset[, c("PageValues", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "Revenue")],
                    online_shoppers_purchasing_intention_dataset_transformed)


##################################################################################
# 8. Data Splitting and Scaling
##################################################################################

# Set seed for reproducibility
set.seed(3350)

# Split the data
split <- sample.split(final_data$Revenue, SplitRatio = 0.8)
train_data <- subset(final_data, split == TRUE)
test_data <- subset(final_data, split == FALSE)

# Standardise the numerical features in the training and testing data
train_data_scaled <- as.data.frame(lapply(train_data[, c("PageValues", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates")], scale))
test_data_scaled <- as.data.frame(lapply(test_data[, c("PageValues", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates")], scale))

# Add the categorical variables and target (Revenue) back into the datasets
train_data <- cbind(train_data_scaled, train_data[, -c(1:5)])
test_data <- cbind(test_data_scaled, test_data[, -c(1:5)])

##################################################################################
# 9. Model 1: MLP Model
##################################################################################

# Ensure that Revenue is a factor for classification
train_data$Revenue <- as.factor(train_data$Revenue)
test_data$Revenue <- as.factor(test_data$Revenue)

# Train the MLP model for classification
mlp_model <- nnet(Revenue ~ ., data = train_data, size = 10, maxit = 200)

# Summary of the model
summary(mlp_model)

# Generate predictions using the MLP model
predictions <- predict(mlp_model, test_data, type = "class")

# Convert predictions and test_data$Revenue to factors with the same levels
predictions <- factor(predictions, levels = levels(test_data$Revenue))
test_data$Revenue <- factor(test_data$Revenue)

# Confusion matrix to evaluate the model
confusionMatrix(predictions, test_data$Revenue)


##################################################################################
# 10. Hyperparameter Tuning for MLP Model
##################################################################################

# Define the grid of hyperparameters (size and decay only)
tune_grid <- expand.grid(size = c(5, 10, 15),        # Number of neurons in the hidden layer
                         decay = c(0.01, 0.1, 0.5))  # Regularisation parameter to prevent overfitting

# Set up cross-validation control
train_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

# Train the MLP model with grid search
set.seed(3350)
mlp_tuned <- train(Revenue ~ ., 
                   data = train_data, 
                   method = "nnet", 
                   tuneGrid = tune_grid, 
                   trControl = train_control, 
                   trace = FALSE,  # Suppress training output
                   maxit = 200)    # Set maxit outside of tuneGrid

# Print the best model found by grid search
print(mlp_tuned$bestTune)

# Predict on the test data using the best tuned model
best_predictions <- predict(mlp_tuned, newdata = test_data)

# Confusion matrix to evaluate the tuned model
confusionMatrix(best_predictions, test_data$Revenue)



##################################################################################
# 11. Model 2: Random Forest Model
##################################################################################

# Train the Random Forest model
set.seed(3350)
rf_model <- randomForest(Revenue ~ ., data = train_data, ntree = 100, mtry = 3, importance = TRUE)

# Print the summary of the model
print(rf_model)

##################################################################################
# 12. Model Comparison and Feature Importance
##################################################################################

# Evaluate the Random Forest Model
rf_predictions <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_predictions, test_data$Revenue)

# View feature importance from Random Forest
importance(rf_model)
varImpPlot(rf_model)


# Calculate performance metrics for the MLP model
mlp_conf_matrix <- confusionMatrix(predictions, test_data$Revenue)
mlp_accuracy <- mlp_conf_matrix$overall['Accuracy']
mlp_precision <- mlp_conf_matrix$byClass['Pos Pred Value']
mlp_recall <- mlp_conf_matrix$byClass['Sensitivity']
mlp_f1 <- 2 * (mlp_precision * mlp_recall) / (mlp_precision + mlp_recall)

# Calculate performance metrics for the Random Forest model
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$Revenue)
rf_accuracy <- rf_conf_matrix$overall['Accuracy']
rf_precision <- rf_conf_matrix$byClass['Pos Pred Value']
rf_recall <- rf_conf_matrix$byClass['Sensitivity']
rf_f1 <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)

# Print model comparison results
model_comparison <- data.frame(
  Model = c("MLP", "Random Forest"),
  Accuracy = c(mlp_accuracy, rf_accuracy),
  Precision = c(mlp_precision, rf_precision),
  Recall = c(mlp_recall, rf_recall),
  F1_Score = c(mlp_f1, rf_f1)
)

print(model_comparison)

# Now, extract and visualise the importance of features based on the MLP model's weights
weights <- mlp_model$wts

# Find the number of input features (including bias)
n_input <- ncol(train_data) - 1  # Subtract 1 for the target variable

# Extract the weights connecting the input layer to the hidden layer (skip the output layer for now)
input_weights <- weights[1:(n_input * 10)]  # 10 is the number of hidden neurons used

# Calculate the importance of each input feature as the sum of absolute values of their weights
importance_scores <- apply(matrix(input_weights, nrow = n_input), 1, function(x) sum(abs(x)))

# Create a data frame for feature importance
mlp_importance <- data.frame(Feature = colnames(train_data)[1:n_input], Importance = importance_scores)

# Sort by importance and visualise
mlp_importance_sorted <- mlp_importance[order(-mlp_importance$Importance), ]

# Visualise the importance of the top 10 features
top_features <- head(mlp_importance_sorted, 10)
ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = 'blue') +
  labs(title = "MLP Feature Importance", x = "Features", y = "Importance") +
  coord_flip()



# Data Source:
# Sakar, C. and Kastro, Y. (2018) *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. Available at: https://doi.org/10.24432/C5F88Q.
