###########################################################
# Data: Housing data used to predict price
# Project: Hierarchical model for predicting price  
# Author: Michael Issa
###########################################################

# Load libraries
library(caret)
library(rstan)
library(bridgesampling)
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggpubr)
library(MASS)
library(corrplot)
library(e1071)
library(glmnet)

###########################################################

###########################################################

# Load data
train <- read.csv("/Users/issam_biodcm6/Documents/R/Housing Prices Regression-Kaggle/data/house-prices-advanced-regression-techniques/train.csv")
test <- read.csv("/Users/issam_biodcm6/Documents/R/Housing Prices Regression-Kaggle/data/house-prices-advanced-regression-techniques/test.csv")

# Check the number of samples and features before dropping features
cat("The train data size before dropping Id feature is : ", nrow(train), "x", ncol(train), "\n")
cat("The test data size before dropping Id feature is : ", nrow(test), "x", ncol(test), "\n")

# Save the 'Id' column
train_ID <- train$Id
test_ID <- test$Id

# Now drop the 'Id' column since it's unnecessary for the prediction process.
train <- train[, !(names(train) %in% c("Id"))]
test <- test[, !(names(test) %in% c("Id"))]

# Check the data size again after dropping the 'Id' variable
cat("\nThe train data size after dropping Id feature is : ", nrow(train), "x", ncol(train), "\n")
cat("The test data size after dropping Id feature is : ", nrow(test), "x", ncol(test), "\n")

# Create a scatter plot using ggplot
ggplot(train, aes(x = GrLivArea, y = SalePrice)) +
  geom_point() +
  labs(x = "GrLivArea", y = "SalePrice", title = "Scatter Plot")

# Deleting outliers
train <- train[!(train$GrLivArea > 4000 & train$SalePrice < 300000), ]

# Create a scatter plot after removing outliers
ggplot(train, aes(x = GrLivArea, y = SalePrice)) +
  geom_point() +
  labs(x = "GrLivArea", y = "SalePrice", title = "Scatter Plot")

# Distribution plot
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(aes(y = ..density..), binwidth = 5000, fill = "lightblue", color = "black") +
  geom_density(alpha = 0.2, fill = "#FF6666") +
  labs(title = "SalePrice Distribution", x = "SalePrice", y = "Density") +
  theme_minimal()

# Fitted parameters
fit <- fitdistr(train$SalePrice, "normal")
mu <- fit$estimate[1]
sigma <- fit$estimate[2]
cat("\n mu = ", round(mu, 2), " and sigma = ", round(sigma, 2), "\n")

# QQ-plot
qqnorm(train$SalePrice)
qqline(train$SalePrice, col = 2)

###########################################################
# Apply log scale and plot quantiles
###########################################################


# Apply log(1+x) to the 'SalePrice' column
train$SalePrice <- log1p(train$SalePrice)

# Distribution plot
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(binwidth = 0.1, fill = "lightblue", color = "black") +
  geom_density(alpha = 0.2, fill = "red") +
  labs(title = "SalePrice Distribution", x = "Log(1 + SalePrice)", y = "Frequency") +
  theme_minimal()

# Get the fitted parameters
fit <- fitdistr(train$SalePrice, "normal")
mu <- fit$estimate[1]
sigma <- fit$estimate[2]
cat("\n mu = ", round(mu, 2), " and sigma = ", round(sigma, 2), "\n")

# Plot the distribution
ggplot(train, aes(x = SalePrice)) +
  geom_density(aes(y = ..density..), fill = "red", alpha = 0.2) +
  stat_function(fun = dnorm, args = list(mean = mu, sd = sigma), color = "blue", size = 1) +
  labs(title = "SalePrice Distribution",
       x = "Log(1 + SalePrice)",
       y = "Density") +
  theme_minimal()

# QQ-plot
qqplot_data <- data.frame(Theoretical = qnorm(ppoints(length(train$SalePrice))), Observed = sort(train$SalePrice))
ggplot(qqplot_data, aes(x = Theoretical, y = Observed)) +
  geom_point() +
  geom_abline(intercept = mu, slope = sigma, color = "red", linetype = "dashed") +
  labs(title = "QQ-Plot", x = "Theoretical Quantiles", y = "Observed Quantiles") +
  theme_minimal()


###########################################################
# Feature Engineering
###########################################################

# Get the number of rows in the train and test datasets
ntrain <- nrow(train)
ntest <- nrow(test)

# Extract the 'SalePrice' column from the train dataset
y_train <- train$SalePrice

# Concatenate the train and test datasets
all_data <- bind_rows(train, test)

# Reset the row indices
row.names(all_data) <- NULL

# Drop the 'SalePrice' column
all_data <- subset(all_data, select = -SalePrice)

# Print the size of all_data
cat("all_data size is: ", dim(all_data), "\n") # By rows, columns 

# Calculate the percentage of missing values for each column
all_data_na <- (colSums(is.na(all_data)) / nrow(all_data)) * 100

# Remove columns with no missing values
all_data_na <- all_data_na[all_data_na > 0]

# Sort missing values in descending order and select the top 30
all_data_na <- all_data_na[order(-all_data_na)][1:30]

# Create a data frame to display missing ratios
missing_data <- data.frame('Missing Ratio' = all_data_na)

# Print the first 20 rows of the missing_data data frame
print(head(missing_data, 20))

# Create a bar plot
p <- ggplot(data = NULL, aes(x = names(all_data_na), y = all_data_na)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Features", y = "Percent of missing values", title = "Percent missing data by feature", fontsize = 15)

# Print the plot
print(p)

###########################################################
# Fix missing values
###########################################################

# Replace missing values in each variable
all_data$PoolQC <- ifelse(is.na(all_data$PoolQC), "None", all_data$PoolQC)
all_data$MiscFeature <- ifelse(is.na(all_data$MiscFeature), "None", all_data$MiscFeature)
all_data$Alley <- ifelse(is.na(all_data$Alley), "None", all_data$Alley)
all_data$Fence <- ifelse(is.na(all_data$Fence), "None", all_data$Fence)
all_data$FireplaceQu <- ifelse(is.na(all_data$FireplaceQu), "None", all_data$FireplaceQu)

all_data$LotFrontage <- ifelse(is.na(all_data$LotFrontage), median(all_data$LotFrontage, na.rm = TRUE), all_data$LotFrontage)


for (col in c('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond')) {
  all_data[, col] <- ifelse(is.na(all_data[, col]), 'None', all_data[, col])
}

for (col in c('GarageYrBlt', 'GarageArea', 'GarageCars')) {
  all_data[, col] <- ifelse(is.na(all_data[, col]), 0, all_data[, col])
}


for (col in c('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath')) {
  all_data[, col] <- ifelse(is.na(all_data[, col]), 0, all_data[, col])
}

for (col in c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')) {
  all_data[, col] <- ifelse(is.na(all_data[, col]), 'None', all_data[, col])
}


all_data$MasVnrType <- ifelse(is.na(all_data$MasVnrType), "None", all_data$MasVnrType)
all_data$MasVnrArea <- ifelse(is.na(all_data$MasVnrArea), 0, all_data$MasVnrArea)

all_data$MSZoning <- ifelse(is.na(all_data$MSZoning), mode(all_data$MSZoning), all_data$MSZoning)

# Remove redundant variables 
all_data <- all_data[, !colnames(all_data) %in% 'Utilities']

all_data$Functional <- ifelse(is.na(all_data$Functional), "Typ", all_data$Functional)
all_data$Electrical <- ifelse(is.na(all_data$Electrical), mode(all_data$Electrical), all_data$Electrical)
all_data$KitchenQual <- ifelse(is.na(all_data$KitchenQual), mode(all_data$KitchenQual), all_data$KitchenQual)

all_data$Exterior1st <- ifelse(is.na(all_data$Exterior1st), mode(all_data$Exterior1st), all_data$Exterior1st)
all_data$Exterior2nd <- ifelse(is.na(all_data$Exterior2nd), mode(all_data$Exterior2nd), all_data$Exterior2nd)
all_data$SaleType <- ifelse(is.na(all_data$SaleType), mode(all_data$SaleType), all_data$SaleType)
all_data$MSSubClass <- ifelse(is.na(all_data$MSSubClass), 'None', all_data$MSSubClass)

# Calculate the percentage of missing values for each column
all_data_na <- (colSums(is.na(all_data)) / nrow(all_data)) * 100
all_data_na <- all_data_na[all_data_na > 0]
all_data_na <- all_data_na[order(-all_data_na)]
missing_data <- data.frame('Missing Ratio' = all_data_na)
head(missing_data)

###########################################################
# Modify numerical variables to categorical
###########################################################

# MSSubClass = The building class
all_data$MSSubClass <- as.character(all_data$MSSubClass)

# Changing OverallCond into a categorical variable
all_data$OverallCond <- as.character(all_data$OverallCond)

# Year and month sold are transformed into categorical features.
all_data$YrSold <- as.character(all_data$YrSold)
all_data$MoSold <- as.character(all_data$MoSold)

# Define the columns to be encoded
cols <- c('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
          'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
          'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
          'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
          'YrSold', 'MoSold')

# Process columns, apply factor encoding to categorical features
for (c in cols) {
  all_data[[c]] <- as.numeric(factor(all_data[[c]], levels = unique(all_data[[c]])))
}

# Print the shape of all_data
cat('Shape all_data:', dim(all_data), '\n')


###########################################################
# Add mixtures of variables
###########################################################

# Adding total sqfootage feature
all_data$TotalSF <- all_data$TotalBsmtSF + all_data$X1stFlrSF + all_data$X2ndFlrSF

# Find numeric features
numeric_feats <- names(all_data)[sapply(all_data, is.numeric)]

# Check the skew of all numerical features using the MASS package
skewed_feats <- sapply(all_data[numeric_feats], function(x) skewness(x, na.rm = TRUE)) %>%
  sort(decreasing = TRUE)

# Print skewness in numerical features
cat("\nSkew in numerical features: \n")
skewness <- data.frame(Skew = skewed_feats)
print(head(skewness, 10))

# Identify skewed numerical features
skewness <- skewness[abs(skewness) > 0.75]
print(sprintf("There are %d skewed numerical features to Box Cox transform\n", nrow(skewness)))

# Apply Box-Cox transformation to skewed features
skewed_features <- rownames(skewness)
lam <- 0.15

for (feat in skewed_features) {
  all_data[[feat]] <- boxcox1p(all_data[[feat]], lam)
}

###########################################################
# Categorical Encoding: Create Dummy Variables with matrix
###########################################################

# Identify categorical columns
categorical_cols <- sapply(all_data, is.character)

# Create dummy variables using model.matrix
dummy_matrix <- model.matrix(~ . - 1, data = all_data[, categorical_cols])

# Convert the dummy matrix to a data frame
dummy_df <- as.data.frame(dummy_matrix)

# Add dummy variables to the original dataset
all_data <- cbind(all_data, dummy_df)


# Print the shape of all_data after creating dummy variables
cat('Shape all_data after creating dummy variables:', dim(all_data), '\n')

# Final train and test data
train <- all_data[1:ntrain, ]
test <- all_data[(ntrain + 1):nrow(all_data), ]

###########################################################
# Model building and testing
###########################################################
# Set the number of folds
n_folds <- 5

# Define the RMSLE function 
rmsle_cv <- function(model, train_data, response) {
  # Create a train control object for cross-validation
  ctrl <- trainControl(method = "cv", number = n_folds, classProbs = TRUE)
  
  # Use the train function from caret for cross-validation
  rmse <- train(x = train_data, y = response,
                method = model,
                trControl = ctrl,
                metric = "RMSE")
  
  # Extract the root mean squared error from the results
  return(sqrt(rmse$results$RMSE))
}

# Example usage
rf_result <- rmsle_cv(model = "rf", train_data = train, response = y_train)
cat("\nRF RMSE: ", sprintf("%.4f", mean(rf_result)), " (", sprintf("%.4f", sd(rf_result)), ")\n")


