library(caret)
library(rstan)
library(bridgesampling)
library(dplyr)
library(ggplot2)
library(tidyr)
library(gridExtra)



Houses <- read.csv("AmesHousing.csv") %>% mutate(SalePrice = SalePrice/1000)

str(Houses)
summary(Houses)

New_Houses <- subset(Houses, Houses$Gr.Liv.Area <= 4000)

cat("Number of rows:", nrow(New_Houses), "\n")
cat("Number of columns:", ncol(New_Houses), "\n")

summary(New_Houses)

numeric_types <- c("integer", "numeric")



Numeric_Houses <- New_Houses %>%
  select_if(~ any(sapply(., function(x) class(x) %in% numeric_types)))

# Create a grid of subplots for plotting
nrows <- 8
ncols <- 5
plot_list <- list()

for (col in names(Numeric_Houses)) {
  p <- ggplot(Numeric_Houses, aes(x = !!sym(col))) +
    geom_density(fill = "red", alpha = 0.5) +
    labs(title = col) +
    theme_minimal()
  plot_list[[col]] <- p
}

# Arrange the plots in a grid
grid.arrange(grobs = plot_list, nrow = nrows, ncol = ncols)

# Convert categorical variables to numerical
one_hot_cols <- names(New_Houses)[sapply(New_Houses, is.character)]
New_Houses <- New_Houses[one_hot_cols, drop=FALSE]

head(New_Houses)

dummy_vars <- dummyVars(~., data = New_Houses[, one_hot_cols], fullRank = TRUE)

New_Houses <- data.frame(predict(dummy_vars, newdata = New_Houses))
summary(New_Houses)

float_cols <- names(Filter(is.numeric, New_Houses))

library(moments)

skew_limit <- .75
skew_vals <- sapply(New_Houses[float_cols], skewness)

head(skew_vals)



skew_cols <- skew_cols %>%
  filter(abs(Skew) > skew_limit) %>%
  arrange(desc(Skew))

# Apply log transformation to skewed columns
New_Houses[skew_cols$Column] <- log1p(New_Houses[skew_cols$Column])

# Create a bar plot of skewed columns
options(repr.plot.width = 20, repr.plot.height = 5)

p <- ggplot(data = skew_cols, aes(x = reorder(Column, Skew), y = Skew, fill = Skew)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
  labs(title = "\nSkew Columns (Log Transformed)\n", x = "Column", y = "Skew")

print(p)




