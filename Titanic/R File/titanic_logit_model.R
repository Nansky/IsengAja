##############################################################################
# Predict Survivor from Titanic Accident using Logistic Regression Algorithm #
##############################################################################

library(dplyr)
library(Amelia)
library(ggplot2)

setwd("/Users/nanskyarts/Documents/NaN/Personal/Data Science (DS)/R/Projects/Online/Kaggle/Titanic/")

train <- read.csv('Dataset/titanic_train.csv')

missing <- missmap(obj = train, main = "Missing Value", legend = F, col = c('yellow', 'black'), y.at=1)

impute_age <- function(age,class){
    out <- age
    for (i in 1:length(age)){
        if(is.na(age[i])){
            if(class[i] == 1){
                out[i] <- 37
            }
            else if(class[i] == 2){
                out[i] <- 29
            }
            else{
                out[i] <- 24
            }
        }
        else{
            out[i] <- age[i]
        }
    }
    
    return(out)
}

impute_test_age <- function(age,class){
    out <- age
    for (i in 1:length(age)){
        if(is.na(age[i])){
            if(class[i] == 1){
                out[i] <- 43
            }
            else if(class[i] == 2){
                out[i] <- 26
            }
            else{
                out[i] <- 24
            }
        }
        else{
            out[i] <- age[i]
        }
    }
    
    return(out)
}

## Feature Selection ##
new_age_train <- impute_age(train$Age, train$Pclass)
train$Age <- new_age_train
new_train <- train

fin_train <- select(new_train, -PassengerId, -Name, -Cabin, -Ticket)

# Build Logit Model
logistic_regression <- glm(Survived ~ .,family = binomial(link = 'logit'), data= fin_train)
new_logit_model <- step(logistic_regression)

## Read Test_data
test <- read.csv('Dataset/titanic_test.csv')
any(is.na(test)) #TRUE

missing <- missmap(obj = test, main = "Missing Value", legend = F, col = c('yellow', 'black'), y.at=1)

new_age_test <- impute_test_age(test$Age, test$Pclass)
test$Age <- new_age_test

#missing <- missmap(obj = test, main = "Missing Value", legend = F, col = c('yellow', 'black'), y.at=1)

### Predict the model 
predicted <- predict(object = new_logit_model, newdata = test, type = 'response')
fitted.res <- ifelse(predicted > 0.5,1,0)
names(predicted) <- "Survived"

# read Gender Submission Data 
gender_sub <- read.csv('Dataset/gender_submission.csv')

# Count MissclassError
missclassEr <- mean(gender_sub$Survived != fitted.res)

# Generate Submission data 
submission_data <- as.data.frame(fitted.res,predicted)
submission_data <- cbind(gender_sub$PassengerId, submission_data)
colnames(submission_data) <- c("PassengerID", "Survived")


write.csv(submission_data, file = "Output/My Submission.csv", row.names = FALSE)

