Advanced Regression Techniques for House Pricing
================
Anthony A. Boyles
September 19, 2016

-   [Assemble the Data!](#assemble-the-data)
-   [Clean the Data!](#clean-the-data)
    -   [Bad Names](#bad-names)
    -   [Feature Engineering](#feature-engineering)
    -   [Missing Values](#missing-values)
    -   [Outcome Transformation](#outcome-transformation)
    -   [Near-Zero Variance](#near-zero-variance)
    -   [Write the Data!](#write-the-data)
-   [Model the Data!](#model-the-data)
    -   [Linear Model](#linear-model)
    -   [ElasticNet](#elasticnet)
    -   [Cubist](#cubist)
    -   [Random Forest](#random-forest)
    -   [SVM](#svm)
    -   [Gradient Boosting](#gradient-boosting)
    -   [eXtreme Gradient Boosting](#extreme-gradient-boosting)
    -   [Make Some Predictions!](#make-some-predictions)

Note: while I haven't actually used any code from it, I owe a debt of gratitude to Stephanie Kirmer for [this Kernel](https://www.kaggle.com/skirmer/house-prices-advanced-regression-techniques/fun-with-real-estate-data), which was useful in guiding me through my own early data management and modeling.

``` r
library("readr")
library("plyr")
library("dplyr")
library("intubate")
library("ggplot2")
library("parallel")
library("doMC")
library("caret")
library("MASS")
library("glmnet")
library("glmnetUtils")
library("Cubist")
library("randomForest")
library("e1071")
library("nnet")
library("xgboost")
library("ShRoud")
library("magrittr")
```

Assemble the Data!
==================

``` r
training <- read_csv("rawdata/train.csv")
```

That was uneventful.

Clean the Data!
===============

Bad Names
---------

First things first! Some of these columns have names that start with numerals. That makes R ...itchy. Let's just fix that right quick:

``` r
training %<>%
  dplyr::rename(
    FirstFlrSF  = `1stFlrSF`,
    SecondFlrSF = `2ndFlrSF`,
    ThreeSsnPorch = `3SsnPorch`
  )
```

Note that this won't affect the models in any meaningful way.

Feature Engineering
-------------------

I could do my own, but [Thibaut95k](https://www.kaggle.com/thibaut95k/house-prices-advanced-regression-techniques/notebook44688d615f/run/370172) has already done such an awesome job.

``` r
training %<>%
  mutate(
    Age = YrSold - YearBuilt,
    AgeRemod = YrSold - YearRemodAdd,
    Baths = FullBath + HalfBath,
    BsmtBaths = BsmtFullBath + BsmtHalfBath,
    OverallQualSquare = OverallQual*OverallQual,
    OverallQualCube = OverallQual*OverallQual*OverallQual,
    OverallQualExp = expm1(OverallQual),
    GrLivAreaLog = log(GrLivArea),
    TotalBsmtSFGrLivArea = TotalBsmtSF/GrLivArea,
    OverallCondSqrt = sqrt(OverallCond),
    OverallCondSquare = OverallCond*OverallCond,
    LotAreaSqrt = sqrt(LotArea),
    FirstFlrSFSqrt = sqrt(FirstFlrSF),
    TotRmsAbvGrdSqrt = sqrt(TotRmsAbvGrd)
  )
```

Missing Values
--------------

There are a ton of them. They make the models fail. In a perfect world, we would analyze each column for its missingness and do multiple imputation to fill in the values we could reasonably impute, but I don't have the time or the patience for that. Instead, I'm just going to replace all missing values with the arithmetic mean of values in the that column for numeric columns, and "Unknown" for character columns.

``` r
training <- fixNAs(training)
```

Outcome Transformation
----------------------

If we take a look at the distribution of our outcome metric...

``` r
training %>%
  ggplot(aes(SalePrice)) +
  geom_histogram()
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](Readme_files/figure-markdown_github/unnamed-chunk-1-1.png)

You'll note that these values vary over several orders of magnitude (as [Alexandru Papieu pointed out](https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models)), so it may make more sense to predict the log-transformation of the data.

``` r
training %>%
  mutate(SalePrice = log1p(SalePrice)) %>%
  ggplot(aes(SalePrice)) +
  geom_histogram()
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](Readme_files/figure-markdown_github/unnamed-chunk-2-1.png)

It certainly apprears to be closer to normally-distributed (which is helpful). What about the predictors? Well, let's take a look at some correlations:

Note: I'll start cross-validating once I'm building models for prediction. These are just to give us a feel for whether or not a particular treatment (in this case, log-transformation) helps us.

``` r
training %>%
  ntbt_lm(SalePrice ~ .) %>%
  summary() %>%
  use_series(r.squared)
```

    ## [1] 0.9388063

``` r
training %>%
  mutate(SalePrice = log1p(SalePrice)) %>%
  ntbt_lm(SalePrice ~ .) %>%
  summary() %>%
  use_series(r.squared)
```

    ## [1] 0.9488686

So we get a tiny boost from log-transforming the outcome. Let's keep it.

``` r
training <- mutate(training, SalePrice = log1p(SalePrice))
```

Near-Zero Variance
------------------

The biggest problem I encountered in early modeling efforts was factors with values that occur infrequently in the data. Basically, what happens is we partition the data for cross-validation and there's a factor with one (or a few) especially rare value. All instances of that rare value land in the test data, so we have no way to assign a coefficient to it, and the model fails.

The simplest way to handle this is to drop any categorical variable with a value that is rarer than some tolerance threshold for model failure (basically, how patient you are). That's what I did for my first pass:

``` r
# Note that this code is not run:
training <- training %>% dplyr::select(-c(MSZoning, Street, Alley, LotShape, Utilities, LandSlope, Neighborhood, Condition1, Condition2, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual, ExterCond, Foundation, BsmtCond, BsmtFinType2, Heating, HeatingQC, Electrical, Functional, GarageType, GarageQual, GarageCond, PoolQC, Fence, MiscFeature, SaleType, SaleCondition, LotConfig))
```

A better way to solve this problem is to translate ordinal variables onto a continuous scale. A computer can't figure out how far "good" is from "poor," but it can definitely figure out the difference between 4 and 1. That works adequately for ordinal variables, but its throwing away some discernable signal, and it works less well for nominal variables. For example, there is no inherent ordinality in countertop materials, but the market values granite more highly than formica. This is particularly instructive: instead of assuming that an ordinal variable follows its order, let's actually take the mean price for each category and see whether it follows the implied ordering.

So, let's take a category we'd otherwise throw away, and figure out how to numberify it.

``` r
table(training$ExterCond)
```

    ## 
    ##   Ex   Fa   Gd   Po   TA 
    ##    3   28  146    1 1282

Perfect. With a 70-30 training-test partitioning, the probability that Po ("Poor") has no representatives in the training data is .3, which is totally unworkable. (The generalized formula for that metric, by the way, is *P*(Model Failure)=(1 − Training Proportion)<sup>size\\ of\\ smallest\\ category</sup>.) To fix it, let's look at the mean house price for each member of the class:

``` r
training %>%
  group_by(ExterCond) %>%
  summarise(AveragePrice = mean(SalePrice))
```

    ## # A tibble: 5 × 2
    ##   ExterCond AveragePrice
    ##       <chr>        <dbl>
    ## 1        Ex     12.11973
    ## 2        Fa     11.45517
    ## 3        Gd     11.96946
    ## 4        Po     11.24506
    ## 5        TA     12.04308

Here we can see that houses in Typical/Average shape on their exteriors actually fetch a slightly higher price, on average, than houses rated as being in "Good" shape. Cool! So, for every categorical variable that has a sufficiently high probability of causing a modeling failure, let's replace the categories with their average SalePrice.

``` r
failureThreshold <- 1e-6

transformedCategories <- list()

for(column in colnames(training)){
  if(is.character(training[[column]])){
    # This is not a sane way to do this, but I don't know any better way.
    temp <- eval(parse(text = paste0("group_by(training, ", column, ")"))) %>%
      summarise(AveragePrice = mean(SalePrice))
    replacements <- as.list(temp$AveragePrice)
    names(replacements) <- temp[[column]]
    training <- eval(parse(text = paste0("mutate(training, ", column," = as.numeric(replacements[", column,"]))")))
    transformedCategories[[column]] <- replacements
  }
}
```

OK, that's fun, but did it help us?

``` r
training %>%
  ntbt_lm(SalePrice ~ .) %>%
  summary() %>%
  use_series(r.squared)
```

    ## [1] 0.921737

Sadly, not, though it doesn't seem to hurt us much. More importantly, it resolves some modelling problems down the road, so Let's keep it anyway.

Write the Data!
---------------

Training is all set to go! Now we just need to give test the same treatment...

``` r
preparedtraining <- training

preparedtest <- read_csv("rawdata/test.csv") %>%
  dplyr::rename(
    FirstFlrSF  = `1stFlrSF`,
    SecondFlrSF = `2ndFlrSF`,
    ThreeSsnPorch = `3SsnPorch`
  ) %>% 
  mutate(
    Age = YrSold - YearBuilt,
    AgeRemod = YrSold - YearRemodAdd,
    Baths = FullBath + HalfBath,
    BsmtBaths = BsmtFullBath + BsmtHalfBath,
    OverallQualSquare = OverallQual*OverallQual,
    OverallQualCube = OverallQual*OverallQual*OverallQual,
    OverallQualExp = expm1(OverallQual),
    GrLivAreaLog = log(GrLivArea),
    TotalBsmtSFGrLivArea = TotalBsmtSF/GrLivArea,
    OverallCondSqrt = sqrt(OverallCond),
    OverallCondSquare = OverallCond*OverallCond,
    LotAreaSqrt = sqrt(LotArea),
    FirstFlrSFSqrt = sqrt(FirstFlrSF),
    TotRmsAbvGrdSqrt = sqrt(TotRmsAbvGrd)
  ) %>%
  fixNAs()

for(column in names(transformedCategories)){
  replacements <- transformedCategories[[column]]
  preparedtest <- eval(parse(text = paste0("mutate(preparedtest, ", column, " = replacements[preparedtest$", column, "][[1]])")))
}
```

And, we're done! On to...

Model the Data!
===============

Now, to make a preliminary preparation, let's partition the data into training and test sets so we can do some of our own scoring without having to submit new entries to Kaggle all the time.

``` r
temp <- training %>% mutate(train = runif(n()) < .7)
train <- temp %>% filter( train) %>% dplyr::select(-train)
test  <- temp %>% filter(!train) %>% dplyr::select(-train)
```

Also, I'm going to use Caret to fit the hyperparameters on models where that's useful, so I'm going to need a training controller for cross-validation.

``` r
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
registerDoMC(cores = detectCores() - 1)
```

Linear Model
------------

``` r
#modelLM <- lm(SalePrice ~ ., data=train)
modelLM <- train %>%
  ntbt_train(SalePrice ~ ., method = "lm", trControl = fitControl)
  
summary(modelLM)$r.squared
```

    ## [1] 0.9267868

Not bad for a first stab, but how well does it actually score?

``` r
modelLM %>%
  predict(test) %>%
  rmse(test$SalePrice)
```

    ## [1] 0.1385049

OK, so that's our first quality benchmark.

ElasticNet
----------

I thought about doing Ridge Regression or LASSO, but why do either when you can do both at once?

``` r
train %>%
  ntbt_train(SalePrice ~ ., method = "glmnet", trControl = fitControl) %>%
  predict(test) %>%
  rmse(test$SalePrice)
```

    ## [1] 0.1463396

Cubist
------

This one will burn through a few cycles, caveat emptor.

``` r
train %>%
  ntbt_train(SalePrice ~ ., method = "cubist", trControl = fitControl) %>%
  predict(test) %>%
  rmse(test$SalePrice)
```

    ## [1] 0.1391231

Random Forest
-------------

This one will burn through a few cycles, caveat emptor.

``` r
train %>%
  factorize() %>%
  ntbt_train(SalePrice ~ ., method = "rf", trControl = fitControl) %>%
  predict(test) %>%
  rmse(test$SalePrice)
```

    ## [1] 0.1345625

SVM
---

I actually started using Caret specifically to fit hyperparameters on SVMs.

``` r
train %>%
  ntbt_train(SalePrice ~ ., method = "svmLinear2", trControl = fitControl) %>%
  predict(test) %>%
  rmse(test$SalePrice)
```

Gradient Boosting
-----------------

``` r
train %>%
  ntbt_train(SalePrice ~ ., method = "gbm", trControl = fitControl, verbose = FALSE) %>%
  predict(test) %>%
  rmse(test$SalePrice)
```

    ## Loading required package: gbm

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loaded gbm 2.1.1

    ## [1] 0.1291257

eXtreme Gradient Boosting
-------------------------

This one actually runs so long, I've disabled it.

``` r
train %>%
  ntbt_train(SalePrice ~ ., method = "xgbLinear", trControl = fitControl) %>%
  predict(test) %>%
  rmse(test$SalePrice)
```

Make Some Predictions!
----------------------

Let's rerun it on the entire Kaggle training set, predict on the test set, write and submit it.

``` r
preparedtraining %>%
  ntbt_train(SalePrice ~ ., method = "glmnet", trControl = fitControl) %>%
  predict(preparedtest) %>%
  cbind(preparedtest, SalePrice = .) %>%
  dplyr::transmute(Id, SalePrice = expm1(SalePrice)) %>%
  write_csv("predictions/predictionCubist.csv")
```
