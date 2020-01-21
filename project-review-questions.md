## How did you pick the question(s) that you did?
After building the final, I wanted to use it to solve a "real" problem.  It seemed natural to ask how the model could be used to increase the sale price of a home from the seller's standpoint.

## Why are these questions important from a business perspective?
I think that's obvious.  The answers to the question at the end of my project provide a deterministic path and well defined strategy, step-by-step, on how to increase the sale price of his/her home and thereby potentially make the seller a lot of money.  It is also relevant to contractors since a key component of the strategy involves hiring a contractor to add liveable square footage.

## How did you decide on the data cleaning options you performed?
I relied heavily on Regression Diagnostics plots, as well as a linearity (with Price) study using scatter plots.

## Why did you choose a given method or library?
Both statsmodels and scikit learn were important libraries.  I used statsmodels primarily for OLS since it seems to provide many more statistics at a high level and it does so in a very straightforward manner.

Scikit Learn was invaluable but provides a lot more granularity.  I used it mainly for Cross-Validation and K-Folds.  I couldn't have built my Cross-Validation Feature Selection algorithm/routine without it.

## Why did you select those visualizations and what did you learn from each of them?
For me it was key to show incremental progress.  This meant that I had to a show a lot of visual comparisons to demonstrate improvement in a quick and easy way.  Also, Regression Diagnostics plots were a key part of the EDA phase.  Of course scatter plots and histrograms provided more granular views into EDA.

## Why did you pick those features as predictors?
Multicollinearity (or the reduction of it) was a key driving force toward incremental improvement of statistical reliability.  Incremental development sort of drove itself and ended picking the final feature set.  I just "listened" to the math and made adjustments as necessary to keep it moving in the direction of improvement in statistical reliability.  I made no assumptions up front about which predictors "should" be the best or most predictive.  I simply let math be the driving force and the final feature set is where it took me.

## How would you interpret the results?
I think the results speak for themselves.  The real "gold" in this exercise was using the model to answer concrete, very specific questions.  I didn't stop merely at building a final model.  I used it and the results were amazing!

## How confident are you in the predictive quality of the results?
I am very confident.  The key for me was getting a low measure of collinearity (Condition Number), paired with a high R-squared.

## What are some of the things that could cause the results to be wrong?
I would say that an invertible transformation is key.  Also, since I didn't spend nearly enough time on investigating outliers then predicting based on outliers could cause skewed results.  I would want to eventually give outliers the treatment they deserve to make the model air-tight.