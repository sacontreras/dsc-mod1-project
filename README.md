# Contents of this Repository

1. KingCountyHousingSales.ipynb
2. Step2.ipynb
3. CrossValidationFeatureSelection.ipynb
4. Appendix.ipynb
5. Data sets
   1. kc_house_data.csv
   2. kc_house_cleaned_data.csv
   3. kc_house_candidate_cv_selection_data.csv
   4. kc_cv_sel_optimized_features_scores_df.csv
6. Custom Python API/library (source code) written for this project

## KingCountyHousingSales.ipynb
This is the top level Jupyter notebook that constitutes "the project".  

Start with this notebook to view the project.

## Step2.ipynb
This notebook is referenced within the top level project notebook (KingCountyHousingSales.ipynb).

It carries out the steps to clean the data set.

## CrossValidationFeatureSelection.ipynb
After the steps to clean the data set (in Step2.ipynb) are complete, this notebook carries out the steps necessary for Cross Validation Selection of optimized feature subsets.

*Cross-validation* over 5 *k-folds* is used with a scoring method to select the model (built on training data) that produces the least RMSE and the difference between that RMSE vs. the RMSE computed on the testing data, *with Condition Number <= 100 (in order <b>to minimize colinearity</b>)*.  This basis is taken *directly* from statsmodels Github [source code](https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#RegressionResults.summary) for the OLS fit results `summary` method but I restrict it even further (statsmodels defines non-colinearity by virtue of this value being less than 1000).

See the Appendix for a more detailed explanation on how this works.

## Appendix.ipynb
This notebook houses the technical and theoritical explanation of how Cross Validation selection of optimal features (minimizing colinearity as well as RMSE works.

## Data sets
### kc_house_data.csv
This is the base data set from which we start.

### kc_house_cleaned_data.csv
This is the <i>output</i> of the notebook that carries out the steps to clean the data set (Step2.ipynb).

This data set is the basis upon which linear regression models are built.

### kc_house_candidate_cv_selection_data.csv
This is the <i>input</i> to the <i>Cross Validation selection of optimal features</i> notebook (CrossValidationFeatureSelection.ipynb).

It is based on the kc_house_cleaned_data.csv but contains a restricted feature set that has been preprocessed for <i>Cross Validation Feature Selection</i>.

### kc_cv_sel_optimized_features_scores_df.csv
This is <i>output</i> of the <i>Cross Validation selection of optimal features</i> notebook.

It houses a dataframe of "optimized" feature subsets used to build the final linear regression models, which are used to answer "real world questions", run experiments, and finally suggest a strategy to maximize market value when a hypothetical seller wants to put his home up for sale in King County.

### Custom Python API/library (source code) written for this project
This source is used throughout the project.  Rather than clutter the various notebooks with inline Python source code, I chose to place it in a separate library.  The effect is a cleaner presentation.

The files are located in:
1. scjpnlib/utils/\_\_init\_\_.py
2. scjpnlib/regression/api.py


# Happy "Linear Regressioning"!
Yep.  I turned that into a verb.

Enjoy.