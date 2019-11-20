import pandas as pd
from IPython.core.display import HTML
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.graphics.utils as smgrpu
from matplotlib.gridspec import GridSpec
import statsmodels.tools.tools as smtt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.graphics.regressionplots as smgrp
from ..utils import *
from itertools import combinations
from sklearn.model_selection import cross_val_score

plot_edge = 4

def histograms(df):
    feats = df.columns
    
    s_html = "<h3>Distributions:</h3><ul>"
    s_html += "<li><b>feature set</b>: {}</li>".format(feats)
    s_html += "</ul>"
    display(HTML(s_html))
    
    n_feats = len(feats)
    
    r_w = 4*plot_edge if n_feats > plot_edge else (n_feats*4 if n_feats > 1 else plot_edge)
    r_h = plot_edge if n_feats > 4 else (plot_edge if n_feats > 1 else plot_edge)
    
    c_n = 4 if n_feats > 4 else n_feats
    r_n = n_feats/c_n
    r_n = int(r_n) + (1 if n_feats > 4 and r_n % int(r_n) != 0 else 0)        

    fig = plt.figure(figsize=(r_w, r_h*r_n))

    axes = fig.subplots(r_n, c_n)
    unused = list(range(0, len(axes.flatten()))) if n_feats > 1 else [0]

    included_feats = []
    for index, feat in enumerate(feats):
        ax = fig.add_subplot(r_n, c_n, index+1)
        sns.distplot(df[feat], ax=ax)
        plt.xlabel(feat)
        included_feats.append(feat)
        unused.remove(index)

    flattened_axes = axes.flatten() if n_feats > 1 else [axes]
    for u in unused:
        fig.delaxes(flattened_axes[u])

    fig.tight_layout()
    plt.show();

    return included_feats

def histograms_comparison(feature_series, xlabels, titles):    
    s_html = "<h3>Distributions:</h3><ul>"
    s_html += "</ul>"
    display(HTML(s_html))
    
    n_feats = len(feature_series)
    
    r_w = 4*plot_edge if n_feats > plot_edge else (n_feats*4 if n_feats > 1 else plot_edge)
    r_h = plot_edge if n_feats > 4 else (plot_edge if n_feats > 1 else plot_edge)
    
    c_n = 4 if n_feats > 4 else n_feats
    r_n = n_feats/c_n
    r_n = int(r_n) + (1 if n_feats > 4 and r_n % int(r_n) != 0 else 0)        

    fig = plt.figure(figsize=(r_w, r_h*r_n))

    axes = fig.subplots(r_n, c_n)
    unused = list(range(0, len(axes.flatten()))) if n_feats > 1 else [0]

    for index, fs in enumerate(feature_series):
        ax = fig.add_subplot(r_n, c_n, index+1)
        sns.distplot(fs, ax=ax)
        plt.xlabel(xlabels[index])
        plt.title(titles[index])
        unused.remove(index)

    flattened_axes = axes.flatten() if n_feats > 1 else [axes]
    for u in unused:
        fig.delaxes(flattened_axes[u])

    #fig.tight_layout()
    plt.show();

def plot_corr(df, filter=None):
    corr = df[filter].corr() if (filter is not None and len(filter) > 0) else df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 18))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr
        , mask=mask
        , cmap=cmap
        , vmax=1.0
        , center=0
        , square=True
        , linewidths=.5
        , cbar_kws={"shrink": .5}
    )
    plt.title("Feature Correlations")
    plt.show();

def feature_regression_summary(
    df
    , feat_idx
    , target
    , model_fit_results
    , display_regress_diagnostics=False):

    feat = df.columns[feat_idx]
    
    v = vif(np.matrix(df), feat_idx)
    colinear = v > 10
    
    if display_regress_diagnostics:   
        # ‘endog versus exog’, ‘residuals versus exog’, ‘fitted versus exog’ and ‘fitted plus residual versus exog’
        fig = plt.figure(constrained_layout=True, figsize=(2.25*plot_edge,4*plot_edge))
        fig = smgrpu.create_mpl_fig(fig)
        gs = GridSpec(4, 2, figure=fig)
        ax_fit = fig.add_subplot(gs[0, 0])
        ax_partial_residuals = fig.add_subplot(gs[0, 1])
        ax_partregress = fig.add_subplot(gs[1, 0])
        ax_ccpr = fig.add_subplot(gs[1, 1])
        ax_dist = fig.add_subplot(gs[2:4, 0:2])
        
        exog_name, exog_idx = smgrpu.maybe_name_or_idx(feat, model_fit_results.model)
        smresults = smtt.maybe_unwrap_results(model_fit_results)
        y_name = smresults.model.endog_names
        x1 = smresults.model.exog[:, exog_idx]
        prstd, iv_l, iv_u = wls_prediction_std(smresults)
        
        # endog versus exog
        # use wrapper since it's availab;e!
        sm.graphics.plot_fit(model_fit_results, feat, ax=ax_fit)
        
        # residuals versus exog
        ax_partial_residuals.plot(x1, smresults.resid, 'o')
        ax_partial_residuals.axhline(y=0, color='black')
        ax_partial_residuals.set_title('Residuals versus %s' % exog_name, fontsize='large')
        ax_partial_residuals.set_xlabel(exog_name)
        ax_partial_residuals.set_ylabel("resid")
        
        # Partial Regression plot: fitted versus exog
        exog_noti = np.ones(smresults.model.exog.shape[1], bool)
        exog_noti[exog_idx] = False
        exog_others = smresults.model.exog[:, exog_noti]
        from pandas import Series
        smgrp.plot_partregress(
            smresults.model.data.orig_endog
            , Series(
                x1
                , name=exog_name
                , index=smresults.model.data.row_labels
            )
            , exog_others
            , obs_labels=False
            , ax=ax_partregress
        )        
        
        # CCPR: fitted plus residual versus exog
        # use wrapper since it's availab;e!
        sm.graphics.plot_ccpr(model_fit_results, feat, ax=ax_ccpr)
        ax_ccpr.set_title('CCPR Plot', fontsize='large')
        
        sns.distplot(df[feat], ax=ax_dist)
        
        fig.suptitle('Regression Plots for %s' % exog_name, fontsize="large")
        #fig.tight_layout()
        #fig.subplots_adjust(top=.90)        
        plt.show()  
        
        display(
            HTML(
                "Variance Inflation Factor (<i>VIF</i>) for <b>{}</b>: <b>{}</b> ({})<br><br>".format(
                    feat
                    , round(v, 2)
                    , "<b>GOOD</b>, low colinearity $\\iff VIF \\le 10$" if not colinear else "BAD, <b>HIGH COLINEARITY</b> $\\iff VIF>10$<br><br>"
                )
            )
        )
    
    return v

def model_fit_summary(
    df
    , sel_features
    , their_pvals
    , target
    , model
    , tr
    , train_mse
    , test_mse
    , mv_r_sq_th
    , mv_mse_delta_th
    , mv_bad_vif_ratio_th
    , display_regress_diagnostics=False):

    # get results of OLS fit from previously computed model
    model_fit_results = model.fit()
    
    delta_mse = abs(test_mse - train_mse)
    valid_r_sq = model_fit_results.rsquared >= mv_r_sq_th

    # for each feature, compute VIF and optionally display QQ plot
    display(HTML("<h3>Regression Diagnostics</h3>"))
    good_vifs = []
    bad_vifs = []
    for idx, feat in enumerate(sel_features):        
        v = feature_regression_summary(df[sel_features], idx, model_fit_results, display_regress_diagnostics)
        if v <= 10:
            good_vifs.append((feat, round(v, 2), their_pvals[sel_features.index(feat)]))
        else:
            bad_vifs.append((feat, round(v, 2), their_pvals[sel_features.index(feat)]))    
    # order good VIF features by p-value
    good_vifs = sorted(good_vifs, key=lambda good_vif: good_vif[2])    
    # order bad VIF features by VIF
    bad_vifs = sorted(bad_vifs, key=lambda bad_vif: bad_vif[1], reverse=True)
    good_vif_ratio = len(good_vifs)/len(sel_features)
    bad_vif_ratio = 1 - good_vif_ratio
    
    # VIF/colinearity summary
    display(HTML("<h3>VIF Summary</h3>"))
    s_html = "<br><b>'GOOD' FEATURES</b> (<i>with LOW COLINEARITY, VIF <= 10</i>), ordered by favorable (increasing) p-val:<ol>"
    for good_vif in good_vifs:
        s_html += "<li><b>{:30}</b>: p-val $= {}$</li>".format(good_vif[0], good_vif[2])
    s_html += "</ol>"
    display(HTML(s_html))    
    s_html = "<br><b>'BAD' FEATURES</b> (<i>with HIGH COLINEARITY, VIF > 10</i>), ordered by unfavorable (decreasing) VIF:<ol>"
    for bad_vif in bad_vifs:
        s_html += "<li><b>{:30}</b>: VIF $= {}$</li>".format(bad_vif[0], bad_vif[1])
    s_html += "</ol>"
    display(HTML(s_html)) 
    good_vif_ratio = len(good_vifs)/len(sel_features)
    display(HTML("<br><b>{}% of features are 'GOOD'.<br>".format(round(good_vif_ratio*100,2))))
    plot_corr(df)
    
    # always displays QQ plot of residuals and density plots of target 
    fig = plt.figure(figsize=(10, 5))
    axes = fig.subplots(1, 2)
    sm.graphics.qqplot(model_fit_results.resid, dist=stats.norm, line='45', fit=True, ax=axes[0])
    sns.distplot(df[target], ax=axes[1])   
    fig.tight_layout()
    plt.show();
    
    # display the OLS summary
    display(HTML(model_fit_results.summary().as_html()))
    
    s_html = "<h3>Model Validation Summary</h3><ol>"
    s_html += "<li><b>$R^2={} \ge $</b> acceptable threshold ({}): <b>{}</b></li>".format(
        model_fit_results.rsquared
        , mv_r_sq_th
        , "PASS" if valid_r_sq else "FAIL"
    )    
    valid_delta_mse = delta_mse <= mv_mse_delta_th
    s_html += "<li><b>$\Delta MSE=|{}-{}|={} \le $</b> acceptable threshold ({}): <b>{}</b></li>".format(
        test_mse
        , train_mse
        , delta_mse
        , mv_mse_delta_th
        , "PASS" if valid_delta_mse else "FAIL"
    )    
    valid_bad_vif_ratio = mv_bad_vif_ratio_th > bad_vif_ratio
    s_html += "<li><b>bad VIF ratio = ${}$%</b> < acceptable threshold ({}%): <b>{}</b></li>".format(
        round(bad_vif_ratio*100,2)
        , round(mv_bad_vif_ratio_th*100,2)
        , "PASS" if valid_bad_vif_ratio else "FAIL"
    )
    s_html += "</ol>"
    s_html += "<b>Model Validation Assessment: {}</b>".format("PASS" if (valid_r_sq and valid_delta_mse and valid_bad_vif_ratio) else "FAIL")
    display(HTML(s_html))
    
    return (model_fit_results, good_vifs, bad_vifs)

def skl_lin_reg_validation(X, y, tr, verbose=False):
    linreg = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tr, random_state=42)
    linreg.fit(X_train, y_train)
    y_hat_train = linreg.predict(X_train)
    y_hat_test = linreg.predict(X_test)
    train_residuals = y_hat_train - y_train
    test_residuals = y_hat_test - y_test
    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)

    if verbose:
        s_html = "<table><table><thead><thead><tr><td><b>train/test-split ratio</b>: <b>{}/{}</b></td></tr></thead><tbody>".format(1-tr, tr)
        s_html += "<tr><td><table><tbody><tr><td><table><tbody><tr><td>|X_train| = {}</td><td>|X_test| = {}</td><td>|y_train| = {}</td><td>|y_test| = {}</td></tr></tbody></table></td></tr>".format(len(X_train), len(X_test), len(y_train), len(y_test))
        s_html += "<tr><td><table><tbody><tr><td>Train MSE:</td><td>{}</td></tr></tbody></table></td></tr>".format(train_mse)
        s_html += "<tr><td><table><tbody><tr><td>Test MSE:</td><td>{}</td></tr></tbody></table></td></tr>".format(test_mse)
        s_html += "<tr><td><table><tbody><tr><td><b>delta</b>:</td><td><b>{}</b></td></tr></tbody></table></td></tr>".format(abs(train_mse - test_mse))        
        s_html += "</tbody></table></table>"
        display(HTML(s_html))

    return (X_train, X_test, y_train, y_test, train_mse, test_mse, linreg)

def find_best_train_test_split(X, y, step_size = 5, verbose=False):
    delta_min = -1
    best_tr = 0
    X_train_best = None
    X_test_best = None
    y_train__best = None
    y_test_best = None
    train_mse_best = None
    test_mse_best = None
    linreg_best = None

    for test_ratio in range(step_size, 100, step_size):
        tr = test_ratio/100
        (X_train, X_test, y_train, y_test, train_mse, test_mse, linreg) = skl_lin_reg_validation(X, y, tr, verbose)
        delta = abs(train_mse - test_mse)
        if delta_min == -1 or delta < delta_min:
            delta_min = delta
            best_tr = tr
            X_train_best = X_train
            X_test_best = X_test
            y_train_best = y_train
            y_test_best = y_test
            train_mse_best = train_mse
            test_mse_best = test_mse
            linreg_best = linreg

    s_html = "<table><table><thead><thead><tr><td>best train/test-split ratio: {}/{}</td></tr></thead><tbody>".format(1-best_tr, best_tr)
    s_html += "<tr><td><table><tbody><tr><td><table><tbody><tr><td>|X_train| = {}</td><td>|X_test| = {}</td><td>|y_train| = {}</td><td>|y_test| = {}</td></tr></tbody></table></td></tr>".format(len(X_train_best), len(X_test_best), len(y_train_best), len(y_test_best))
    s_html += "<tr><td><table><tbody><tr><td>Train MSE:</td><td>{}</td></tr></tbody></table></td></tr>".format(train_mse_best)
    s_html += "<tr><td><table><tbody><tr><td>Test MSE:</td><td>{}</td></tr></tbody></table></td></tr>".format(test_mse_best)
    s_html += "<tr><td><table><tbody><tr><td>delta:</td><td>{}</td></tr></tbody></table></td></tr>".format(delta_min)        
    s_html += "</tbody></table></table>"
    display(HTML(s_html))

    return (X_train_best, X_test_best, y_train_best, y_test_best, train_mse_best, test_mse_best, linreg_best)

def stepwise_selection(
    X
    , y
    , initial_list=[]
    , 
    threshold_in=0.01
    , threshold_out = 0.05
    , verbose=True):

    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """

    starting_features = list(X.columns)

    included = list(initial_list)
    included_pvals = []
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            included_pvals.append(best_pval)
            changed=True
            if verbose:
                print("stepwise_selection: Add  {:30} with p-value {:.6}".format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            #worst_feature = pvalues.argmax()
            worst_feature = pvalues.idxmax()
            del included_pvals[included.index(worst_feature)]
            included.remove(worst_feature)
            if verbose:
                print("stepwise_selection: Drop {:30} with p-value {:.6}".format(worst_feature, worst_pval))
        if not changed:
            break

    dropped = list(set(starting_features) - set(included))
            
    print("\nstepwise_selection: starting features:\n{}".format(starting_features))
    print("\nstepwise_selection: selected features:\n{}".format(included))
    print("\nstepwise_selection: dropped statistically insignificant features:\n{}".format(dropped))
        
    return (included, included_pvals)

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

def cv_build_feature_combinations(X, upper_bound=2**18, boundary_test=False):
    feat_combos = dict()
    
    r = range(1, len(X.columns)+1)
        
    n = max(r)
    
    # determine whether or not we will exhause memory!
    len_total_combos = 0
    
    s_n_choose_k = "{} \\choose {}"
    if not boundary_test:
        s_total_combos = "<br><br>Builing all $\\sum_{i=" + str(min(r)) + "}^{" + str(n) + "}{" + s_n_choose_k.format(n, "i") + "}=$"
        for k in r:
            s_total_combos += " ${" + s_n_choose_k.format(n, k) + "}$" + (" +" if k!=n else "")
            len_combos = nCr(n, k)
            len_total_combos += len_combos
        s_total_combos += " $={}$ combinations of feature set: {}...".format(len_total_combos, X.columns)
        display(HTML(s_total_combos))
    else:
        len_total_combos = 2**n - 1
    
    if len_total_combos > upper_bound:
        display(HTML("<h2><font color=\"red\">Building all combinations of {} features would result in cross-validating {} models which will most likely exhaust memory (exceeds upper bound: {})!  Please reduce the size of the feature set and try again!</font></h2>".format(n, len_total_combos, upper_bound)))
        feat_combos = None
    else:
        if not boundary_test:
            for k in r:
                feat_combos_of_length_k = list(combinations(X, k))
                feat_combos[k] = feat_combos_of_length_k
        else:
            display(HTML("<h2><font color=\"green\">Boundary test PASSED!  Building all combinations of {} features will result in cross-validating {} models, which is less than upper bound: {}.</font></h2>".format(n, len_total_combos, upper_bound)))
    
    display(HTML("All done!"))    
    
    return (feat_combos, len_total_combos)

def cv_selection(X, y, tr, folds, scoring_method, fn_better_score):
    cv_feat_combo_map, len_total_combos = cv_build_feature_combinations(X)
    if cv_feat_combo_map is None:
        return
    
    base_feature_set = list(X.columns)
    n = len(base_feature_set)
    
    linreg = LinearRegression()
    
    best_score = 0
    best_feat_combo = None
    for n_feats, list_of_feat_combos in cv_feat_combo_map.items():
        n_choose_k = len(list_of_feat_combos)
        k = len(list_of_feat_combos[0])
        s_n_choose_k = "{} \\choose {}"
        display(HTML("Cross-validating ${}={}$ combinations of {} features (out of {}) over {} folds...".format("{" + s_n_choose_k.format(n, k) + "}", n_choose_k, k, n, folds)))
        
        for feat_combo in list_of_feat_combos:           
            feat_combo = list(feat_combo)
            
            # get neg_mean_squared score of all folds and theb compute the mean
            mean_cv_score = np.mean(
                cross_val_score(
                    linreg
                    , X[feat_combo]
                    , y
                    , cv=folds
                    , scoring=scoring_method
                )
            )
            #print("\tscore: {}".format(mean_cv_scor))
            #print()
            if best_score == 0 or fn_better_score(mean_cv_score, best_score):
                best_score = mean_cv_score
                best_feat_combo = feat_combo
                print("new best {} score: {}, from feature-set combo: {}".format(scoring_method, best_score, best_feat_combo))
    
    display(HTML("<h4>cv_selected best {} = {}</h4>".format(scoring_method, best_score)))
    display(HTML("<h4>cv_selected best feature-set combo ({} of {} features):{}<h/4>".format(len(best_feat_combo), len(base_feature_set), best_feat_combo)))
    display(HTML("<h4>starting feature-set:{}</h4>".format(base_feature_set)))
    to_drop = list(set(base_feature_set).difference(set(best_feat_combo)))
    display(HTML("<h4>cv_selection suggests dropping {}.</h4>".format(to_drop if len(to_drop)>0 else "<i>no features</i> from {}".format(base_feature_set))))
    
    return (best_feat_combo, best_score, to_drop)

def lin_reg_model_from_auto_selected_features(
    df
    , target
    , fn_feature_selection=stepwise_selection
    , tr = None
    , title="Linear Regression Model"):

    s_html = "<h1>{}</h1><ul>".format(title)
    s_html += "<li><b>target</b>: {}</li>".format(target)
    s_html += "<li><b>starting feature set</b>: {}</li>".format(df.drop(target, axis=1).columns)
    s_html += "<li><b>training/test split ratio</b>: ${}/{}$</li>".format(1-tr, tr)
    s_html += "</ul>"
    display(HTML(s_html))

    print("Feature selection method: {}".format(fn_feature_selection))
    
    data_fin_df = df.copy()
    
    y = data_fin_df[target]
    X = data_fin_df.drop([target], axis=1)
    
    sel_features, their_pvals = fn_feature_selection(X, y)
    f = target + '~' + "+".join(sel_features)
    print("\nformula: {}".format(f))
    
    if tr is None:
        X_train, X_test, y_train, y_test, train_mse, test_mse, _ = find_best_train_test_split(X[sel_features], y)
    else:
        X_train, X_test, y_train, y_test, train_mse, test_mse, _ = skl_lin_reg_validation(X[sel_features], y, tr, True)
        
    #pca = PCA()
    #X_train_PCA = pca.fit_transform(X_train)
    #X_test_PCA = pca.transform(X_test)
    #explained_variance = pca.explained_variance_ratio_
    #print(explained_variance)

    data_fin_df = pd.concat([y_train, X_train], axis=1, join='inner').reset_index()
    model = ols(formula=f, data=data_fin_df)

    return (sel_features, their_pvals, X_train, X_test, y_train, y_test, train_mse, test_mse, model)

def scatter_plots(df, target=None):
    df_minus_target = df.drop(target, axis=1) if target is not None else df
    
    s_html = "<h3>Scatter Plots:</h3><ul>"
    if target is not None:
        s_html += "<li><b>target</b>: {}</li>".format(target)
    s_html += "<li><b>feature set</b>: {}</li>".format(df_minus_target.columns)
    s_html += "</ul>"
    display(HTML(s_html))
    
    r_w = 20
    r_h = 4

    c_n = 4 if len(df_minus_target.columns) >= 4 else len(df_minus_target.columns)
    r_n = len(df_minus_target.columns)/c_n
    r_n = int(r_n) + (1 if r_n % int(r_n) != 0 else 0)

    fig = plt.figure(figsize=(r_w, r_h*r_n))

    axes = fig.subplots(r_n, c_n)

    for index, feat in enumerate(df_minus_target):
        ax = fig.add_subplot(r_n, c_n, index+1)
        plt.scatter(df[feat], df[target], alpha=0.2)
        plt.xlabel(feat)
        plt.ylabel(target)

    flattened_axes = axes.flatten()
    for unused in range(len(flattened_axes)-1, index, -1):
        fig.delaxes(flattened_axes[unused])

    fig.tight_layout()
    plt.show();

def split_categorical(df, p_cat, target=None):
    df_minus_target = df.drop(target, axis=1) if target is not None else df
    
    s_html = "<h2>Split Categorical:</h2><ul>"
    s_html += "<li><b>p-cat threshold</b>: {}</li>".format(p_cat)
    s_html += "<li><b>feature set</b>: {}</li>".format(df_minus_target.columns)
    s_html += "</ul>"
    display(HTML(s_html))
    
    if target is not None:
        scatter_plots(df, target)
    
    cat_classification_df = classify_as_categorical(df_minus_target, p_cat, False)
    display(HTML("<b>Categorical Features ($p\\_cat \\ge {}$):</b><br><br>".format(p_cat)))
    print_df(cat_classification_df)
    
    categorical_features = list(cat_classification_df['name'])
    continuous_features = list(df_minus_target.columns)

    s_html = "The following features are <i>apparently</i> <b>categorical</b> (based on $p\\_cat \\ge {}$):<br><ul>".format(p_cat)
    for cat_feat in categorical_features:
        continuous_features.remove(cat_feat)
        s_html += "<li><b>{}</b></li>".format(cat_feat)
    s_html += "</ul>"
    display(HTML(s_html))
    
    s_html = "<br>Based on the above, the following features are <i>apparently</i> <b>continuous</b>:<br><ul>"
    for cont_feat in continuous_features:
        s_html += "<li><b>{}</b></li>".format(cont_feat)
    s_html += "</ul>"
    display(HTML(s_html))
        
    return (cat_classification_df, categorical_features, continuous_features)

def run_full_regression_experiment(
    transformed_and_scaled_df
    , target
    , to_drop
    , tr
    , mv_r_sq_th
    , mv_mse_delta_th
    , mv_bad_vif_ratio_th
    , p_cat
    , fn_init_bin_bases
    , cont_and_cat_features_tuple=None
    , title=None):  

    # handle restricting to filter features: continuous + categorical
    if cont_and_cat_features_tuple is not None:
        continuous_features = cont_and_cat_features_tuple[0]
        categorical_features = cont_and_cat_features_tuple[1]
        all_features = continuous_features + categorical_features
        transformed_and_scaled_minus_todrop_df = pd.concat(
            [
                transformed_and_scaled_df[[target]]
                , transformed_and_scaled_df[all_features]
            ]
            , axis=1
            , join='inner'
        )
    else:
        transformed_and_scaled_minus_todrop_df = transformed_and_scaled_df.copy()

    if to_drop is not None and len(to_drop) > 0:
        transformed_and_scaled_minus_todrop_df = transformed_and_scaled_minus_todrop_df.drop(to_drop, axis=1)  
    
    # if it is None then we need to partition...
    if cont_and_cat_features_tuple is None:      
        (
            kchd_cat_classification_df
            , categorical_features
            , continuous_features
        ) = split_categorical(transformed_and_scaled_minus_todrop_df, p_cat, target)
    
    (
        default_handling_categoricals_list
        , special_handling_categoricals_map
    ) = fn_init_bin_bases(transformed_and_scaled_minus_todrop_df, categorical_features)
    
    transformed_and_scaled_and_categorized_df = categorize(
        transformed_and_scaled_minus_todrop_df
        , default_handling_categoricals_list
        , special_handling_categoricals_map
    )
    transformed_and_scaled_and_categorized_df = encode_col_names(transformed_and_scaled_and_categorized_df)
    
    (
        sel_features
        , their_pvals
        , X_train
        , X_test
        , y_train
        , y_test
        , train_mse
        , test_mse
        , model
    ) = lin_reg_model_from_auto_selected_features(transformed_and_scaled_and_categorized_df, target, tr=tr, title=title)

    (model_fit_results, good_vif_features, bad_vif_features) = model_fit_summary(
        transformed_and_scaled_and_categorized_df
        , sel_features
        , their_pvals
        , target
        , model
        , tr
        , train_mse
        , test_mse
        , mv_r_sq_th
        , mv_mse_delta_th
        , mv_bad_vif_ratio_th
    )
    
    return (sel_features, model_fit_results, train_mse, test_mse, good_vif_features, bad_vif_features, transformed_and_scaled_and_categorized_df)

def summarize_multicolinearity(df, target, corr_filter_threshold = 0.75):
    #features_only_df = df.drop([target], axis=1)
    features_only_df = df
    features_only_corr = features_only_df.corr()
    features_only_corr_bool_df = abs(features_only_corr) > corr_filter_threshold
    
    display(HTML("<b>Independent features that are greater than {}% correlated</b>:".format(round(corr_filter_threshold*100, 2))))
    print_df(features_only_corr_bool_df)

    plt.figure(figsize=(15,12)) # custom dims (not based on plot_edge)
    sns.heatmap(features_only_corr, center=0)
    plt.show();
    
    correlations = dict()
    max_corr = None
    correlated = []
        
    for index, col in enumerate(features_only_corr_bool_df.columns):
        corr_bool_counts = features_only_corr_bool_df[col].value_counts()
        if corr_bool_counts[1] > 1:  # then we have a pair-wise correlation > corr_filter_threshold
            # print("\n{} has {} correlations in corr_bool_counts:\n{}".format(col, corr_bool_counts[1], kchd_features_only_corr_bool_df[kchd_features_only_corr_bool_df[col]==True]))
            corrs = []
            for c in list(features_only_corr_bool_df[features_only_corr_bool_df[col]==True].index):
                if c != col:
                    corr = round(features_only_corr[col][c], 2)                    
                    print("{} {}% correlated to {}".format(col, round(corr*100, 2), c))                    
                    if col in correlations:
                        correlations[col].append((c, corr))
                    else:
                        correlations[col] = [(c, corr)]
                    if max_corr is None:
                        max_corr = (col, correlations[col])
                    elif len(correlations[col]) > len(max_corr[1]):
                        max_corr = (col, correlations[col])
                    corrs.append(c)
            correlated.append(corrs)
    if len(correlations) == 0:
        print("Congratulations! The {} feature-set does not manifest multicolinearity for threshold {}!".format(features_only_df.columns, corr_filter_threshold))       
    
    print("\nmost severe correlation: {}".format(max_corr))
    
    #largest_intersection(correlated)
    
    return (correlations, max_corr) 