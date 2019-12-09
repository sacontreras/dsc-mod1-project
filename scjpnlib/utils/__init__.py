import pandas as pd
from IPython.core.display import HTML
import numpy as np
from functools import reduce
import operator as op
import folium
from folium import plugins

# this function allows displaying dataframes using print() with pandas "pretty" HTML formatting
#   so that multiple "pretty" displays of dataframes can be rendered "inline"
#   default behavior (without specifying a range) is identical to that of df.head()
def print_df(df, n = None):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', -1)

    if n is None:
        n = len(df)
    display(HTML(df.head(n).to_html()))
    display(HTML("<br>{} rows x {} columns<br><br>".format(n, len(df.columns))))

    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_colwidth')

def print_df_head(df):
    print_df(df, n=5)

# note that we can qualitatively retrieve this information via df.info() but this function
#   provides us with an object we can manipulate programatically, whereas df.info() just 
#   prints the display of this information
def cols_with_nulls(df):
    cols_with_null_vals = []
    summary_cols = ['name', 'index', 'dtype', 'n_null']
    for idx, col in enumerate(df.columns):
        n_null = df[col].isna().sum()
        if n_null > 0:
            cols_with_null_vals.append([col, idx, df[col].dtype, n_null])
    return pd.DataFrame(cols_with_null_vals, columns=summary_cols)

def get_remaining_features(df, target, features):
    return df.drop([target] + features, axis=1).columns

# ordered difference
def list_difference(l1, l2):
    list_diff = []
    for item in l1:
        if item not in l2:
            list_diff.append(item)
    return list_diff

def categorical_probability(df, col, exclude_null_vals=True):
    unique_vals = df[col].unique()
    n_unique = len(unique_vals) 
    if exclude_null_vals:
        if df[col].dtype is int or df[col].dtype is float or df[col].dtype is np.float64:
            unique_vals_without_null = unique_vals[~np.isnan(unique_vals)]
        else:
            unique_vals_without_null = unique_vals[~pd.isnull(unique_vals)]
        if len(unique_vals_without_null) < len(unique_vals):
            unique_vals = unique_vals_without_null
            n_unique -= 1
    unique_vals = sorted(unique_vals)
    return (n_unique, round(1 - (n_unique/(len(df[col]))), 4), unique_vals)

def classify_as_categorical(
    df
    , p_cat_th
    , exclude_null_vals=True):
    
    cols_classified = []
    summary_cols = ['name', 'index', 'dtype', 'n_unique', 'p_cat', 'unique_vals']
    for idx, col in enumerate(df.columns):
        n_unique, p_cat, unique_vals = categorical_probability(df, col, exclude_null_vals)
        if p_cat >= p_cat_th:
            cols_classified.append([col, idx, df[col].dtype, n_unique, p_cat, unique_vals])
    return pd.DataFrame(cols_classified, columns=summary_cols)

# numeric_replacement_rules should be of the form:
#    {<name_of_col>: [(outlier_val_1, 'median'|'mean'|<numeric_replacement_value>), ((outlier_val_2, 'median'|'mean'|<numeric_replacement_value>)), ... , (outlier_val_n, 'median'|'mean'|<numeric_replacement_value>)]}}
def clean_offending_values(
    df
    , numeric_replacement_rules = None
    , string_replacement_rules = None
    , friendly_name_of_df = ""):

    friendly_name = friendly_name_of_df if len(friendly_name_of_df) > 0 else "df"
    print("*** CLEANING VALUES of {}: BEGIN ***".format(friendly_name))
    
    has_numeric_rules = numeric_replacement_rules is not None
    has_string_rules = string_replacement_rules is not None

    if has_numeric_rules or has_string_rules:

        if has_numeric_rules:
            for col, rules in numeric_replacement_rules.items():
                print("Rules for '{}' column value replacement are: {}".format(col, rules))
                # first determine if there are any offending values, which means we first need to collate said values
                offending_vals = []
                for val, rule in rules:
                    offending_vals.append(val)
                print("Looking for rows with '{}' values {} ...".format(col, offending_vals))
                df_rows_with_outliers = df[df[col].isin(offending_vals)==True]
                if len(df_rows_with_outliers) > 0:
                    offending_indexes = df_rows_with_outliers.index
                    print("Rows with offending values occur at {}.".format(offending_indexes))
                    # now handle specific replacement rule - we only have three allowable: median, mean, or a constant value
                    # if the rule is a constant value, then we can replace in place
                    data_type = type(rule)
                    replace_with_numeric_literal = data_type is int or data_type is float or data_type is np.float64
                    replace_with_median = data_type is str and rule.lower() == 'median'
                    replace_with_min = data_type is str and rule.lower() == 'min'
                    replace_with_max = data_type is str and rule.lower() == 'max'
                    replace_with_mean = data_type is str and rule.lower() == 'mean'
                    replace_with_callable = callable(rule)
                    if replace_with_numeric_literal:
                        df.loc[offending_indexes, col] = rule
                        print("Replaced {} offending instances in column '{}' with literal value {}\n".format(len(offending_indexes), col, rule))
                    elif replace_with_median or replace_with_min or replace_with_max or replace_with_mean:
                        # first we need to get a subset of the rows WITHOUT any of the offending values
                        df_rows_without_outliers = df[df[col].isin(offending_vals)==False]
                        if replace_with_median or replace_with_min or replace_with_max:
                            unique_vals = sorted(df_rows_without_outliers[col].unique())
                            imputed = np.min(unique_vals) if replace_with_min else (np.min(unique_vals) if replace_with_max else np.median(unique_vals))
                            s_imputed_from = "unique values: {}".format(unique_vals)
                        else:
                            imputed = np.mean(df_rows_without_outliers[col])
                            s_imputed_from = "values from index domain:\n{}".format(df_rows_without_outliers.index) 
                        df.loc[offending_indexes, col] = imputed 
                        s_desc = "the {} of column '{}'".format(rule, col)  
                        print("{} is {} and was imputed from {}".format(s_desc.capitalize(), imputed, s_imputed_from))
                        print("Replaced {} offending instances in column '{}' with {} ({})".format(len(offending_indexes), col, s_desc, imputed))
                    elif replace_with_callable:
                        df.loc[offending_indexes, col] = df.loc[offending_indexes, col].map(rule)
                        print("Replaced {} offending instances in column '{}' with results from callable {}".format(len(offending_indexes), col, rule))
                    else:
                        print("Unsupported or unknown rule: {}".format(rule))                   
                else:
                    print("There are no rows that contain values of '{}' in {}!".format(col, offending_vals))

        if has_string_rules:
            for col, rules in string_replacement_rules.items():
                print("Rules for '{}' are: {}".format(col, rules))
                # first determine if there are any offending values, which means we first need to collate said values
                offending_vals = []
                for val, rule in rules:
                    offending_vals.append(val)
                print("Looking for rows with '{}' values in {} ...".format(col, offending_vals))
                df_rows_with_outliers = df[df[col].isin(offending_vals)==True]
                if len(df_rows_with_outliers) > 0:
                    offending_indexes = df_rows_with_outliers.index
                    print("Rows with offending values occur at {}.".format(offending_indexes))
                    df.loc[offending_indexes, col] = rule
                    print("Replaced {} offending instances in column '{}' with literal value '{}'".format(len(offending_indexes), col, rule))                 
                else:
                    print("There are no rows that contain values of '{}' in {}!".format(col, offending_vals))

    else:
        print("Cannot clean outliers from {} since there were no replacement rules provided!".format(friendly_name))
    
    print("*** CLEANING VALUES of {}: END ***".format(friendly_name))

def log_transform(df, target_features, inplace=False):
    if not inplace:
        df = df.copy(deep=True)
    for feat in target_features:
        df[feat] = df[feat].map(lambda v: np.log(v))
    return df

# Min-max scaling: this will not be particularly useful to us in this case since this way of scaling brings values between 0 and 1
def min_max_scaling(df, target_features, inplace=False):
    if not inplace:
        df = df.copy(deep=True)
    for feat in target_features:
        min_feat = df[feat].min()
        max_feat = df[feat].max()
        df[feat] = df[feat].map(lambda x: (x-min_feat)/(max_feat-min_feat))
    return df

# standardization does not make data  moremore  normal, it will just change the mean and the standard error!
def standardization(df, target_features, inplace=False):
    if not inplace:
        df = df.copy(deep=True)
    for feat in target_features:
        mean_feat = df[feat].mean()
        sqr_var_feat = np.sqrt(np.var(df[feat]))
        df[feat] = df[feat].map(lambda x: (x-mean_feat)/sqr_var_feat)
    return df

# The distribution will have values between -1 and 1, and a mean of 0.
def mean_normalization(df, target_features, inplace=False):
    if not inplace:
        df = df.copy(deep=True)
    for feat in target_features:
        mean_feat = df[feat].mean()
        min_feat = df[feat].min()
        max_feat = df[feat].max()
        df[feat] = df[feat].map(lambda x: (x-mean_feat)/(max_feat-min_feat))
    return df    

# Unit vector transformation
def unit_vector_normalization(df, target_features, inplace=False):
    if not inplace:
        df = df.copy(deep=True)
    for feat in target_features:
        norm_feat = np.sqrt((df[feat]**2).sum())
        df[feat] = df[feat].map(lambda x: x/norm_feat)
    return df

def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)

def partition(df, feature, basis=None):
    if basis is None:
        basis = list(df[feature].unique())
    basis.append(min(basis)-1)
    basis = sorted(basis)
    binned_feature = pd.cut(df[feature], basis)
    feature_bins = binned_feature.cat.categories.values
    binned_feature = binned_feature.cat.as_unordered()
    #print("{} categorical bins are:\n{}".format(feature, feature_bins))
    return (binned_feature, feature_bins)

def categorize_feature(df, feature, basis=None):
    binned_feature, feature_bins = partition(df, feature, basis)
    feature_dummy = pd.get_dummies(binned_feature, prefix=feature, drop_first=True)
    return (pd.concat([df.drop([feature], axis=1), feature_dummy], axis=1), feature_bins)

def categorize(
    df
    , default_handling_categoricals_list
    , special_handling_categoricals_map):

    categorized_df = df
    
    # categorize features with "special-handling" bins
    feature_bins_list = []
    for feat, feat_bin_basis in special_handling_categoricals_map.items():
        categorized_df, feat_bins = categorize_feature(
            categorized_df
            , feat
            , feat_bin_basis
        )
        feature_bins_list.append((feat, feat_bins))
    if len(feature_bins_list) > 0:
        s_html = "Binning of \"special-handling\" categorical features:<br><ul>"
        for feat, feat_bins in feature_bins_list:
            s_html += "<li><b>{}</b>: {}</li>".format(feat, feat_bins)
        s_html += "</ul>"
        display(HTML(s_html))

    # categorize features with default binning semantic
    feature_bins_list = []
    for feat in default_handling_categoricals_list:
        categorized_df, feat_bins = categorize_feature(
            categorized_df
            , feat
        )
        feature_bins_list.append((feat, feat_bins))
    if len(feature_bins_list) > 0:
        s_html = "Binning of remaining categorical features:<br><ul>"
        for feat, feat_bins in feature_bins_list:
            s_html += "<li><b>{}</b>: {}</li>".format(feat, feat_bins)
        s_html += "</ul>"
        display(HTML(s_html))
    
    return categorized_df

def encode_col_names(df):
    return df.rename(columns=lambda x: x.replace(" ", "").replace(",", "__").replace(".", "_").replace("(", "e").replace("]", "i"))

def map_points(
    df
    , lat_col='lat'
    , lon_col='long'
    , zoom_start=11
    , plot_points=False
    , pt_radius=15
    , draw_heatmap=False
    , heat_map_weights_col=None
    , heat_map_weights_normalize=True
    , heat_map_radius=15):

    """Creates a map given a dataframe of points. Can also produce a heatmap overlay

    Arg:
        df: dataframe containing points to maps
        lat_col: Column containing latitude (string)
        lon_col: Column containing longitude (string)
        zoom_start: Integer representing the initial zoom of the map
        plot_points: Add points to map (boolean)
        pt_radius: Size of each point
        draw_heatmap: Add heatmap to map (boolean)
        heat_map_weights_col: Column containing heatmap weights
        heat_map_weights_normalize: Normalize heatmap weights (boolean)
        heat_map_radius: Size of heatmap point

    Returns:
        folium map object
    """
    
    df_local = df.copy()

    ## center map in the middle of points center in
    middle_lat = df_local[lat_col].median()
    middle_lon = df_local[lon_col].median()

    curr_map = folium.Map(
        location=[middle_lat, middle_lon]
        , zoom_start=zoom_start
        , control_scale=True
    )
    


    # add points to map
    if plot_points:
        for _, row in df_local.iterrows():
            folium.CircleMarker(
                [row[lat_col], row[lon_col]]
                , radius=pt_radius
                , popup=row[heat_map_weights_col]
                , fill_color="#3db7e4", # divvy color
            ).add_to(curr_map)

    # add heatmap
    if draw_heatmap:
        # convert to (n, 2) or (n, 3) matrix format
        if heat_map_weights_col is None:
            cols_to_pull = [lat_col, lon_col]
        else:
            # if we have to normalize
            if heat_map_weights_normalize:
                df_local[heat_map_weights_col] = df_local[heat_map_weights_col]/df_local[heat_map_weights_col].sum()
            cols_to_pull = [lat_col, lon_col, heat_map_weights_col]
        locations = df_local[cols_to_pull].values
        curr_map.add_child(
            plugins.HeatMap(
                locations
                , min_opacity=0.2
                , max_val=df_local[heat_map_weights_col].max()
                , radius=heat_map_radius
                , max_zoom=1
                , blur=15
                , gradient={
                    0: 'blue'
                    , .25: 'lime'
                    , .50: 'yellow'
                    , .75: 'orange'
                    , 1: 'red'
                }
            )
        )

    return curr_map