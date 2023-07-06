"""
Utility functions for data analysis.
"""
import pickle
from itertools import product
from typing import List

import altair as alt
import numpy as np
import pandas as pd
from data_science_tools.base import *
from data_science_tools.plot import chart_with_markers
from IPython.core.display import display
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import ConvergenceError
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

table_nr = 0


class Pyramid(object):

    def __init__(self):
        self.data = []

    def _repr_html_(self):
        return self.as_dataframe()._repr_html_()

    def as_dataframe(self):
        return pd.DataFrame(self.data)

    def update(self, msg_or_filter: str, df: pd.DataFrame):
        self.data.append({
            'description': msg_or_filter,
            'n': df.shape[0],
        })


def load_env(filename):
    ENV_FILE = 'env.pickle'
    ENV_KEY = 'prostate'

    try:
        with open(filename, 'rb') as fp:
            env_ = pickle.load(fp)

    except FileNotFoundError:
        print(f'Could not load pickled environment from "{filename}".')
        print("Created clean environment ...")
        env_ = {}

    else:
        print(f'Loaded pickled environment from "{filename}".')

    return env_


def save_env(env_, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(env_, fp)


def print_env_rec(dict_, level=0, max_str_len=20):
    spaces = ' ' * level + ' - '

    for key, item in dict_.items():
        if isinstance(item, dict):
            print(spaces + key)
            print_env_rec(item, level+4)

        elif isinstance(item, str):
            s = (item[:max_str_len] + '..') if len(item) > max_str_len else item
            print(f'{spaces}{key}: {s}')

        elif isinstance(item, list):
            print(f'{spaces}{key}: [ ... ]')

        else:
            r = repr(item)
            s = (r[:max_str_len] + '..') if len(r) > max_str_len else r
            print(f'{spaces}{key}: {s}')


def print_env(env_):
    print('Environment contents: ')
    print_env_rec(env_)



def prepare_data_for_regression(data, drop_first=True):
    """Prepare data for use in (logistic) regression.

    This entails dummifying all categorical variables and dropping their
    first columns.

    Return:
        tuple: (DataFrame, dropped column names)
    """
    keep = []
    dummify = []

    for var in data.columns:
        if data[var].dtype.name == 'category':
            dummify.append(var)
        else:
            keep.append(var)

    if drop_first:
        dummies, reference_columns = get_dummies_and_ref_column_names(data, dummify)

    else:
        dummies = pd.get_dummies(data[dummify], drop_first=False)
        reference_columns = []

    data = data[keep].join(dummies)

    return data, reference_columns


def compute_propensity_model(data, outcome):
    """Compute a Logistic Regression of data onto outcome."""

    if outcome.dtype.name not in ('int64', 'bool'):
        raise Exception('Incorrect dtype for "outcome"')

    clf = LogisticRegression(random_state=0).fit(data, outcome)
    return clf


def compute_propensity_scores(covariables, outcome, quantiles=5):
    """Compute propensity scores using logistic regression.

        covariables: pandas.DataFrame to use as covariables
        outcome: pandas.Series (int or bool) to use as outcome
    """
    data, reference_columns = prepare_data_for_regression(covariables)
    clf = compute_propensity_model(data, outcome)

    # Predict treatment allocation (probability); results in a matrix with two columns: P(X=0), P(X=1)
    # We'll only keep the second column.
    treatment_proba = clf.predict_proba(data)[:,1]

    # Cast np.array to Series
    ps = pd.Series(treatment_proba, name="score", index=data.index)

    # Bin the predicted probabilities into quintiles. The result is a pandas.Categorical, not
    # a Series. Cast it ...
    labels = [f'q{q}' for q in range(1, quantiles+1)]
    q, bins = pd.qcut(treatment_proba, q=quantiles, labels=labels, retbins=True)
    q = pd.Series(q, name="quantile", index=data.index)

    return pd.concat([ps, q], axis=1)


def predict_survival(df, cph, tx_var, covariables, fu_multiplier=12, nyear=5,
                     model_name='1 - CoxPH', population_name=''):
    """Predict survival curve using a CoxPH model.

    Prediction is stratified by treatment, and age/sex if present in the model.

    Args:
        df (pd.DataFrame): dataframe, primarily used to obtain potential values
            for categorical data.
        cph (CoxPHFitter): Cox Model
        tx_var (str): treatment variable (i.e. variable to stratify on)
        covariables List[str]: list of (undummified) variables used to train the
            model.
    """

    # If follow-up was in months, multiply by 12
    times = np.array(range(1, nyear + 1)) * fu_multiplier

    # Create a dataframe as input for cph.predict(). Each row should correspond
    # to a case we're trying to predict. The columns should correspond to the parameters
    # in the model. These are (based on) the covariables, but may have been dummified
    # (and will not contain the first category).
    col_idx = cph.params_.index.copy()

    def get_options(df, var):
        return df[var].cat.categories.tolist()

    treatment = df[tx_var].cat.categories.tolist()
    age, sex = [], []

    cols = {
        tx_var: treatment
    }

    if 'age' in covariables:
        age = get_options(df, 'age')
        cols['age'] = age

    if 'sex' in covariables:
        sex = get_options(df, 'sex')
        cols['sex'] = sex

    # Create a dataframe containing all combinations of values
    combinations = pd.DataFrame(product(*cols.values()), columns=cols.keys())

    # Dummify the above combinations.
    dummies = pd.get_dummies(combinations, prefix_sep='.')

    # Find the columns that also occur in the CPH model.
    overlap_idx = cph.params_.index.intersection(dummies.columns)
    overlapping_dummies = dummies[overlap_idx]

    # Create a dataframe with all zeroes for the parameters in the model that
    # are *not* in dummies
    col_idx = cph.params_.index.difference(overlap_idx)
    additional_zeroes = pd.DataFrame(
        0,
        index=range(0, len(overlapping_dummies)),
        columns=col_idx
    )

    # Combine the dummies with the additional zeroes
    X = pd.concat([overlapping_dummies, additional_zeroes], axis=1)

    # display(X)

    pred = cph.predict_survival_function(X, times).transpose()
    pred.columns = (pred.columns / fu_multiplier).astype('int')
    pred = pred * 100

    pred_with_combinations = pd.concat([combinations, pred], axis=1)
    pred_with_combinations = pred_with_combinations.set_index(list(cols.keys()))
    pred_with_combinations.columns.name = 'survival'
    pred_with_combinations = pred_with_combinations.melt(ignore_index=False, value_name='pct')
    pred_with_combinations = pred_with_combinations.reset_index()

    pred_with_combinations['Model'] = model_name
    pred_with_combinations['Population'] = population_name

    return pred_with_combinations


def population_from_covariables(covariables: dict) -> str:
    covariables = remove_none_values_from_dict(covariables)
    s = ', '.join([f'{k}: {v}' for k, v in covariables.items()])
    return s or 'All'


def create_input_for_prediction(df, cph, qd, qv=None, simple_index=True):
    """
    Args:
        df (pd.DataFrame) used to train the Cox model. Should contain the un-dummified variables.
        cph (CoxPHFitter): trained cox model. who knows what we might need it for.
        qd (list): query "distribution". set of variables for which we'll determine possible states
            and compute all possible combinations.
        qv (dict): query "values". specific values of variables that we want to compute a prediction
            for.
    """
    if qv is None:
        qv = {}

    def get_variables_in_cph_model():
        return [*set([s.split('.')[0] for s in cph.params_.index])]

    central_values = df.mode().loc[0].to_dict()

    # First calculate the combinations
    def get_states(df, var):
        return df[var].cat.categories.tolist()

    cols = {var: get_states(df, var)  for var in qd}

    # Add the fixed values
    for variable, value in qv.items():
        if not isinstance(value, (list, tuple)):
            value = [value]

        cols[variable] = value

    # Next, make sure that all remaininig columns are present and set to their central values.
    for variable in get_variables_in_cph_model():
        try:
            if variable not in cols:
                cols[variable] = [central_values[variable]]
        except:
            pass

    combinations = pd.DataFrame(product(*cols.values()), columns=cols.keys())

    # The *categorical* input combinations need to be converted to
    # dummy-variables.
    # FIXME: Without setting `columns=` get_dummies() only processes categorical
    #        variables. As a result of calculating the combinations, this
    #        dtype may have been lost.
    dummies = pd.get_dummies(
        combinations,
        prefix_sep='.',
        columns=combinations.columns.tolist()
    )

    # Now that we have dummies, let's check which also occur in the model (and remove the others).
    overlap_idx = cph.params_.index.intersection(dummies.columns)
    overlapping_dummies = dummies[overlap_idx]

    additional_zeroes = pd.DataFrame(
        0,
        index=range(0, len(overlapping_dummies)),
        columns=cph.params_.index.difference(overlap_idx)
    )

    combined = pd.concat(
        [overlapping_dummies, additional_zeroes],
        axis=1
    ).sort_index(axis=1)

    if simple_index:
        keys = [*qd, *qv.keys()]
    else:
        keys = combinations.columns.tolist()

    combined = pd.concat([combinations[keys], combined], axis=1)
    combined.set_index(keys, inplace=True)
    return combined


# def predict_survival_from_CPH(df: pd.DataFrame, cph: CoxPHFitter,
#                               stratify: List[str], covariables: dict,
#                               fu_multiplier=12, nyear=5, model_name='1 - CoxPH'):
#     """Predict survival curve using a CoxPH model.
#
#     Prediction is stratified by treatment, and age/sex if present in the model.
#
#     Args:
#         df (pd.DataFrame): dataframe, primarily used to obtain potential values
#             for categorical data.
#         cph (CoxPHFitter): Cox Model
#         stratify:
#         covariables: dict
#     """
#     X = create_input_for_prediction(df, cph, qd=stratify)
#
#     # If follow-up was in months, multiply by 12
#     times = np.array(range(1, nyear + 1)) * fu_multiplier
#
#     pred = cph.predict_survival_function(X, times).transpose()
#     pred.columns = (pred.columns / fu_multiplier).astype('int')
#     pred = pred * 100
#     pred.index = X.index
#
#     # pred_with_combinations = pred_with_combinations.set_index(list(cols.keys()))
#     pred.columns.name = 'survival'
#     pred = pred.melt(ignore_index=False, value_name='pct')
#     pred = pred.reset_index()
#
#     pred['Model'] = model_name
#     pred['Population'] = population_from_covariables(covariables)
#
#     return pred


def reduce_expected_survival(es, cols):
    """Return the mean/CI over the groups present in the predicted survival."""
    lower_bound, upper_bound = norm.ppf(0.025), norm.ppf(0.975)

    mean = es.groupby(cols).pct.mean()
    std = es.groupby(cols).pct.std()
    lower_y = np.maximum(0, mean + (std * lower_bound))
    upper_y = np.minimum(100, mean + (std * upper_bound))

    reduced = pd.DataFrame({
        'pct': mean,
        'y_lower': lower_y,
        'y_upper': upper_y,
    }).reset_index()
    reduced['Model'] = '0 - Expected survival'
    return reduced

def predict_survival_from_CPH(df: pd.DataFrame, cph: CoxPHFitter,
                              stratify: List[str], covariables: dict,
                              fu_multiplier=12, nyear=5, model_name=None):
    """Predict survival curve using a CoxPH model.

    Prediction is stratified by treatment, and age/sex if present in the model.

    Args:
        df (pd.DataFrame): dataframe, primarily used to obtain potential values
            for categorical data.
        cph (CoxPHFitter): Cox Model
        stratify:
        covariables: dict
    """
    import copy

    from lifelines import utils

    input_ = create_input_for_prediction(df, cph, qd=stratify)

    # If follow-up was in months, multiply by 12
    times = np.array(range(1, nyear + 1)) * fu_multiplier

    def predict_without_CI(cph, input_, value, times, fu_multiplier):
        cvs = input_.loc[value, :]

        if isinstance(cvs, pd.Series):
            cvs = pd.DataFrame(cvs).transpose()

        try:
            pred = cph.predict_survival_function(cvs, times).transpose()
        except Exception as e:
            raise

        pred.columns = (pred.columns / fu_multiplier).astype('int')
        pred = pred * 100
        pred.index = cvs.index

        pred.columns.name = 'survival'
        pred = pred.melt(ignore_index=False, value_name='coef')
        pred = pred.reset_index()
        pred['coef lower 95%'] = np.nan
        pred['coef upper 95%'] = np.nan
        pred[input_.index.names[0]] = value

        try:
            pred.drop(columns=['index'], inplace=True)
        except:
            # display(pred)
            pass

        return pred

    def predict_with_CI(cph, input_, value, times, fu_multiplier):
        # Create input to use to predict survival with, which requires a value
        # for each covariable.
        varname = input_.index.names[0]
        cvs = input_.loc[value, :]

        # Some work is required to keep the index as-is
        if isinstance(cvs, pd.Series):
            cvs = pd.DataFrame(cvs).transpose()
            cvs.index.name = varname
        else:
            cvs = pd.concat({value: cvs}, names=[varname])


        X = cph.regressors.transform_df(cvs)["beta_"]
        X = utils.normalize(X, cph._norm_mean.values, 1)

        n = X.shape[0]
        times_to_evaluate_at = np.tile(times, (n, 1))
        c_0 = utils.interpolate_at_times(
            cph.baseline_cumulative_hazard_,
            times_to_evaluate_at
        ).T

        for param_est in ['coef', 'coef lower 95%', 'coef upper 95%']:
            # params will hold the 'regular' coefficients for each of the
            # covariables in the model
            params = copy.deepcopy(cph.params_)

            # Replace the 'regular' coefficient with either the regular
            # estimate (which is a bit redundant), lower 95% estimate or
            # upper 95% estimate.
            params[covariate] = cph.summary[param_est][covariate]

            log_partial_hazard = pd.Series(
                np.dot(X, params),
                index=X.index
            )
            partial_hazard = np.exp(log_partial_hazard)
            col = utils._get_index(partial_hazard)

            cumulative_hazard_ = pd.DataFrame(
                c_0 * partial_hazard.values,
                columns=col,
                index=times
            )

            pred = np.exp(-cumulative_hazard_) * 100
            pred.index = (pred.index / fu_multiplier).astype('int')
            pred.index.name = 'survival'

            pred.columns = cvs.index
            pred = pred.melt(
                value_name=param_est,
                ignore_index=False
            )
            pred = pred.reset_index()

            survival_results[param_est] = pred

        return pd.concat(
            [
                survival_results['coef'],
                survival_results['coef lower 95%']['coef lower 95%'],
                survival_results['coef upper 95%']['coef upper 95%'],
            ],
            axis=1
        )

    survival_results = {}
    tmp = None

    variable = stratify[0]
    value_set = df[variable].cat.categories

    for value in value_set:
        covariate = f'{variable}.{value}'

        # If a value is not in the model parameters, we'll use the baseline instead
        if covariate not in input_.columns.tolist():
            # We're probably dealing with the reference value. We can't
            # compute a CI here :-(.
            prediction_as_df = predict_without_CI(cph, input_, value, times, fu_multiplier)
            # return prediction_as_df

        else:
            # Ok then!
            # predict_without_CI(cph, input_, value, times, fu_multiplier)
            prediction_as_df = predict_with_CI(cph, input_, value, times, fu_multiplier)
            # return prediction_as_df

        if tmp is None:
            tmp = prediction_as_df
        else:
            tmp = pd.concat([tmp, prediction_as_df], axis=0)

    references = covariables.copy()
    for stratifier in stratify:
        references.pop(stratifier, None)

    population = population_from_covariables(references)

    if model_name is None:
        model_name = f'1 - CoxPH ({population})'

    tmp['Model'] = model_name
    tmp['Population'] = population
    tmp = tmp.rename(columns={
        'coef': 'pct',
        'coef lower 95%': 'y_lower',
        'coef upper 95%': 'y_upper',
    })

    return tmp


def predict_survival_from_BN(bn, tx_var='adj_therapy_type',
                             additional_evidence=None,
                             years: List[int] = range(1, 6)):
    """Calculate survival curves for the BN."""
    rows = []

    if additional_evidence is None:
        additional_evidence = {}

    # for combination in product(bn.sex.states, bn.age.states, bn.adj_therapy_type.states):
    treatment_states = bn[tx_var].states

    for combination in product(bn.age.states, treatment_states):
        e = {
            'age': combination[0],
            tx_var: combination[1],
            **additional_evidence
        }

        for y in years:
            probability = bn.compute_posterior(
                [],
                {f'surv_{y:02}y': 'true'},
                [],
                e
            )
            row = (combination + (y, ) + tuple(100 * probability))
            rows.append(row)

    df = pd.DataFrame(rows, columns=['age', tx_var, 'survival', 'pct'])
    df[tx_var] = pd.Categorical(
        df[tx_var],
        categories=treatment_states,
        ordered=True
    )
    return df


def plot_prediction(ps, es, color, column=None, row=None, title='',
                    reduce_es=True, combine: List[str] = None):
    """Plot predicted and expected survival.

    Args:
        ps (pd.DataFrame): predicted survival
        es (pd.DataFrame): expected survival
        color (str): column that holder adjuvant therapy type
        title (str): title to use for the plot

    Return:
        alt.Chart()
    """
    kwargs = remove_none_values_from_dict({
        'column': column,
    })

    if (column is None) and ('age' in ps.columns):
        column = 'age'
        kwargs['column'] = 'age:O'
    else:
        kwargs['column'] = column

    if (row is None) and ('sex' in ps.columns):
        row = 'sex'
        kwargs['row'] = 'sex:N'
    else:
        kwargs['row'] = row

    # Altair is going to stratify by
    # 'survival', 'Model' and color plus, potentially, by column and row.
    cols = remove_none_values_from_list([
        'survival',
        'Model',
        color,
        column,
        row
    ])

    if reduce_es:
        # Remove any shorthand type tags
        cols = list(set([c.split(':')[0] for c in cols]))

        # print('Reducing by', cols)
        es = reduce_expected_survival(es, cols)

    cd = pd.concat([ps, es])

    if combine:
        stratum = 'stratum'
        cd[stratum] = cd[combine].agg(', '.join, axis=1)
    else:
        stratum = color

    return chart_with_markers(
        cd,
        x='survival:Q',
        y='pct:Q',
        color=f'{stratum}:N',
        strokeDash='Model',
        title=title,
        **kwargs
    )


# ------------------------------------------------------------------------------
# Cox PH related
# ------------------------------------------------------------------------------
def get_dummies_and_ref_column_names(df, dummify, sep='.'):
    """Dummify categorical columns, drop the first column, return dummies and first column names."""
    dummies = pd.get_dummies(df[dummify], prefix_sep=sep, drop_first=True)
    reference_columns = [f'{var}{sep}{df[var].cat.categories[0]}' for var in dummify]

    return dummies, reference_columns


def add_references_to_cox_summary(summary, references):
    """Add the reference categories (i.e. the ones left out) to the summary.

    Arguments:
        summary: cph.summary object
        references: list of reference values (strings)
    """

    # Create rows for the reference values
    reference_rows = pd.DataFrame(np.nan, index=references, columns=summary.columns)

    reference_rows.loc[:, 'coef'] = 0
    reference_rows.loc[:, 'exp(coef)'] = 1
    reference_rows.loc[:, 'reference'] = 1

    # return summary.index.str.split('.', expand=True)

    summary = summary.append(reference_rows)
    summary.index = summary.index.str.split('.', expand=True)
    summary.index.names = ['variable', 'value']

    # determine the original sort order ...
    level0_values = summary.index.get_level_values(0).drop_duplicates().tolist()

    summary = summary.reset_index()
    summary = summary.sort_values(by=['variable', 'reference', 'value'])
    summary = summary.set_index(['variable', 'value'])
    summary = summary.loc[level0_values, :]
    # summary = summary.sort_index(key=lambda x: x.str.lower())

    del summary['reference']

    return summary


#  highlight(s):
#  """highlight ..."""
#  is_significant = s['p'] <= 0.05
#
#  styles = []
#
#  for idx in range(len(s)):
#      if is_significant:
#          styles.append('background-color: #FFFFBB; font-weight: bold')
#      elif idx == 0:
#          styles.append('font-weight: bold')
#      else:
#          styles.append('')
#
#  return styles


def display_summary(cph_or_summary, references=None, caption='', table_nr=''):
    if isinstance(cph_or_summary, CoxPHFitter):
        summary = cph_or_summary.summary
    else:
        summary = cph_or_summary

    # Just to be sure
    summary = summary.copy()
    del summary['z']
    del summary['-log2(p)']
    del summary['coef lower 95%']
    del summary['coef upper 95%']


    if references:
        summary = add_references_to_cox_summary(summary, references)

    summary['sign.'] = ''
    summary.loc[summary.p <= 0.05, 'sign.'] = '*'
    summary.loc[summary.p <= 0.005, 'sign.'] = '**'
    summary.loc[summary.p <= 0.0005, 'sign.'] = '***'

    styled = summary.style
    styled = styled.set_properties(**{
        'font-family': 'monospace',
        #'border': '1px solid red',
    })

    if caption and table_nr:
        caption = f'<b>Table {table_nr}:</b> {caption}'
    elif caption:
        caption = f'<b>Table:</b> {caption}'

    styled = styled.set_caption(caption)
    styled = styled.set_precision(3)

    styled = styled.format(None, na_rep="")
    table_styles = [
        {
            'selector': '',
            'props': [
                ('border', '1px solid #cfcfcf'),
            ]
        }, {
            'selector': 'th',
            'props': [
                ('font-family', 'monospace'),
                ('font-weight', 'bold'),
                ('vertical-align', 'top'),
            ]
        }, {
            'selector': 'caption',
            'props': [
                ('caption-side', 'bottom'),
                ('font-family', 'monospace'),
                ('text-align', 'left'),
                ('color', '#666666'),
                ('padding', '10px 40px'),
            ]
        },
    ]

    for i in summary.index[summary.p <= 0.05]:
        iloc = summary.index.get_loc(i)
        table_styles.append({
            'selector': f'.row{iloc}',
            'props': [
                ('font-weight', 'bold'),
                ('color', '#003366'),
                ('background-color', '#ffffbb'),
            ]
        })

    styled = styled.set_table_styles(table_styles)

    display(styled)
    return summary


def CoxItUp(df, covariates: list, time='vitfup', event='vitstat', silent=False, penalizer=0):
    """Helper function.

        Dummifies categorical variables, performs Cox regression and displays the
        results. Depends on global variable "table_nr" for displaying the caption.

        Note: any rows with NAs are dropped.
    """
    global table_nr

    table_nr += 1
    dummify = []
    keep = []

    for col in covariates:
        if df[col].dtype.name == 'category':
            # If the columns type is categorical, we'll dummify the variable later.
            dummify.append(col)

            # if col not in dummify:
            #     # Pick the 1st value
            #     dummify[col] = df[col].cat.categories[0]

        else:
            # Else, we'll treat it as a numerical/continuous value
            keep.append(col)

    #
    dummies, reference_columns = get_dummies_and_ref_column_names(df, dummify)
    keep = keep + [time, event]

    # Only keep the columns/variables needed for the Cox regression.
    subset = df[keep].join(dummies)
    subset = subset.dropna()

    caption = f'{" + ".join(covariates)} ~ survival'

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(subset, duration_col=time, event_col=event)

    if not silent:
        display_summary(cph, reference_columns, caption=caption, table_nr=table_nr)

    return cph


def get_HR_details(cph, description, row_idx):
    """Extract the HR for treatment and compute probability of survival."""

    # Extract HR & CIs
    HR = cph.summary.loc[row_idx, 'exp(coef)']
    CI_min = cph.summary.loc[row_idx, 'exp(coef) lower 95%']
    CI_max = cph.summary.loc[row_idx, 'exp(coef) upper 95%']
    p = cph.summary.loc[row_idx, 'p']

    result = [f'{HR:.2f}', f'{CI_min:.2f}-{CI_max:.2f}', p]
    return pd.DataFrame(
        [result],
        index=[description],
        columns=['HR', '95% CI', 'p'],
    )

    # Predict effect of adjuvant treatment survival
    X = pd.DataFrame(0, index=['No', 'Yes'], columns=cph.params_.index.copy())
    X.loc['Yes', row_idx] = 1

    times = [2 * 12, 3 * 12, 5 * 12]
    pred = cph.predict_survival_function(X, times)
    pred.index = ['2-year survival', '3-year survival', '5-year survival', ]
    pred['delta'] = pred['Yes'] - pred['No']

    pred = pred.round(3)
    pred = pd.DataFrame(pred.stack()).transpose()
    pred.index = [description]

    col_idx = pd.MultiIndex.from_product([[''], ['HR', '95% CI', 'p']])
    d1 = pd.DataFrame(
        [result],
        index=[description],
        columns=col_idx,
    )

    return pd.concat([d1, pred], axis=1)



# %%
