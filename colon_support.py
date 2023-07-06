"""Functions to support analysis of colon cancer dataset."""
from typing import List
import logging

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from util import (
    CoxItUp,
    filter_dict_by_keys,
    population_from_covariables,
    get_HR_details,
    predict_survival_from_CPH,
)
from scipy import stats
from data_science_tools.boxes import info, warn, error

log = logging.getLogger()

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
# dtypes for the columns in the dataset; Primarily used to read in the dataset.
ASA_dtype = pd.api.types.CategoricalDtype(categories=['1', '2', '3', '4', '9'], ordered=True)
sex_dtype = pd.api.types.CategoricalDtype(categories=['1', '2'], ordered=False)
yod_dtype = pd.api.types.CategoricalDtype(categories=range(2005, 2013), ordered=True)
comorbi_dtype = pd.api.types.CategoricalDtype(categories=['0', '1', '2', '9'], ordered=True)

pT_dtype = pd.api.types.CategoricalDtype(categories=['1', '2', '3', '4'], ordered=True)
pN_dtype = pd.api.types.CategoricalDtype(categories=['1', '2'], ordered=True)

schema_dtype = pd.api.types.CategoricalDtype(categories=['0', '1', '2'], ordered=True)
# recurrence_dtype = pd.api.types.CategoricalDtype(categories=['0', '1'], ordered=True)
# death_dtype = pd.api.types.CategoricalDtype(categories=['0', '1'], ordered=True)


# Mappings between codes and human-readable categories; Primarily used to read
# in the dataset.
sex_mapping = {
    '1': 'male',
    '2': 'female',
}

topo_mapping = {
    # "Proximal"
    'C180': 'cecum',
    'C181': 'appendix',
    'C182': 'ascending colon',
    'C183': 'hepatic flexure of colon',
    'C184': 'transverse colon',
    'C185': 'splenic flexure of colon',

    # "Distal"
    'C186': 'descending colon',
    'C187': 'sigmoid colon',

    # "NOS"
    'C188': 'overlapping lesion of colon',
    'C189': 'colon, NOS',
}

topo_simplification = {
    'proximal': ['C180', 'C181', 'C182', 'C183', 'C184', 'C185', ],
    'distal': ['C186', 'C187', ],
    'other': ['C188', 'C189', ],
}

comorbity_mapping = {
    '0': 'none',
    '1': '1',
    '2': '2+',
    '9': 'unknown',
}

ct_schema_mapping = {
    '0': 'None',
    '1': 'CAPOX',
    '2': 'CapMono',
}

grade_mapping = {
    '1': 'g1', # well differentiated
    '2': 'g2', # moderately differentiated
    '3': 'g3', # poorly differentiated
    '4': 'g4', # undifferentiated
    '9': 'unknown',
}


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def int_to_date(i: float) -> datetime:
    """Convert a SAS date back to datetime.

    A SAS date value represents the number of days since January 1, 1960.
    """
    sas_reference_date = datetime(1960, 1, 1)

    if pd.notna(i):
        return sas_reference_date + timedelta(i)

    return None


def load_dataset(filename):
    """Load the CSV, exported by Felice, from disk."""

    # Read data from disk.
    df = pd.read_csv(
        filename,
        dtype={
            'c3_ASA_class': ASA_dtype,
            'P_GESL': sex_dtype,
            'T_TOPOG': 'category',
            'T_DIFFGR': 'category',
            'T_INCJR': yod_dtype,
            'comorbi': comorbi_dtype,
            'pT': pT_dtype,
            'pN': pN_dtype,
            'ct_schema': schema_dtype,
            'recidief': 'boolean',
            'dood': 'boolean',
            'dood90dag_ok': 'boolean',
        }
    )

    df.set_index('key_id', inplace=True)

    # Change/fix the dtype of the SAS date columns.
    try:
        df['vitdat_updated_date'] = df['vitdat_updated'].map(int_to_date)
        df['datum_laatste_ok_date'] = df['datum_laatste_ok'].map(int_to_date)
    except KeyError:
        warn("<code>vitdat_updated_date</code> or <code>datum_laatste_ok_date</code> is not available!")


    # Rename the columns to readable English.
    df = df.rename(
        columns={
            'c3_ASA_class': 'ASA',
            'P_GESL': 'sex',
            'T_TOPOG': 'topography',
            'T_DIFFGR': 'grade',
            'T_INCJR': 'yod',
            'T_LEEFT': 'age_cont',
            'comorbi': 'comorbidities',
            'ct_schema': 'adj_therapy_type',
            'nw_compleet': 'adj_therapy_finished',
            'recidief': 'recurrence',

            'dood': 'deceased',
            'dood90dag_ok': 'died_within_90d_of_surg',

            'survdag_ok_updated': 'surv_days_post_surg',
            'survmnd_ok': 'surv_months_post_surg',
            'survmnd_eind_ct': 'surv_months_post_chemo',

            'RFSmnd_ok': 'RFS_months_post_surg',
            'RFSmnd_eind_ct': 'RFS_months_post_chemo',
        },
        errors = 'raise',
    )

    # Consolidate different granularities of post-surgery survival: the *days*
    # were updated, not the months or years.
    #
    # We'll need to divide by the nr. of days in a month to update
    # `surv_months_post_surg`.
    #
    # It seems Felice used 30.5 days/month.
    #   idx = ~df.recurrence & (df.rfs_event == 1)
    #   df[idx].surv_days_post_surg / df[idx].RFS_months_post_surg).mean()
    #   -> 30.49
    month_divider = 30.5

    try:
        idx_update_available = df.surv_days_post_surg.notna()
        df.loc[idx_update_available, 'surv_months_post_surg'] = df.loc[idx_update_available, 'surv_days_post_surg'] / month_divider

    except (KeyError, AttributeError):
        warn("<code>surv_days_post_surg</code> is not available!")

    df['surv_years_post_surg'] = df.surv_months_post_surg / 12

    # Fix boolean dtypes
    df['deceased'] = df['deceased'].astype(bool)
    df['died_within_90d_of_surg'] = df['died_within_90d_of_surg'].astype(bool)

    # Map categorical codes to readable values.
    df.ASA.cat.rename_categories({'9': 'unknown'}, inplace=True)
    df.sex.cat.rename_categories(sex_mapping, inplace=True)
    df['subsite'] = df['topography']
    df.subsite.cat.rename_categories(topo_mapping, inplace=True)
    df.grade.cat.rename_categories(grade_mapping, inplace=True)
    df.comorbidities.cat.rename_categories(comorbity_mapping, inplace=True)
    df.adj_therapy_type.cat.rename_categories(ct_schema_mapping, inplace=True)
    df.adj_therapy_type.cat.reorder_categories(['None', 'CapMono', 'CAPOX'], inplace=True)

    # Simplify treatment - three-way
    df['treatment'] = 'Surgery'
    df.loc[df.adj_therapy_type == 'CAPOX', 'treatment'] = 'Surgery + CAPOX'
    df.loc[df.adj_therapy_type == 'CapMono', 'treatment'] = 'Surgery + CapMono'

    # Simplify treatment - two-way
    adj_therapy_dtype = pd.api.types.CategoricalDtype(categories=['No', 'Yes'], ordered=True)
    df['adj_therapy'] = 'Yes'
    df.loc[df.adj_therapy_type == 'None', 'adj_therapy'] = 'No'
    df.adj_therapy = df.adj_therapy.astype(adj_therapy_dtype)

    # Simplify location
    df['location'] = 'other/unknown'
    df.loc[df.topography.isin(topo_simplification['proximal']), 'location'] = 'proximal'
    df.loc[df.topography.isin(topo_simplification['distal']), 'location'] = 'distal'

    location_dtype = pd.api.types.CategoricalDtype(categories=['proximal', 'distal', 'other/unknown'], ordered=True)
    df.location = df.location.astype(location_dtype)

    # Rename pT/pN codes to make it clear they're categories (and not integers)
    df.pT = df.pT.cat.rename_categories({
        '1': 'T1',
        '2': 'T2',
        '3': 'T3',
        '4': 'T4',
    })

    df.pN = df.pN.cat.rename_categories({
        '0': 'N0',
        '1': 'N1',
        '2': 'N2',
        'X': 'NX',
    })

    # Add TNM edition
    edition = pd.cut(
        df.yod,
        bins=[2002, 2009, 2016],
        labels=['TNM 6', 'TNM 7'],
        right=True
    )
    edition.name = 'edition'
    df['edition'] = edition


    return df


def create_cox_model_and_predict_survival(
        df: pd.DataFrame, covariables: List[str], description: str,
        tx_col: str, hr_summary_state, hr_summary: List = None,
        additionally_stratify_prediction: List = None, penalizer = 0,
        nyear = 10
    ):
    if hr_summary is None:
        hr_summary = {}

    if additionally_stratify_prediction is None:
        additionally_stratify_prediction = []

    # Function constants.
    time_col = 'surv_days_post_surg'
    event_col = 'deceased'

    # Calculate central values.
    central_values = df.mode().loc[0].to_dict()
    central_values['TxQ_bool'] = 'q3'

    references = filter_dict_by_keys(central_values, covariables)

    # Fit a Cox Model
    cph = CoxItUp(
        df,
        [*covariables, tx_col],
        time_col,
        event_col,
        penalizer=penalizer,
        silent=True
    )

    # Extract the HR for treatment and add it to the summary
    hr_summary[description] = (
        get_HR_details(
            cph,
            description,
            f'{tx_col}.{hr_summary_state}')
    )

    # Predict survival, stratified by treatment and additional variables.
    stratify_by = [tx_col, *additionally_stratify_prediction]
    ps = predict_survival_from_CPH(
        df,
        cph,
        stratify_by,
        references,
        nyear = nyear,
        fu_multiplier = 365
    )


    return hr_summary, ps


def table1(df, vars, outcome, p_name='p', p_precision=None, title=''):
    """Prepare Table 1"""
    def replace(string, dict_):
        for key, replacement in dict_.items():
            if string == key:
                return replacement

        return string

    # We're going to create multiple tables, one for each variable.
    tables = []

    col2 = df[outcome]
    if hasattr(col2, 'cat'):
        col2 = col2.cat.remove_unused_categories()

    totals = col2.value_counts()
    # headers = {
    #     header: f'{header} (n={total})' for header, total in totals.iteritems()
    # }

    headers = {
        header: f'{header}' for header, total in totals.iteritems()
    }

    # Iterate over the variables
    for v in vars:
        if v == outcome:
            continue

        col1 = df[v]

        if hasattr(col1, 'cat'):
            col1 = col1.cat.remove_unused_categories()

        # Crosstab with absolute numbers
        x1 = pd.crosstab(col1, col2, dropna=False)

        # Crosstab with percentages
        x2 = pd.crosstab(col1, col2, normalize='columns', dropna=False)
        x2 = (x2 * 100).round(1)

        # Chi2 is calculated using absolute nrs.
        try:
            chi2, p, dof, expected = stats.chi2_contingency(x1)
        except ValueError as e:
            log.warn(f"Could not calculate Chi2 statistic for '{v}': {e}")
            p = np.nan

        # Combine absolute nrs. with percentages in a single cell.
        xs = x1.astype('str') + ' (' + x2.applymap('{:3.1f}'.format) + ')'

        # Add the totals ('n={total}') to the headers
        xs.columns = [replace(h, headers) for h in list(xs.columns)]

        # Add the p-value in a new column, but only in the top row.
        xs[p_name] = ''

        if p_precision:
            p_tpl = f"{{:.{p_precision}f}}"
            xs.iloc[0, len(xs.columns) - 1] = p_tpl.format(p)
        else:
            xs[p_name] = np.nan
            xs.iloc[0, len(xs.columns) - 1] = p


        # If title is provided, we'll add a level to the column index and put
        # it there (on top).
        if title:
            colidx = pd.MultiIndex.from_product(
                [[title, ], list(xs.columns)],
            )

            xs.columns = colidx


        # Prepend the name of the current variable to the row index, so we can
        # concat the tables later ...
        xs.index = pd.MultiIndex.from_product(
            [[v, ], list(xs.index)],
            names=['variable', 'values']
        )

        tables.append(xs)

    return pd.concat(tables)

