import datetime
import sys
from collections import defaultdict, Counter
from operator import itemgetter
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import ceil
from sklearn.preprocessing import StandardScaler

# Turn this on if the prints are needed.
DEBUG_MODE = False;

# Global constants
g_col_types = {
    "OBJECTID": pd.Int64Dtype(),  # useful, clean, shows record ID starting from 1
    "FOD_ID": pd.Int64Dtype(),  # not useful
    "FPA_ID": str,  # not useful
    "SOURCE_SYSTEM_TYPE": str,  # not useful
    "SOURCE_SYSTEM": str,  # not useful
    "NWCG_REPORTING_AGENCY": str,  # not useful
    "NWCG_REPORTING_UNIT_ID": str,  # not useful
    "NWCG_REPORTING_UNIT_NAME": str,  # not useful
    "SOURCE_REPORTING_UNIT": str,  # not useful
    "SOURCE_REPORTING_UNIT_NAME": str,  # not useful
    "LOCAL_FIRE_REPORT_ID": str,  # not useful
    "LOCAL_INCIDENT_ID": str,  # not useful
    "FIRE_CODE": str,  # not useful
    "FIRE_NAME": str,  # not useful
    "ICS_209_INCIDENT_NUMBER": str,  # not useful
    "ICS_209_NAME": str,  # not useful
    "MTBS_ID": str,  # not useful
    "MTBS_FIRE_NAME": str,  # not useful
    "COMPLEX_NAME": str,  # not useful
    "FIRE_YEAR": pd.Int64Dtype(),  # useful, clean
    "DISCOVERY_DATE": float,  # useful, clean
    "DISCOVERY_DOY": pd.Int64Dtype(),  # useful, clean
    "DISCOVERY_TIME": pd.Int64Dtype(),  # useful, not clean, 882638 missing values
    "STAT_CAUSE_CODE": float,  # useful, clean
    "STAT_CAUSE_DESCR": str,  # useful, clean
    "CONT_DATE": float,  # useful, not clean, 891531 missing values
    "CONT_DOY": pd.Int64Dtype(),  # useful, not clean, 891531 missing values
    "CONT_TIME": pd.Int64Dtype(),  # useful, not clean, 972553 missing values
    "FIRE_SIZE": float,  # useful, clean
    "FIRE_SIZE_CLASS": str,  # useful, clean
    "LATITUDE": float,  # useful, clean
    "LONGITUDE": float,  # useful, clean
    "OWNER_CODE": float,  # not useful
    "OWNER_DESCR": str,  # useful, clean
    "STATE": str,  # useful, clean
    "COUNTY": str,  # useful, not clean, 678148 missing values
    "FIPS_CODE": str,  # not useful
    "FIPS_NAME": str,  # not useful
    "DAYS_2_CONTROL": pd.Int64Dtype(),  # useful, clean, generated column
};

# Columns to be dropped from the data frame
g_not_useful_cols = [
    "FOD_ID",
    "FPA_ID",
    "SOURCE_SYSTEM_TYPE",
    "SOURCE_SYSTEM",
    "NWCG_REPORTING_AGENCY",
    "NWCG_REPORTING_UNIT_ID",
    "NWCG_REPORTING_UNIT_NAME",
    "SOURCE_REPORTING_UNIT",
    "SOURCE_REPORTING_UNIT_NAME",
    "LOCAL_FIRE_REPORT_ID",
    "LOCAL_INCIDENT_ID",
    "FIRE_CODE",
    "FIRE_NAME",
    "ICS_209_INCIDENT_NUMBER",
    "ICS_209_NAME",
    "MTBS_ID",
    "MTBS_FIRE_NAME",
    "COMPLEX_NAME",
    "OWNER_CODE",
    "OWNER_DESCR",
    "FIPS_CODE",
    "FIPS_NAME"
];

g_default_values = {
    "FIRE_YEAR": 0,
    "DISCOVERY_DATE": 0,
    "DISCOVERY_DOY": 0,
    "DISCOVERY_TIME": 9999,
    "STAT_CAUSE_CODE": 0,
    "STAT_CAUSE_DESCR": 'missing',
    "CONT_DATE": 0,
    "CONT_DOY": 0,
    "CONT_TIME": 9999,
    "FIRE_SIZE": 0,
    "FIRE_SIZE_CLASS": 'missing',
    "LATITUDE": 999,
    "LONGITUDE": 999,
    "OWNER_DESCR": 'missing',
    "STATE": 'missing',
    "COUNTY": 'missing',
    "DAYS_2_CONTROL": 0
}


def inspect_dataset(fname, col_dtypes):
    '''
    :param fname      : file name to read. This file is read as a CSV.
    :param col_dtypes : column data type dictionary.
    :return:            data frame
                        list of data frame columns,
                        dictionary of columns with null value count
                        dictionary of columns with number of unique values found
    '''

    base_df = pd.read_csv(fname, dtype=col_dtypes);
    base_col_names = base_df.columns.values.tolist();
    col_unique_vals = defaultdict(int);
    clean_cols = []
    dirty_cols = {}

    for col in base_col_names:
        col_unique_vals[col] = base_df[col].nunique();

        if DEBUG_MODE:
            print(col + " has NA values : " + str(len(base_df[base_df[col].isna()].index)));

        if len(base_df[base_df[col].isna()].index) == 0:
            clean_cols.append(col);
        else:
            dirty_cols[col] = len(base_df[base_df[col].isna()].index);

        if DEBUG_MODE:
            print("\n");
    # end for

    if DEBUG_MODE:
        print("\n\nColumns with no Null values");
        for col in clean_cols:
            print(col + " has no Null values. It has " + str(col_unique_vals[col]) + " unique values");
        print("\n\nColumns with some Null values");
        for key in dirty_cols.keys():
            print(key + " has " + str(dirty_cols[key]) + " Null values and " + str(
                col_unique_vals[col]) + " unique values (including Null)");
    # end if DEBUG_MODE

    return base_df, base_col_names, dirty_cols, col_unique_vals;

# END function inspect_dataset


def drop_uninteresting_cols(df, col_drop_list):
    '''
    :param df:              Input data frame
    :param col_drop_list:   column list that must be dropped
    :return:                Returns a data frame in which all the columns listed in the col_drop_list is dropped.
    '''
    return df.drop(col_drop_list, axis=1);


# END drop_uninteresting_cols


def time_2_contain(row):
    '''
    :param   row: Input data frame row
    :return:      Returns the how many days were required to contain the fire.
    '''

    start_day = row['DISCOVERY_DOY']
    end_day = row['CONT_DOY']

    if end_day == 0:
        return 0;
    elif end_day >= start_day:
        return end_day - start_day + 1;
    else:
        return 365 + end_day - start_day + 1;


# END time_2_contain


def extrapolate_days2ctrl(df, y1, y2):
    '''
    As we can see the the days_2_control field has many 0 values. These values are missing because cont_doy is missing.
    But these values can be predicted based on certain assumptions about the wildfire. Over the years US wildfire
    mitigation have definitely improved the equipment that has been used to subdue the wildfire. But within a given year
    we can assume that the time taken to contain a wildfire is directly proportional to size of the Fire. With this
    assumption we can fit a linear regression model to fit a curve for existing values of FIRE_SIZE vs DAYS_2_CONTROL.
    And then use the learned model to extrapolate and guess the DAYS_2_CONTROL for missing rows.
    This function performs this operation and fills up the rows where DAYS_2_CONTROL is 0
    :param  df: data frame with DAYS_2_CONTROL field
    :return df: Updated df where all the values of DAYS_2_CONTROL are non-zero.
    '''

    if y1 + 5 != y2:
        assert False;

    df2use = df;
    df2use = df2use[df2use['FIRE_YEAR'] > y1];  # Filter records > y1
    df2use = df2use[df2use['FIRE_YEAR'] <= y2];  # Filter records <= y1
    fitdf = df2use[df2use['DAYS_2_CONTROL'] != 0];  # Filter out data for modeling
    preddf = df2use[df2use['DAYS_2_CONTROL'] == 0];  # Filter the data for predicting based on model

    # Get the mode for DAYS_2_CONTROL
    fitdf = fitdf[["FIRE_SIZE", "DAYS_2_CONTROL"]].groupby("FIRE_SIZE").agg(lambda x: x.value_counts().index[0]). \
        sort_values(by="FIRE_SIZE", ascending=False);
    X = fitdf.index.values.tolist();
    Y = fitdf['DAYS_2_CONTROL'].tolist();
    z = np.polyfit(X, Y, 2);  # Fit a n-degree polynomial
    f = np.poly1d(z);  # Model handle

    X_2predict = np.array(list(set(preddf['FIRE_SIZE'].tolist())));
    Y_predicted = f(X_2predict);
    temp_dict = dict(zip(X_2predict, Y_predicted));

    for index, row in preddf.iterrows():
        oid = row['OBJECTID'];
        fsz = row['FIRE_SIZE'];
        if df.DAYS_2_CONTROL.iloc[int(oid) - 1] != 0:
            assert False;
        else:
            df.DAYS_2_CONTROL.iloc[int(oid) - 1] = ceil(temp_dict[fsz]);

    # # plt.legend();
    # plt.plot(X, Y, label=str(year));
    # plt.xlabel("Days2Control");
    # plt.ylabel("Fire_size");
    # plt.title("Trend for year - " + str(year));
    # plt.savefig("./year_trends/" + str(year)+ "_" + s +"_trend.png", facecolor="grey");
    # plt.close();
    return df


# Usage - This program requires the data set in csv format input as a command line argument.
# $> python data_inspection.py <data-set file in csv format>
if __name__ == "__main__":

    # Command line input processing done here....
    if len(sys.argv) != 2:
        print("Usage:\n  python data_inspection.py <dataset.csv>")
        # Terminate the program if there is no command line input
        exit();

    # NOTE: If using command line arguments is not required then assign the file name directly to dataset_fname variable
    #       instead of obtaining it from sys.argv[1]
    dataset_fname = sys.argv[1];
    # Command line inputs processing ends here....

    # At this point variable 'dataset_fname' contains the filename that must be read as a Pandas data-frame.

    # Load the data set into the data frame....
    base_df, base_cols, base_cols_na, base_cols_uniq = inspect_dataset(dataset_fname, g_col_types)

    # Drop uninteresting columns. Uninteresting columns are listed in global list 'g_not_useful_cols'
    base_df = drop_uninteresting_cols(base_df, g_not_useful_cols);

    if DEBUG_MODE:
        print(base_df.columns.values.tolist());

    base_df = base_df.fillna(value=g_default_values);
    base_df['DAYS_2_CONTROL'] = base_df.apply(time_2_contain, axis=1);
    base_df.to_csv("columns_cleaned.csv", index=False);

    # Update the base_df with newly created file - columns_cleaned.csv
    base_df, base_cols, base_cols_na, base_cols_uniq = inspect_dataset("columns_cleaned.csv", g_col_types)
    years = [1995, 2000, 2005, 2010, 2015, 2020];
    for y in years:
        # Extrapolate the values and make an educated guess on time required to contain the fire for missing values
        base_df = extrapolate_days2ctrl(base_df, y - 5, y);

        if DEBUG_MODE:
            print("Year: " + str(y) + " completed.");

        if (min(base_df['DAYS_2_CONTROL'].tolist())) > 0:
            break;

    if DEBUG_MODE:
        print("Minimum no. of days required to control the fire: " + str(min(base_df['DAYS_2_CONTROL'].tolist())));
        print("Maximum no. of days required to control the fire: " + str(max(base_df['DAYS_2_CONTROL'].tolist())));

    base_df.to_csv("columns_cleaned_extrapolated.csv", index=False);

    # Update the base_df with newly created file - columns_cleaned_extrapolated.csv

    base_df, base_cols, base_cols_na, base_cols_uniq = inspect_dataset("columns_cleaned_extrapolated.csv", g_col_types)

    # Going further use the data frame created using "columns_cleaned_extrapolated" for further analysis
    # ...
    # ...

# END MAIN
