import pandas as pd
import numpy as np
import re
import os

# Inflation rate for years of 2015, 2016, and 2017.
# Source: http://www.in2013dollars.com/2017-dollars-in-2018?amount=50000&future_pct=0.025

INFLATION_2012_to_2018 = 1.1014
INFLATION_2013_to_2018 = 1.0855
INFLATION_2014_to_2018 = 1.0682
INFLATION_2015_to_2018 = 1.0669
INFLATION_2016_to_2018 = 1.0537
INFLATION_2017_to_2018 = 1.0317

PICKLE_PATH_15_18 = '/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H1B_15-18.pickle'
CSV_RAW_PATH_15_18 = '/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H1B_RAW_15-18.csv'
PICKLE_PATH_12_14 = '/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H1B_12-14.pickle'
PICKLE_PATH_NEWONLY = '/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H1B_15-18_new.pickle'

RELEVANT_COLUMNS = ['CASE_STATUS', 'EMPLOYER_CITY', 'EMPLOYER_STATE', 'FULL_TIME_POSITION', 'H1B_DEPENDENT',
                    'NAICS_CODE', 'PREVAILING_WAGE', 'PW_UNIT_OF_PAY', 'PW_SOURCE', 'SOC_CODE', 'VISA_CLASS',
                    'WAGE_RATE_OF_PAY_FROM', 'WAGE_UNIT_OF_PAY',
                    'WILLFUL_VIOLATOR', 'WORKSITE_CITY', 'WORKSITE_STATE','JOB_TITLE', 'EMPLOYER_NAME']


def load_H1B_df_15_18():
    if os.path.isfile(PICKLE_PATH_15_18):
        return pd.read_pickle(PICKLE_PATH_15_18)

    # df_15 = pd.read_pickle('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H-1B_15.pickle')
    # df_16 = pd.read_pickle('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H-1B_16.pickle')
    # df_17 = pd.read_pickle('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H-1B_17.pickle')
    # df_18 = pd.read_pickle('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H-1B_18.pickle')
    df_15 = pd.read_excel('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/PERM_DOL/H-1B_Disclosure_Data_FY15_Q4.xlsx')
    df_16 = pd.read_excel('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/PERM_DOL/H-1B_Disclosure_Data_FY16.xlsx')
    df_17 = pd.read_excel('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/PERM_DOL/H-1B_Disclosure_Data_FY17.xlsx')
    df_18 = pd.read_excel('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/PERM_DOL/H-1B_Disclosure_Data_FY2018_Q4_EOY.xlsx')
    # df_15.to_pickle("/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/PERM_DOL/H-1B_15.pickle")
    # df_16.to_pickle("/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/PERM_DOL/H-1B_16.pickle")
    # df_17.to_pickle("/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/PERM_DOL/H-1B_17.pickle")
    # df_18.to_pickle("/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/PERM_DOL/H-1B_18.pickle")

    df_18 = df_18.rename(columns={'NEW_CONCURRENT_EMP': 'NEW_CONCURRENT_EMPLOYMENT'})

    df15_wagerate = df_15['WAGE_RATE_OF_PAY'].str.split("-", n=1, expand=True)
    df_15['WAGE_RATE_OF_PAY_FROM'] = df15_wagerate[0].apply(lambda x: int(float(x)) if x.strip().isdigit() else np.nan)
    df_15['WAGE_RATE_OF_PAY_TO'] = df15_wagerate[1].apply(lambda x: int(float(x)) if x.strip().isdigit() else np.nan)

    df_15['EMPLOYER_ADDRESS'] = df_15['EMPLOYER_ADDRESS1'] + ',' + df_15['EMPLOYER_ADDRESS2']
    df_15.drop(columns=['EMPLOYER_ADDRESS1','EMPLOYER_ADDRESS2','PW_WAGE_LEVEL','WAGE_RATE_OF_PAY'], inplace=True)
    df_15 = df_15.rename(columns={'TOTAL WORKERS': 'TOTAL_WORKERS', 'WILLFUL VIOLATOR':'WILLFUL_VIOLATOR',
                                  'PW_WAGE_SOURCE_YEAR':'PW_SOURCE_YEAR', 'PW_WAGE_SOURCE_OTHER':'PW_SOURCE_OTHER',
                                  'PW_WAGE_SOURCE':'PW_SOURCE', 'H-1B_DEPENDENT': 'H1B_DEPENDENT',
                                  'NAIC_CODE':'NAICS_CODE'})

    df_16 = df_16.rename(columns={'PW_WAGE_SOURCE':'PW_SOURCE', 'H-1B_DEPENDENT': 'H1B_DEPENDENT', 'NAIC_CODE':'NAICS_CODE'})

    df_15['PREVAILING_WAGE'] = df_15['PREVAILING_WAGE'] * INFLATION_2015_to_2018
    df_15['WAGE_RATE_OF_PAY_FROM'] = df_15['WAGE_RATE_OF_PAY_FROM'] * INFLATION_2015_to_2018
    # df_15['WAGE_RATE_OF_PAY_TO'] = df_15['WAGE_RATE_OF_PAY_TO'] * INFLATION_2015_to_2018
    df_16['PREVAILING_WAGE'] = df_16['PREVAILING_WAGE'] * INFLATION_2016_to_2018
    df_16['WAGE_RATE_OF_PAY_FROM'] = df_16['WAGE_RATE_OF_PAY_FROM'] * INFLATION_2016_to_2018
    # df_16['WAGE_RATE_OF_PAY_TO'] = df_16['WAGE_RATE_OF_PAY_TO'] * INFLATION_2016_to_2018
    df_17['PREVAILING_WAGE'] = df_17['PREVAILING_WAGE'] * INFLATION_2017_to_2018
    df_17['WAGE_RATE_OF_PAY_FROM'] = df_17['WAGE_RATE_OF_PAY_FROM'] * INFLATION_2017_to_2018
    # df_17['WAGE_RATE_OF_PAY_TO'] = df_17['WAGE_RATE_OF_PAY_TO'] * INFLATION_2017_to_2018

    # df16_ignore = [c for c in df_16.columns if c not in df_15.columns]
    # df17_ignore = [c for c in df_17.columns if c not in df_15.columns]
    # df18_ignore = [c for c in df_18.columns if c not in df_15.columns]

    # df_16 = df_16.drop(columns=df16_ignore)
    # df_17 = df_17.drop(columns=df17_ignore)
    # df_18 = df_18.drop(columns=df18_ignore)

    df = pd.concat([df_15[RELEVANT_COLUMNS], df_16[RELEVANT_COLUMNS],
                    df_17[RELEVANT_COLUMNS], df_18[RELEVANT_COLUMNS]])
    # df = df.drop(['PW_WAGE_LEVEL','WAGE_RATE_OF_PAY_TO'])
    # df.to_csv('/tmp/H1B_data.csv', encoding='utf-8')
    # df.to_pickle(PICKLE_PATH_15_18)

    return df


def clean_up_df(df_new, df_old=None):
    def clean_up_SOC_CODE(code_str):
        try:
            #99% of samples go through here.
            parts = re.search(".?(\d{2})[\-,\.](\d{4}\.?.*)", code_str).groups()
            if len(parts[1]) < 4:
                print("ERROR: {}".format(code_str))
                return np.nan
            elif len(parts[1]) > 4:
                return str(parts[0])+"-"+str(int(np.round(float(parts[1]))))
            else:
                if len(parts[0]) == 2:
                    # this is a valid string format
                    return str(code_str)
                else:
                    print("ERROR: {}".format(code_str))
                    return np.nan
        except Exception as e:
            print("{}, {}".format(e, code_str))
        try:
            # handle cases without "-" but a valid 6-digit number
            if re.match("^\d{6}$", code_str):
                return str(code_str[:2]) + "-" + str(code_str[2:])
            elif re.match("^\d{6}\.\d+$", code_str):
                strValue = str(int(float(code_str)))
                return strValue[:2] + "-" + strValue[2:]

        except Exception as e:
            print("{}, {}".format(e, code_str))
        return np.nan

    def clean_up_NAICS_CODE(code_float):
        try:
            code_str = str(int(float(code_float)))
            if re.match("^\d{5,6}$", code_str):
                return str(int(float(code_str)))
            # elif re.match("^\d{5}$", code_str):
            #     return "0"+str(int(float(code_str)))

        except Exception as e:
            print("{}, {}".format(e, code_float))

        return np.nan

    def clean_up_CompanyName(name):
        try:
            return ''.join(filter(str.isalpha, str(name.encode('utf-8').strip()))).upper()
        except Exception as e:
            # print("NAME: {},   ::{}".format(name, e))
            return name

    df_new['EMPLOYER_NAME'] = df_new['EMPLOYER_NAME'].apply(clean_up_CompanyName)
    # Some samples have trailing ".00" character for SOC_CODE.  Remove them.
    df_new['SOC_CODE'] = df_new['SOC_CODE'].apply(clean_up_SOC_CODE)
    # Fill in SOC code with job title for samples with denied label
    df_new['NAICS_CODE'] = df_new['NAICS_CODE'].apply(clean_up_NAICS_CODE)
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['EMPLOYER_CITY'].isnull()), 'EMPLOYER_CITY'] = 'Unknown'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['EMPLOYER_STATE'].isnull()), 'EMPLOYER_STATE'] = 'Unknown'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['FULL_TIME_POSITION'].isnull()), 'FULL_TIME_POSITION'] = 'Y'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['H1B_DEPENDENT'].isnull()), 'H1B_DEPENDENT'] = 'N'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['PREVAILING_WAGE'].isnull()), 'PREVAILING_WAGE'] = df_new.PREVAILING_WAGE.mean()
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['PW_UNIT_OF_PAY'].isnull()), 'PW_UNIT_OF_PAY'] = 'Year'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['PW_SOURCE'].isnull()), 'PW_SOURCE'] = 'Other'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['VISA_CLASS'].isnull()), 'VISA_CLASS'] = 'H-1B'

    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['WAGE_UNIT_OF_PAY'].isnull()), 'WAGE_UNIT_OF_PAY'] = 'Year'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['WILLFUL_VIOLATOR'].isnull()), 'WILLFUL_VIOLATOR'] = 'N'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['WORKSITE_CITY'].isnull()), 'WORKSITE_CITY'] = 'Unknown'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['WORKSITE_STATE'].isnull()), 'WORKSITE_STATE'] = 'Unknown'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['JOB_TITLE'].isnull()), 'JOB_TITLE'] = 'Unknown'
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['EMPLOYER_NAME'].isnull()), 'EMPLOYER_NAME'] = 'Unknown'

    df_new['WAGE_LOWER_THAN_PW'] = df_new['PREVAILING_WAGE'] > df_new['WAGE_RATE_OF_PAY_FROM']

    # relabel CERTIFIED-WITHDRAWN to CERTIFIED.
    df_new.loc[(df_new['CASE_STATUS'] == 'CERTIFIED-WITHDRAWN'), 'CASE_STATUS'] = 'CERTIFIED'

    df_new = df_new[df_new['CASE_STATUS']!='WITHDRAWN']
    df_new = df_new.drop(columns=['WORKSITE_CITY', 'WAGE_RATE_OF_PAY_FROM', 'PW_UNIT_OF_PAY'])

    # Fill in SOC code with job title for samples with denied label
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['SOC_CODE'].isnull()), 'SOC_CODE'] = \
        df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['SOC_CODE'].isnull()), 'JOB_TITLE']
    df_new['NAICS_CODE'] = df_new['NAICS_CODE'].apply(clean_up_NAICS_CODE)
    # Fill in NAICS code with company name for samples with denied label
    df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['NAICS_CODE'].isnull()), 'NAICS_CODE'] = \
        df_new.loc[(df_new['CASE_STATUS'] == 'DENIED') & (df_new['NAICS_CODE'].isnull()), 'EMPLOYER_NAME']
    df_new.to_pickle(PICKLE_PATH_NEWONLY)
    df_new = df_new.dropna(how='any')
    return df_new

if __name__ == '__main__':

    df_old = load_H1B_df_12_14()
    df_new = load_H1B_df_15_18()
    df_final = clean_up_df(df_new, None)
