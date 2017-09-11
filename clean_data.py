import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def clean_data(filename):
    ''' Fill missing values, engineer features '''
    df = pd.read_csv(filename)
    df_model = df[['SalePrice', 'SalesID', 'YearMade', \
               'MachineHoursCurrentMeter', 'UsageBand', 'ProductSize', 'state', \
               'ProductGroupDesc', 'Enclosure', 'Hydraulics']]
    # drop year 1000
    df_model = df_model[df_model['YearMade'] > 1000]
    # change null values in  MachineHoursCurrentMeter
    df_model['MachineHoursCurrentMeter'].fillna(0.0, inplace=True)
    # drop Null values in Product Size
    df_model = df_model[df_model['ProductSize'].isnull() == False]
    # get dummy variables
    product_size_dummies = pd.get_dummies(df_model['ProductSize']).rename(columns = lambda x: 'ProductSize_' + str(x))
    prod_grp_dummies = pd.get_dummies(df_model['ProductGroupDesc']).rename(columns = lambda x: 'ProductGroupDesc_' + str(x))
    # concat dataframe with dummies and drop original column
    df_model = pd.concat([df_model, product_size_dummies, prod_grp_dummies], axis=1)
    df_model.drop(['ProductGroupDesc', 'ProductSize'], inplace=True)
    return df_model


def train_test_split(df):
    ''' Create the train/test split '''
    X = df['YearMade', 'MachineHoursCurrentMeter', 'ProductSize', 'ProductGroupDesc'].reshape(-1, 4)
    y = df['SalePrice'].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = clean_data('/data/Train.csv')
    print df.head()
