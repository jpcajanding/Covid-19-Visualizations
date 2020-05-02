import pandas as pd


def load_data(case, update=False):
    if update:
        data_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_' + case + '_global.csv'
        df = pd.read_csv(data_path)
        data = df.drop(columns=['Province/State', 'Lat', 'Long'])
        data = data.groupby('Country/Region').agg('sum').transpose()
        data.index = pd.to_datetime(data.index)
        data.to_pickle('datafiles/time_series_covid19_' + case + '_global.pkl')
    else:
        data_path = 'datafiles/time_series_covid19_' + case + '_global.pkl'
        data = pd.read_pickle(data_path)

    return data
