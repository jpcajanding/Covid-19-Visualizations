import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mpl_dates
import numpy as np
from sklearn.linear_model import LinearRegression

# read data from pre-stored csv
df = pd.read_csv('datafiles/doh/DOH COVID Data Drop_ 20200501 - 05 Case Information.csv',
                 parse_dates=['DateRepConf', 'DateRecover', 'DateDied', 'DateRepRem'])

print(df.notna().sum())

print(df.columns)


# data cleaning: output a data frame with Confirmed, deaths and recoveries per day
n = 5
day_case = pd.DataFrame(df['DateRepConf'])
day_case = pd.DataFrame(day_case.groupby('DateRepConf')['DateRepConf'].count().reset_index(name='Confirmed'))
df_removed = df[df['RemovalType'].notna()]
df_removed = df_removed[['RemovalType', 'DateRepRem']]
df_removed = pd.DataFrame(df_removed.groupby(['DateRepRem','RemovalType']).size().reset_index(name='count'))
df_deaths = df_removed[df_removed['RemovalType'] == 'Died'].drop(columns='RemovalType').rename(columns= {'count':'Death'})
df_recovered = df_removed[df_removed['RemovalType'] == 'Recovered'].drop(columns='RemovalType').rename(columns= {'count':'Recovered'})
del df_removed
cases_time_series = pd.merge(day_case, df_deaths, how='left', left_on='DateRepConf', right_on='DateRepRem')
cases_time_series = pd.merge(cases_time_series, df_recovered, how='left', left_on='DateRepConf', right_on='DateRepRem')
cases_time_series = cases_time_series.drop(columns=['DateRepRem_x', 'DateRepRem_y']).set_index('DateRepConf').fillna(0)
del[day_case,df_deaths,df_recovered]
max_bars = cases_time_series.nlargest(n, ['Confirmed'])


# plot the number of cases per day
fig = plt.figure(figsize=(10, 4))
ax = plt.axes()
ax.set_facecolor('#F4EEFF')
plt.title('CoVid-19 Cases in the Philippines')
plt.fill_between(cases_time_series.index.date, cases_time_series['Confirmed'].cumsum(),
                 (cases_time_series['Confirmed'].cumsum()-(cases_time_series['Death'].cumsum()+cases_time_series['Recovered'].cumsum())),
                 label= 'Confirmed',
                 color= '#000839')
plt.annotate('%0.f' % cases_time_series['Confirmed'].cumsum()[-1],
             xy=(1, cases_time_series['Confirmed'].cumsum()[-1]),
             xytext=(1, 0),
             xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize= 7)
plt.fill_between(cases_time_series.index.date,
                 (cases_time_series['Confirmed'].cumsum() -
                  (cases_time_series['Death'].cumsum() + cases_time_series['Recovered'].cumsum())), 0,
                 label= 'Active',
                 color= '#005082')
plt.annotate('%0.f' % (cases_time_series['Confirmed'].cumsum()-(cases_time_series['Death'].cumsum()+cases_time_series['Recovered'].cumsum()))[-1],
             xy=(1, (cases_time_series['Confirmed'].cumsum()-(cases_time_series['Death'].cumsum()+cases_time_series['Recovered'].cumsum()))[-1]),
             xytext=(1, 0),
             xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize= 7)
ax.bar(cases_time_series.index.date, cases_time_series['Confirmed'],
       label='Cases per Day',
       color='#FFA41B')
for i in range(n):
    plt.annotate('%0.f' % max_bars['Confirmed'][i],
                 xy=(max_bars.index[i], max_bars['Confirmed'][i]),
                 xytext=(-6.5,1),
                 xycoords=('data', 'data'), textcoords='offset points', fontsize=7, color='#FFFFFF')
plt.xlim(cases_time_series.index[12],cases_time_series.index[-1])
plt.xticks([cases_time_series.index[20], cases_time_series.index[-1]])
plt.ylim(0,np.ceil((cases_time_series['Confirmed'].cumsum()[-1])/1000)*1000)
ax.yaxis.set_major_locator(ticker.MultipleLocator((np.ceil((cases_time_series['Confirmed'].cumsum()[-1])/1000)*1000)/3))
# ax.set_yticks([(cases_time_series['Confirmed'].cumsum()-(cases_time_series['Death'].cumsum()+cases_time_series['Recovered'].cumsum()))[-1],
#                cases_time_series['Confirmed'].cumsum()[-1]],
#               minor=True)
# ax.yaxis.grid(True, which='minor', linestyle='dotted', linewidth=1)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mpl_dates.DateFormatter('%b %d'))  # format dates as Month-Day eg Mar 01
plt.xlim(pd.Timestamp('2020-03-01'), cases_time_series.index[-1])
plt.xticks(pd.date_range(pd.Timestamp('2020-03-01'), cases_time_series.index[-1] + pd.DateOffset(1),
                         freq='15D', closed='right'))
plt.legend(loc='upper center',
           ncol=3,
           fontsize=10,
           frameon=False)
# plt.show()

#plot the number of cases per day with rolling mean and death and recoveries
fig2 = plt.figure(figsize=(10,4))
ax = plt.axes()
plt.title('Confirmed Cases per Day')
ax.set_facecolor('#F4EEFF')

ax.bar(cases_time_series.index.date, cases_time_series['Confirmed'],
              # label= 'Cases per Day',
              color='#37AB85')
for i in range(n):
    plt.annotate('%0.f' % max_bars['Confirmed'][i],
                 xy=(max_bars.index[i], max_bars['Confirmed'][i]),
                 xytext=(-6.5, 1),
                 xycoords=('data', 'data'), textcoords='offset points', fontsize=7)

plt.plot(cases_time_series.index.date, cases_time_series['Confirmed'].rolling(window= 7).mean(),
         linewidth= 2,
         label= '7-Day Moving Average',
         color= '#FFB052')
plt.plot(cases_time_series.index.date, cases_time_series['Death'].cumsum(),
         linestyle= 'dotted',
         linewidth= 1,
         color= '#000000',
         label= 'Deaths')
plt.annotate('%0.f' % cases_time_series['Death'].cumsum()[-1],
             xy=(cases_time_series.index[-1], cases_time_series['Death'].cumsum()[-1]),
             xytext=(-6.5, 1),
             xycoords=('data', 'data'), textcoords='offset points', fontsize= 7)
plt.plot(cases_time_series.index.date, cases_time_series['Recovered'].cumsum(),
         linestyle= '--',
         linewidth= 1,
         color= '#000000',
         label= 'Recoveries')
plt.annotate('%0.f' % cases_time_series['Recovered'].cumsum()[-1],
             xy=(cases_time_series.index[-1], cases_time_series['Recovered'].cumsum()[-1]),
             xytext=(-6.5, 1),
             xycoords=('data', 'data'), textcoords='offset points', fontsize= 7)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mpl_dates.DateFormatter('%b %d'))  # format dates as Month-Day eg Mar 01
plt.xlim(pd.Timestamp('2020-03-01'), cases_time_series.index[-1] + pd.DateOffset(1))
plt.xticks(pd.date_range(pd.Timestamp('2020-03-01'), cases_time_series.index[-1] + pd.DateOffset(1),
                         freq='15D', closed='right'))
plt.legend(loc='upper left',
           fancybox=True,
           fontsize=10,
           facecolor='#FFFFFF')
# plt.show()
del(n, max_bars)

# get oldest and youngest confirmed, died and recovery

print('Oldest Patient is {0} year old {1}, confirmed on {2}.'.format(df['Age'].loc[df['Age'].idxmax()],
                                                                     df['Sex'].loc[df['Age'].idxmax()],
                                                                     df['DateRepConf'].loc[df['Age'].idxmax()].date()))
print('Youngest Patient is {0} year old {1}, confirmed on {2}.'.format(df['Age'].loc[df['Age'].idxmin()],
                                                                       df['Sex'].loc[df['Age'].idxmin()],
                                                                       df['DateRepConf'].loc[df['Age'].idxmin()].date()))
print('Mean age of cases is {0} while median is {1}. Mode is {2}'.format(round(df['Age'].mean()),
                                                                         df['Age'].median(),
                                                                         df['Age'].mode()))
df_temp = df.loc[df['RemovalType'] == 'Died']
print('Mean age of deaths is {0} while median is {1}. Mode is {2}'.format(round(df_temp['Age'].mean()),
                                                                          df_temp['Age'].median(),
                                                                          df_temp['Age'].mode()))
df_temp = df_temp[df_temp['DateDied'].notna()]
print('Mean days to die is {0} days.'.format((df_temp['DateDied'] - df_temp['DateRepConf']).mean().days))
print('Mean days to report death is {0} days.'.format((df_temp['DateRepRem'] - df_temp['DateDied']).mean().days))
print('Oldest Death is {0} year old {1}, confirmed on {2}.'.format(df_temp['Age'].loc[df_temp['Age'].idxmax()],
                                                                   df_temp['Sex'].loc[df_temp['Age'].idxmax()],
                                                                   df_temp['DateRepConf'].loc[df_temp['Age'].idxmax()].date()))
print('Youngest Death is {0} year old {1}, confirmed on {2}.'.format(df_temp['Age'].loc[df_temp['Age'].idxmin()],
                                                                     df_temp['Sex'].loc[df_temp['Age'].idxmin()],
                                                                     df_temp['DateRepConf'].loc[df_temp['Age'].idxmin()].date()))

df_temp = df.loc[df['RemovalType'] == 'Recovered']
print('Mean age of recoveries is {0} while median is {1}. Mode is {2}'.format(round(df_temp['Age'].mean()),
                                                                              df_temp['Age'].median(),
                                                                              df_temp['Age'].mode()))
df_temp = df_temp[df_temp['DateRecover'].notna()]
print('Mean days to recover is {0} days.'.format((df_temp['DateRecover'] - df_temp['DateRepConf']).mean().days))
print('Mean days to report recovery is {0} days.'.format((df_temp['DateRepRem'] - df_temp['DateRecover']).mean().days))
print('Oldest Recovery is {0} year old {1}, confirmed on {2}.'.format(df_temp['Age'].loc[df_temp['Age'].idxmax()],
                                                                      df_temp['Sex'].loc[df_temp['Age'].idxmax()],
                                                                      df_temp['DateRepConf'].loc[df_temp['Age'].idxmax()].date()))
print('Youngest Recovery is {0} year old {1}, confirmed on {2}.'.format(df_temp['Age'].loc[df_temp['Age'].idxmin()],
                                                                        df_temp['Sex'].loc[df_temp['Age'].idxmin()],
                                                                        df_temp['DateRepConf'].loc[df_temp['Age'].idxmin()].date()))
del df_temp

case_age = pd.DataFrame(df.groupby('AgeGroup')['AgeGroup'].count().reset_index(name='Confirmed'))
df_removed = df[df['RemovalType'].notna()]
df_removed = df_removed[['RemovalType', 'AgeGroup']]
df_removed = pd.DataFrame(df_removed.groupby(['AgeGroup', 'RemovalType']).size().reset_index(name='count'))
df_deaths = df_removed[df_removed['RemovalType'] == 'Died'].drop(columns='RemovalType').rename(columns={'count': 'Death'})
df_recovered = df_removed[df_removed['RemovalType'] == 'Recovered'].drop(columns='RemovalType').rename(columns={'count': 'Recovered'})
del df_removed

case_age = pd.merge(case_age, df_deaths, how='left', left_on='AgeGroup', right_on='AgeGroup')
case_age = pd.merge(case_age, df_recovered, how='left', left_on='AgeGroup', right_on='AgeGroup')
target_row = case_age[case_age['AgeGroup'] == '5 to 9'].index.values
idx = [0, target_row] + [i for i in range(1, len(case_age)) if i != target_row]
case_age = case_age.iloc[idx].set_index('AgeGroup').fillna(0)
del[df_deaths, df_recovered, idx, target_row]

# """""
# fig4 = plt.figure(figsize=(12, 5))
# ax_count = plt.axes()
# plt.title('CoVid-19 Cases per Age')
# ax_count.set_facecolor('#F4EEFF')
#
# ax_count.bar(case_age.index, case_age['Confirmed'],
#          label='Confirmed')
# ax_count.set_xticklabels(case_age.columns, rotation='45')
# ax_percent = ax_count.twinx()
#
# ax_percent.plot(case_age.index, (case_age['Death']/case_age['Death'].sum())*100,
#                 marker= 'o',
#                 label= 'Death',
#                 color= '#37AB85')
# ax_percent.plot(case_age.index, (case_age['Recovered']/case_age['Recovered'].sum())*100,
#                 marker='o',
#                 label= 'Recovered',
#                 color= '#FFB052')
# ax_count.set_xticks(case_age.index)
# ax_percent.set_xticks(case_age.index)
# ax_count.set_xticklabels(case_age.index, rotation='45')
# ax_percent.set_xticklabels(case_age.index, rotation='45')
# # plt.grid(color='r', linestyle='-', linewidth=1,axis='x')
# plt.legend()
# plt.show()
# """""

case_age_dec = (case_age + case_age.shift(-1))[:-2:2].append(case_age.iloc[-1]).reset_index()
case_ages_labels = ['0+', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80+']

fig3 = plt.figure(figsize=(10, 4))
ax_count = plt.axes()
plt.title('CoVid-19 Cases per Age')
ax_count.set_facecolor('#F4EEFF')

ax_count.bar(case_age_dec.index, case_age_dec['Confirmed'],
             label='Confirmed',
             color='#2AB77B')

for i in range(len(case_age_dec)):
    ax_count.annotate('%0.f' % case_age_dec['Confirmed'][i],
                      xy=(case_age_dec.index[i], case_age_dec['Confirmed'][i]),
                      xytext=(-6.5, 1),
                      xycoords=('data', 'data'),
                      textcoords='offset points',
                      fontsize=10)

ax_count.yaxis.set_visible(False)

ax_percent = ax_count.twinx()

ax_percent.plot(case_age_dec.index, (case_age_dec['Death']/case_age_dec['Death'].sum())*100,
                marker='o',
                label='Death',
                color='#FFCF6E')
ax_percent.plot(case_age_dec.index, (case_age_dec['Recovered']/case_age_dec['Recovered'].sum())*100,
                marker='o',
                label='Recovered',
                color='#DF60A7')
ax_percent.set_xticks(case_age_dec.index)
ax_percent.set_xticklabels(case_ages_labels)
ax_percent.set_ylim(0, 50)
ax_percent.yaxis.set_ticks_position('left')
ax_percent.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax_percent.set_ylabel('Percent')
ax_percent.set_xlabel('Age Group')
plt.legend()
# plt.show()
del(ax_percent, ax_count)
#

# plt.close('all')
fig4 = plt.figure(figsize=(6, 4))
ax = plt.axes()
plt.title('Resolved Cases per Age')
ax.set_facecolor('#F4EEFF')

div_line = (case_age_dec['Death']/(case_age_dec['Death'] + case_age_dec['Recovered']) * 100)
ax.plot(case_age_dec.index, div_line, color='#000000')
ax.fill_between(case_age_dec.index, 100, div_line, alpha=0.4)
ax.fill_between(case_age_dec.index, div_line, 0, alpha=0.4)
ax.set_ylim(0, 100)
ax.set_xticklabels(case_ages_labels, rotation='45')
ax.set_xticks(case_age_dec.index)
ax.set_ylabel('Percent')
ax.set_xlabel('Age Group')
# ax.xaxis.set_major_locator(case_age.index)
ax.set_xlim(case_age_dec.index[::len(case_age_dec) - 1])
plt.tight_layout()
# plt.show()
del case_age, case_ages_labels, case_age_dec, div_line

daily_age = df[['DateRepConf', 'AgeGroup']]
daily_age = pd.DataFrame(daily_age.groupby(['DateRepConf', 'AgeGroup']).size().reset_index(name='count'))
daily_age = daily_age.pivot(index='DateRepConf', columns='AgeGroup', values='count').rename_axis(None).fillna(0)
print(daily_age.columns)
daily_age_chart = pd.DataFrame(daily_age[['60 to 64', '65 to 69']].sum(axis=1), columns=['60s'])
daily_age_chart['70+'] = daily_age[['70 to 74', '75 to 79', '80+']].sum(axis=1)
daily_age_chart['25 to 34'] = daily_age[['25 to 29', '30 to 34']].sum(axis=1)
daily_age_chart['50s'] = daily_age[['50 to 54', '55 to 59']].sum(axis=1)


fig5 = plt.figure(figsize=(10, 4))
ax = plt.axes()
plt.title('Daily Cases per Age Group')
ax.set_facecolor('#F4EEFF')

ax.plot(daily_age_chart.index.date, daily_age_chart.cumsum())
ax.set_xlim(pd.Timestamp('2020-03-01'), daily_age_chart.index[-1])
ax.set_xticks([pd.Timestamp('2020-03-01'), pd.Timestamp('2020-03-31'), daily_age_chart.index[-1]])
ax.xaxis_date()
ax.xaxis.set_major_formatter(mpl_dates.DateFormatter('%b %d'))

plt.legend(daily_age_chart.columns, loc='upper left')
# plt.show()
del(cases_time_series, daily_age, daily_age_chart)


def case_status(start_age, end_age):
    case_df = df[df['Age'].isin(np.arange(start_age, end_age + 1))]
    return case_df.groupby('HealthStatus')['HealthStatus'].count().drop(index=['Died', 'Recovered'])


case_25to34 = case_status(25, 34)
case_50s = case_status(50, 59)
case_25to34 = case_25to34.reindex(case_50s.index).fillna(0)


def my_autopct(pct):
    return ('%.2f' % pct) if pct > 0 else ''


def get_new_labels(sizes, labels):
    new_labels = [label if size > 1 else '' for size, label in zip(sizes, labels)]
    return new_labels


plt.close('all')
fig6, (ax25, ax50) = plt.subplots(1, 2, subplot_kw={'aspect': 'equal'}, figsize=(10, 4))
fig6.suptitle('Health Status for Age Groups 25 to 34 and 50s')
ax25.set_facecolor('#F4EEFF')

ax25.pie(case_25to34, autopct=my_autopct,
         labels=get_new_labels(case_25to34, case_25to34.index))
ax25.text(0.5, 0, "25 to 34", size=10, ha="center", transform=ax25.transAxes)

ax50.pie(case_50s, labels=case_50s.index, autopct=my_autopct)
ax50.text(0.5, 0, "50s", size=10, ha="center",  transform=ax50.transAxes)
# plt.show()
del(case_25to34, case_50s)

df = pd.read_csv('datafiles/doh/DOH COVID Data Drop_ 20200501 - 08 Testing Aggregates.csv',
                 parse_dates=['Date'], thousands=',')

print(df.notna().sum())

print(df.columns)
print(df.head(1))
df.drop(list(df.filter(regex="%")), axis=1, inplace=True)  # drop columns with % on their name (useless)

columns = [col for col in df.columns if col not in ['Name of Health Facility/Laboratory', 'Abbrev of Health Facility',
                                                    'Date']]
for column in columns:
    df[column] = pd.to_numeric(df[column], errors='coerce', downcast='float')
# df = df.apply(pd.to_numeric, errors='ignore', downcast='float')
test_dates = df.groupby(['Date']).sum() #.reset_index()
test_daily = (test_dates - test_dates.shift(1))[1:]


def regress_line_time(y):
    x = np.arange(len(y)).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x, y)  # perform linear regression
    y_pred = linear_regressor.predict(x)  # make predictions
    return y_pred

plt.close('all')
fig7 = plt.figure(figsize=(10, 4))
ax_unique = plt.axes()
plt.title('Unique People Tested per day')
ax_unique.set_facecolor('#F0EFE7')

ax_unique.plot(test_daily.index, test_daily['UNIQUE INDIVIDUALS TESTED'], color='#264d59')
regress_line = regress_line_time(test_daily['UNIQUE INDIVIDUALS TESTED'].values.reshape(-1, 1))
ax_unique.plot(test_daily.index, regress_line, color='#d46c4e')
ax_unique.xaxis.set_major_formatter(mpl_dates.DateFormatter('%b %d'))  # format dates as Month-Day eg Mar 01
ax_unique.set_xticks([test_daily.index[0], test_daily.index[-1]])
ax_unique.set_ylim(0, 8500)
ax_unique.set_yticks([0, 4000, 8000])
ax_unique.grid(axis='y', color='#dcd6cd', linestyle='dotted', linewidth=1)
plt.show()

