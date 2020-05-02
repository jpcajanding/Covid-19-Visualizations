import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from covid_load_data import load_data

update = True
confirmed = load_data('confirmed', update= update)
deaths = load_data('deaths', update= update)
recovered = load_data('recovered', update= update)

sns.set(style='darkgrid')

fig = plt.figure(figsize=(12,5))
ax = plt.axes()
ax.set_facecolor('#838c91')
plt.fill_between(confirmed.index.date, confirmed.sum(axis=1), confirmed.sum(axis=1) - (deaths.sum(axis=1) + recovered.sum(axis=1)),
                 label= 'Confirmed',
                 edgecolor='#026aa7',
                 facecolor='#bcd9ea')
plt.plot(confirmed.index.date, deaths.sum(axis=1), label= 'Deaths',
         linestyle= 'dotted',
         linewidth= 1,
         color= '#333333')
plt.plot(confirmed.index.date, recovered.sum(axis=1), label= 'Recoveries',
         linestyle= '--',
         linewidth= 1,
         color= '#222222')
plt.fill_between(confirmed.index.date, confirmed.sum(axis=1) - (deaths.sum(axis=1) + recovered.sum(axis=1)), 0,
                 label= 'Active Cases',
                 edgecolor= '#cf513d',
                 facecolor= '#efb3ab')
plt.xlim(confirmed.index[39], confirmed.index[-1])
ax.yaxis.set_major_locator(ticker.MultipleLocator(500000))
ax.set_yticks([confirmed.iloc[-1].sum(),
               deaths.iloc[-1].sum(),
               recovered.iloc[-1].sum(),
               confirmed.iloc[-1].sum() - (deaths.iloc[-1].sum() + recovered.iloc[-1].sum())],
              minor=True)
ax.yaxis.grid(True, which='minor', linestyle= 'dotted', linewidth= 1)
plt.title('CoVid-19 Cases Worldwide')
plt.legend(loc= 'upper left',
           fancybox= True,
           fontsize= 10)
plt.show()
