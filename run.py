from io import DEFAULT_BUFFER_SIZE
import pandas as pd


df = pd.read_csv('data/TAG Datensatz_20221001_20230429.csv')
df2 = pd.read_csv('data/tageswerte-20230430-195703.csv')

df2.info()

df3 = df.merge(df2, left_on=('time'), right_on='Datum')
df3['Datum']= pd.to_datetime(df3['Datum'])

#import seaborn as sns
#import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

df = df3

import numpy as np

# remove rows with missing values
df = df.dropna(subset=['t', 'Verbrauch'])

# calculate correlation coefficient and regression line
corr_coef = df['t'].corr(df['Verbrauch'])
coefficients = np.polyfit(df['t'], df['Verbrauch'], 3)
polynomial = np.poly1d(coefficients)

# add correlation coefficient and regression line to dataframe
df['corr_coef'] = corr_coef
df['reg_line'] = polynomial(df['t'])

# Weekly
df['Woche'] = pd.to_datetime(df['Datum']).dt.to_period('W')
df_agg_w = df.groupby('Woche').agg({'t': 'mean', 'Verbrauch': 'sum'}).reset_index()
df_agg_w.index=df_agg_w['Woche'].astype(str)
# calculate correlation coefficient and regression line AGG
corr_coef = df_agg_w['t'].corr(df_agg_w['Verbrauch'])
coefficients = np.polyfit(df_agg_w['t'], df_agg_w['Verbrauch'], 3)
polynomial = np.poly1d(coefficients)
# add correlation coefficient and regression line to dataframe agg
df_agg_w['corr_coef'] = corr_coef
df_agg_w['reg_line'] = polynomial(df_agg_w['t'])



# Monthly
df['Monat'] = pd.to_datetime(df['Datum']).dt.to_period('M')
df_agg = df.groupby('Monat').agg({'t': 'mean', 'Verbrauch': 'sum'}).reset_index()
# needed for bar chart not able to plot period
df_agg.index=df_agg['Monat'].astype(str)

# calculate correlation coefficient and regression line AGG
corr_coef = df_agg['t'].corr(df_agg['Verbrauch'])
coefficients = np.polyfit(df_agg['t'], df_agg['Verbrauch'], 3)
polynomial = np.poly1d(coefficients)
# add correlation coefficient and regression line to dataframe agg
df_agg['corr_coef'] = corr_coef
df_agg['reg_line'] = polynomial(df_agg['t'])





fig, (axes, urxn) = plt.subplots(2, 2, figsize=(18, 5))


ax5 = axes[0]
# weekly
ax5.bar(df_agg_w['Woche'].index, df_agg_w['Verbrauch'], color='blue')
ax5.set_xlabel('Datum')
ax5.set_ylabel(u'Verbrauch w\xf6 Summe')
ax5.set_title(u'W\xf6chentliche Daten')
ax6 = ax5.twinx()
ax6.plot(df_agg_w['Woche'].index, df_agg_w['t'], color='red')
ax6.set_ylabel('Temperatur Wien Durchschnitt', color='red')
ax6.tick_params(axis='y', labelcolor='red')
ax5.scatter(df_agg_w['Woche'].index, df_agg_w['reg_line'], color='green')



# plot t on first y-axis
ax1 = plt.subplot(2,1,2)
ax1.bar(df['Datum'], df['Verbrauch'], color='blue')
ax1.set_ylabel(u'Verbrauch t\xe4glich', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()

ax2.plot(df['Datum'], df['t'], color='red')
ax2.set_xlabel('Datum')
ax2.set_ylabel('Temperatur Wien', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# plot regression line on 
ax1.scatter(df['Datum'], df['reg_line'], color='green')



# add title and legend to first plot
ax1.set_title(u'T\xe4gliche Daten')
ax1.legend(['Korrelation'])

# add text box with correlation coefficient to first plot
textstr = f'Correlation Coefficient: {corr_coef:.2f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)







# plot aggregated data on second subplot
ax3 = plt.subplot(2, 3, 3)
ax3.bar(df_agg['Monat'].index, df_agg['Verbrauch'], color='blue')
ax3.set_xlabel('Datum')
ax3.set_ylabel('Verbrauch monatliche Summe')
ax3.set_title('Monatliche Daten')

ax4 = ax3.twinx()


ax4.plot(df_agg['Monat'].index, df_agg['t'], color='red')
ax4.set_ylabel('Temperatur Wien Durchschnitt', color='red')
ax4.tick_params(axis='y', labelcolor='red')


# plot regression line on second y-axis
ax3.scatter(df_agg['Monat'].index, df_agg['reg_line'], color='green')




# display the plot
plt.show()