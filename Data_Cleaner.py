import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper

# %%

weather_data = pd.read_csv("Weather_Data.csv", header=0, sep=',')
weather_data.dropna(axis=0,inplace=True)

d = weather_data['Day']
m =  weather_data['Month']
y = weather_data['Year']
h = weather_data['Hour']-1
weather_data['timestamp'] = pd.to_datetime(d.astype(str)+'-'+ m.astype(str) +'-'+ y.astype(str) +' '+ h.astype(str) + ':00:00',format='%d-%m-%Y %H:%M:%S')
weather_data.drop(columns=['Day', 'Month', 'Year', 'Hour', 'Minute'], inplace=True)

weather_data.columns = ['temperature','wind_speed','humidity','timestamp']
cols = list(weather_data.columns)
a, b = cols.index('temperature'), cols.index('timestamp')
cols[b], cols[a] = cols[a], cols[b]
weather_data = weather_data[cols]
weather_data = weather_data.set_index(['timestamp']).sort_index()

# %%
energy_data = pd.read_csv('Energy_Data.csv',header=0,sep=',')
energy_data['timestamp_dt'] = pd.to_datetime(energy_data['timestamp_dt'], format='%d/%m/%Y %H:%M')
energy_data.dropna(axis=0,inplace=True)
energy_data.columns = ['timestamp','energy_usage']
energy_data = energy_data.set_index(['timestamp']).sort_index()

data = weather_data.merge(right=energy_data, how='right', left_index=True, right_index=True)
print(data)

# %%
source = ColumnDataSource(data={
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "temperature": data.temperature,
    })

TOOLTIPS = [
    ('Time stamp','@ts_str'),
    ('Temperature', '@temperature')
    ]

p = figure(title='temperature for 2022', x_axis_label='Time Stamp', y_axis_label='Temperature', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','temperature',source=source,color='red')
show(p)

# %%
source = ColumnDataSource(data={
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "wind_speed": data.wind_speed,
    })

TOOLTIPS = [
    ('Time stamp','@ts_str'),
    ('Wind Speed', '@wind_speed')
    ]

p = figure(title='Wind speed for 2022', x_axis_label='Time Stamp', y_axis_label='Wind speed', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','wind_speed',source=source,color='green')
show(p)

# %%
source = ColumnDataSource(data={
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "humidity": data.humidity,
    })

TOOLTIPS = [
    ('Time stamp','@ts_str'),
    ('Humidity', '@humidity')
    ]

p = figure(title='Humidity for 2022', x_axis_label='Time Stamp', y_axis_label='Humidity', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','humidity',source=source,color='blue')
show(p)

# %%
source = ColumnDataSource(data={
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "energy_usage": data.energy_usage,
    })

TOOLTIPS = [
    ('Time stamp','@ts_str'),
    ('Energy usage', '@energy_usage{int}')
    ]

p = figure(title='Energy usage for 2022', x_axis_label='Time Stamp', y_axis_label='Energy usage', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_usage',source=source,color='gold')
show(p)
# %%
corr_matrix = data.corr()

plot = figure(title='Correlation matrix',
              x_range=list(corr_matrix.columns),
              y_range=list(corr_matrix.columns),
              toolbar_location=None,
              tools="",
              x_axis_location="below")

colors = ['#066201','#D3BBB1','#A5725D','#7C2400']
mapper = LinearColorMapper(palette=colors, low=-1, 
                           high=1)

data_dict = {
    "col1": [],
    "col2": [],
    "value": [],
}
for row in corr_matrix.index:
    for col in corr_matrix.columns:
        data_dict["col1"].append(row)
        data_dict["col2"].append(col)
        data_dict["value"].append(corr_matrix.loc[row, col])   
        
df_data = pd.DataFrame(data_dict)

plot.rect(source=ColumnDataSource(df_data),
          x="col1",
          y="col2", 
          width=1, height=1, line_color=None,
          fill_color=transform('value',mapper))

color_bar = ColorBar(color_mapper=mapper, 
                     location=(0, 0),
                     ticker=BasicTicker(desired_num_ticks=len(colors)))
plot.add_layout(color_bar, 'right')


show(plot)
# %%
source = ColumnDataSource(data={
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "temperature": data.temperature,
        "energy_usage": data.energy_usage,
    })

TOOLTIPS = [
    ('Time stamp','@ts_str'),
    ('Temperature','@temperature'),
    ('Energy usage', '@energy_usage{int}')
    ]

p = figure(title='Energy usage against temperature 2022', x_axis_label='Temperature', y_axis_label='Energy usage',width=1800,height=900,tooltips=TOOLTIPS)
p.scatter('temperature','energy_usage',source=source,color='purple')
show(p)

# %%
df_forecast = pd.DataFrame(index=pd.date_range(start=pd.Timestamp("2023-01-01 00:00:00"), end=pd.Timestamp("2023-12-31 23:00:00"), freq="1H"))

data_subset = data.loc[~data.index.day.isin([5,6]), :]
df_stats = data_subset.groupby(by=[data_subset.index.month, data_subset.index.dayofweek, data_subset.index.hour]).aggregate({
    "energy_usage": [np.mean, "count"]
})
df_stats.columns = ['energy_forecast', 'number_of_rows']

df_forecast['month'] = df_forecast.index.month
df_forecast['dayofweek'] = df_forecast.index.dayofweek
df_forecast['hour'] = df_forecast.index.hour

df_forecast = df_forecast.merge(right=df_stats.loc[:, ['energy_forecast']], how="left", left_on = ['month','dayofweek','hour'], right_index=True)

data_sub = data.loc[data.index.day.isin([5,6]), :]
df_stats = data_sub.groupby(by=[data_sub.index.day]).aggregate({
    "energy_usage": [np.mean]
})

df_forecast.loc[df_forecast.index.day.isin([5]), 'energy_forecast'] = df_stats.loc[5,'energy_usage'].item()
df_forecast.loc[df_forecast.index.day.isin([6]), 'energy_forecast'] = df_stats.loc[6,'energy_usage'].item()

df_forecast.drop(columns=['month', 'dayofweek', 'hour'], inplace=True)

# %%
source = ColumnDataSource(data={
        "ts_str": df_forecast.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": df_forecast.index,
        "energy_forecast": df_forecast.energy_forecast,
    })

TOOLTIPS = [
    ('Time stamp','@ts_str'),
    ('Energy forecast', '@energy_forecast{int}')
    ]

p = figure(title='Energy forecast for 2023', x_axis_label='Time Stamp', y_axis_label='Energy forecast', x_axis_type='datetime',y_range=[0,3000],width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_forecast',source=source,color='brown')
show(p)

# %%
weather_data2 = pd.read_csv("weather_data2.csv", header=0, sep=',')
weather_data2.dropna(axis=0,inplace=True)

d = weather_data2['Day']
m =  weather_data2['Month']
y = weather_data2['Year']
h = weather_data2['Hour']-1
weather_data2['timestamp'] = pd.to_datetime(d.astype(str)+'-'+ m.astype(str) +'-'+ y.astype(str) +' '+ h.astype(str) + ':00:00',format='%d-%m-%Y %H:%M:%S')
weather_data2.drop(columns=['Day', 'Month', 'Year', 'Hour', 'Minute'], inplace=True)

weather_data2.columns = ['temperature','wind_speed','humidity','timestamp']
cols = list(weather_data2.columns)
a, b = cols.index('temperature'), cols.index('timestamp')
cols[b], cols[a] = cols[a], cols[b]
weather_data2 = weather_data2[cols]
weather_data2 = weather_data2.set_index(['timestamp']).sort_index()

# %%
energy_data2 = pd.read_csv('energy_data2.csv',header=0,sep=',')
energy_data2['timestamp_dt'] = pd.to_datetime(energy_data2['timestamp_dt'], format='%d/%m/%Y %H:%M')
energy_data2.dropna(axis=0,inplace=True)
energy_data2.columns = ['timestamp','energy_usage']
energy_data2 = energy_data2.set_index(['timestamp']).sort_index()

data2 = weather_data2.merge(right=energy_data2, how='right', left_index=True, right_index=True)
print(data2)
# %%
source = ColumnDataSource(data={
        "ts_str": data2.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data2.index,
        "energy_usage": data2.energy_usage,
    })

TOOLTIPS = [
    ('Time stamp','@ts_str'),
    ('Energy usage', '@energy_usage{int}')
    ]

p = figure(title='Energy usage for 2023', x_axis_label='Time Stamp', y_axis_label='Energy usage', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_usage',source=source,color='blue')
show(p)

# %%
source = ColumnDataSource(data={
        "series_name": ["Energy usage 2022"] * data.shape[0],
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "energy_usage": data.energy_usage,
    })
source2 = ColumnDataSource(data={
        "series_name": ["Energy usage 2023"]* data2.shape[0],
        "ts_str": data2.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data2.index,
        "energy_usage": data2.energy_usage,
    })
source3 = ColumnDataSource(data={
        "series_name": ["Energy forecast"]* df_forecast.shape[0],
        "ts_str": df_forecast.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": df_forecast.index,
        "energy_usage": df_forecast.energy_forecast,
    })

TOOLTIPS = [
    ('Series Name', '@series_name'),
    ('Time stamp','@ts_str'),
    ('Energy usage', '@energy_usage{int}')
    ]

p = figure(title='Energy usage and prediction 2022-2023', x_axis_label='Time Stamp', y_axis_label='Energy usage', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_usage',source=source,color='gold',legend_label='Energy usage 2022')
p.line('ts','energy_usage',source=source2, legend_label='Energy usage 2023')
p.line('ts','energy_usage',source=source3,color='grey',legend_label = 'Energy forecast')
p.legend.location='top_left'
p.legend.click_policy="hide"

show(p)

# %%
"""from statsmodels.tsa.seasonal import seasonal_decompose
decompose_data = seasonal_decompose(data.energy_usage, model="additive",period=365)
p = figure(title='decomposed energy usage 2022',x_axis_label='Time stamp',y_axis_label='Energy usage',x_axis_type='datetime',width=1800,height=900)
p.line(decompose_data)
show(p)

# %%
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(data.energy_usage, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)
    
# %%
rolling_mean = data.energy_usage.rolling(window = 12).mean()
data['rolling_mean_diff'] = rolling_mean - rolling_mean.shift()
ax1 = plt.subplot()
data['rolling_mean_diff'].plot(title='after rolling mean & differencing');
ax2 = plt.subplot()
data.plot(title='original');

# %%
dftest = adfuller(data['rolling_mean_diff'].dropna(), autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
  print("\t",key, ": ", val)
  
# %%
from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(data['energy_usage'],order=(1,1,1))
history=model.fit()
history.summary()

# %%
data['forecast']=data.energy_usage.predict(start=90,end=103,dynamic=True)
data[['energy_usage','forecast']].plot(figsize=(12,8))

# %%
data_sarimax_fitting = data.loc[:, ['energy_usage']].copy()
data_sarimax_fitting = data_sarimax_fitting.asfreq("1H")
data_sarimax_fitting = data_sarimax_fitting.fillna(method = 'bfill',limit=3).fillna(method='ffill',limit=3)

from pmdarima import auto_arima  
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(data_sarimax_fitting.energy_usage,order=auto_arima(data_sarimax_fitting['energy_usage']), seasonal_order=auto_arima(data_sarimax_fitting['energy_usage'],seasonal=True,m=24))
results=model.fit()

data["energy_prediction"] = results.predict(start=0,end=data.shape[0]-1,dynamic=True)

source = ColumnDataSource(data={
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "energy_forecast": data.energy_prediction,
    })

TOOLTIPS = [
    ('Time stamp','@ts_str'),
    ('Energy forecast', '@energy_forecast{int}')
    ]

p = figure(title='Energy forecast for 2023', x_axis_label='Time Stamp', y_axis_label='Energy forecast', x_axis_type='datetime',y_range=[0,3000],width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_forecast',source=source,color='brown')
show(p)


# %%
source = ColumnDataSource(data={
        "series_name": ["Energy usage 2022"] * data.shape[0],
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "energy_usage": data.energy_usage,
    })
source2 = ColumnDataSource(data={
        "series_name": ["Energy usage 2023"]* data2.shape[0],
        "ts_str": data2.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data2.index,
        "energy_usage": data2.energy_usage,
    })
source3 = ColumnDataSource(data={
        "series_name": ["Energy rules forecast"]* df_forecast.shape[0],
        "ts_str": df_forecast.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": df_forecast.index,
        "energy_usage": df_forecast.energy_forecast,
    })
source4 = ColumnDataSource(data={
        "series_name": ["Energy sarimax forecast"]* data_prediction.shape[0],
        "ts_str": data_prediction.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data_prediction.index,
        "energy_usage": data_prediction.energy_usage,
    })


TOOLTIPS = [
    ('Series Name', '@series_name'),
    ('Time stamp','@ts_str'),
    ('Energy usage', '@energy_usage{int}')
    ]

p = figure(title='Energy usage and prediction 2022-2023', x_axis_label='Time Stamp', y_axis_label='Energy usage', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_usage',source=source,color='gold',legend_label='Energy usage 2022')
p.line('ts','energy_usage',source=source2, legend_label='Energy usage 2023')
p.line('ts','energy_usage',source=source3,color='grey',legend_label = 'Energy rules forecast')
p.line('ts','energy_usage',source=source4,color='black',legend_label = 'Energy sarimax forecast')
p.legend.location='top_left'
p.legend.click_policy="hide"

show(p) """

# %%
df_compare = df_forecast.loc[df_forecast.index <= pd.Timestamp("2023-09-21 13:00:00"), :].copy()
df_compare = df_compare.merge(right=data2, how='inner', left_index=True, right_index=True)
df_compare.drop(columns=['wind_speed','temperature','humidity'],inplace=True)

import math
MSE = np.square(np.subtract(df_compare.energy_usage,df_compare.energy_forecast)).mean()
RMSE = math.sqrt(MSE)

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

MAE = mae(df_compare.energy_usage, df_compare.energy_forecast)

MBE = np.mean(df_compare.energy_forecast - df_compare.energy_usage)

CVRMSE=(RMSE/np.mean(df_compare.energy_usage))*100
CVMAE=(MAE/np.mean(df_compare.energy_usage))*100
CVMBE=(MBE/np.mean(df_compare.energy_usage))*100

# %%
rest = data.copy()
rest.drop(columns=['energy_usage'],inplace=True)


rest['humidity_sqrt']= rest.humidity**0.5
rest['wnd_sqrt']= rest.wind_speed**0.5
rest['temp_sqd'] = rest.temperature ** 2


reg = LinearRegression().fit(rest,data.energy_usage)

weather = weather_data2.loc[df_forecast.index <= pd.Timestamp("2023-09-21 13:00:00"), :].copy()
weather = weather[weather.index.isin(df_compare.index)]


weather['humidity_sqrt']= weather.humidity**0.5
weather['wnd_sqrt']= weather.wind_speed**0.5
weather['temp_sqd'] = weather.temperature **2

df_compare['reg_forecast'] = reg.predict(weather.loc[:,rest.columns])
rest['reg_forecast'] = reg.predict(rest)

df_compare.loc[df_compare.index.day.isin([5]), 'reg_forecast'] = df_stats.loc[5,'energy_usage'].item()
df_compare.loc[df_compare.index.day.isin([6]), 'reg_forecast'] = df_stats.loc[6,'energy_usage'].item()

rest.loc[rest.index.day.isin([5]), 'reg_forecast'] = df_stats.loc[5,'energy_usage'].item()
rest.loc[rest.index.day.isin([6]), 'reg_forecast'] = df_stats.loc[6,'energy_usage'].item()


# %%
source = ColumnDataSource(data={
        "series_name": ["Energy usage 2022"] * data.shape[0],
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "energy_usage": data.energy_usage,
    })
source2 = ColumnDataSource(data={
        "series_name": ["Energy usage 2023"]* data2.shape[0],
        "ts_str": data2.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data2.index,
        "energy_usage": data2.energy_usage,
    })
source3 = ColumnDataSource(data={
        "series_name": ["Energy forecast"]* df_compare.shape[0],
        "ts_str": df_compare.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": df_compare.index,
        "energy_usage": df_compare.reg_forecast,
    })

TOOLTIPS = [
    ('Series Name', '@series_name'),
    ('Time stamp','@ts_str'),
    ('Energy usage', '@energy_usage{int}')
    ]

p = figure(title='Energy usage and prediction 2022-2023', x_axis_label='Time Stamp', y_axis_label='Energy usage', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_usage',source=source,color='gold',legend_label='Energy usage 2022')
p.line('ts','energy_usage',source=source2, legend_label='Energy usage 2023')
p.line('ts','energy_usage',source=source3,color='grey',legend_label = 'Energy forecast')
p.legend.location='top_left'
p.legend.click_policy="hide"

show(p)

# %%
MSE = np.square(np.subtract(df_compare.energy_usage,df_compare.reg_forecast)).mean()
RMSE = math.sqrt(MSE)

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

MAE = mae(df_compare.energy_usage, df_compare.reg_forecast)

MBE = np.mean(df_compare.reg_forecast - df_compare.energy_usage)

CVRMSE=(RMSE/np.mean(df_compare.energy_usage))*100
CVMAE=(MAE/np.mean(df_compare.energy_usage))*100
CVMBE=(MBE/np.mean(df_compare.energy_usage))*100
print(CVRMSE)
print(CVMAE)
print(CVMBE)

# %%
MSE = np.square(np.subtract(data.energy_usage,rest.reg_forecast)).mean()
RMSE = math.sqrt(MSE)

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

MAE = mae(data.energy_usage, rest.reg_forecast)

MBE = np.mean(rest.reg_forecast - data.energy_usage)

CVRMSE=(RMSE/np.mean(data.energy_usage))*100
CVMAE=(MAE/np.mean(data.energy_usage))*100
CVMBE=(MBE/np.mean(data.energy_usage))*100
print(CVRMSE)
print(CVMAE)
print(CVMBE)
source = ColumnDataSource(data={
        "series_name": ["Energy usage 2022"] * data.shape[0],
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "energy_usage": data.energy_usage,
    })
source2 = ColumnDataSource(data={
        "series_name": ["Reg prediction"]* rest.shape[0],
        "ts_str": rest.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": rest.index,
        "energy_usage": rest.reg_forecast,
    })

TOOLTIPS = [
    ('Series Name', '@series_name'),
    ('Time stamp','@ts_str'),
    ('Reg prediction', '@energy_usage{int}')
    ]

p = figure(title='Energy usage and prediction 2022-2023', x_axis_label='Time Stamp', y_axis_label='Energy usage', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_usage',source=source,color='gold',legend_label='Energy usage 2022')
p.line('ts','energy_usage',source=source2, legend_label='Reg prediction')
p.legend.location='top_left'
p.legend.click_policy="hide"

show(p)

# %%

source = ColumnDataSource(data={
        "series_name": ["Energy usage 2022"] * data.shape[0],
        "ts_str": data.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data.index,
        "energy_usage": data.energy_usage,
    })
source2 = ColumnDataSource(data={
        "series_name": ["Energy usage 2023"]* data2.shape[0],
        "ts_str": data2.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": data2.index,
        "energy_usage": data2.energy_usage,
    })
source3 = ColumnDataSource(data={
        "series_name": ["Energy forecast"]* df_compare.shape[0],
        "ts_str": df_compare.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": df_compare.index,
        "energy_usage": df_compare.reg_forecast,
    })
source4 = ColumnDataSource(data={
        "series_name": ["Reg prediction"]* rest.shape[0],
        "ts_str": rest.index.strftime('%Y-%m-%d %H:%M:%S'),
        "ts": rest.index,
        "energy_usage": rest.reg_forecast,
    })



TOOLTIPS = [
    ('Series Name', '@series_name'),
    ('Time stamp','@ts_str'),
    ('Energy usage', '@energy_usage{int}')
    ]

p = figure(title='Energy usage and prediction 2022-2023', x_axis_label='Time Stamp', y_axis_label='Energy usage', x_axis_type='datetime' ,width=1800,height=900,tooltips=TOOLTIPS)
p.line('ts','energy_usage',source=source,color='gold',legend_label='Energy usage 2022')
p.line('ts','energy_usage',source=source2, legend_label='Energy usage 2023')
p.line('ts','energy_usage',source=source3,color='grey',legend_label = 'Energy forecast')
p.line('ts','energy_usage',source=source4,color='grey',legend_label = 'Energy forecast')
p.legend.location='top_left'
p.legend.click_policy="hide"

show(p)

