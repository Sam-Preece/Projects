# The task

I was initially given two spreadsheets, one with weather data and one with energy usage - both for 2022,
and I was tasked with formatting, modelling and predicting future data based off it.

There were some issues.

## Formatting

The first problem was that the weather data was not given in a nice format - It had different columns for year, month and day.
This made it very tricky to plot any graphs or look into patterns as it would show multiple events for any one of these columns.
So I had to collapse each of these into a timestamp
- This took me a little while but was eventually sorted out.

Luckily the energy data was given with a timestamp so all was well. Time to merge these two dataframes into something I can work with.
Using the merge was simple enough once some SQL logic was learnt for the 'how' parameter and now the initial formattin been completed.

Time to have some fun with the dataframe.

To start I graphed each of the columns against time - here's just the energy usage:

[]!(https://github.com/Sam-Preece/Projects/blob/master/img/Energy_usage_2022.png)

Then I created a correlation matrix to see if I could find any patterens that way.
- As the output was bunch of numbers I thought a more intuitive way of presenting the data would be as a heatmap.
However using bokeh to plot these graphs (It's nice and ineractive - bit a bit more finicky to create with) created some problems.
Nothing a little google and asking for help from people way more experienced than you can't solve:

[]!(https://github.com/Sam-Preece/Projects/blob/master/img/Correlation_heatmap.png)

Seeing this I graphed the energy usage against temperature and found some interesting patterns:

[]!(https://github.com/Sam-Preece/Projects/blob/master/img/Energy_usage_against_temp.png)

As you can see it appears to have three main patterns.
- looking closer I noticed the bottom line only happend on the 6th of every month and the one above on the 5th.
If you look back and the enery usage against time you can see these patterns quite clearly.
(As the energy usage data was made up for this task its obvious they were thrown in to catch me out,
but a real world explanation could be that monthly maintanence or other monthly events cause this - so its not unheard of)

Now that we have some ideas as to the patterns we can get to some modelling!

## Modelling

The first model I created was by far the simplest one I could think of other than just repeating the data from 2022.
Taking an average for and day of the week in a given month other than the 5th and the 6th (- they get their own averages).
Thats it and it looks like ths:

[]!(https://github.com/Sam-Preece/Projects/blob/master/img/First_forecast.png)

As you can see its quite repetative as each day of the week for a month is the exact same.
Calculating the error from the 2022 gives: The CV RMSE: 21.60 %, The CV MAE: 16.43 % and the CV MBE: 0%
Pretty good bt we can do better - now its time for something more intellegent

### Linear regression

Using linear regression for this data is probably the best model I could feasibly create (not just because thats how the data was made)
and implementing it wasn't too challenging either, given the sklearn module has a linear regression function.
Using trial and error I was able to bring the model closer to the actual data by creating merged columns of:
Square root of humidity, Square root of wind speed and the square of the temperature.
And when show with the training data - its quite clear how it's pretty much spot on:

[]!(https://github.com/Sam-Preece/Projects/blob/master/img/Training_prediction.png)

The error values are: The CV RMSE: 13.18 % %, The CV MAE: 8.66 % and the CV MBE:-4.59 %
These are amazing - well within any model quality checks - however these are a lower bound - We can predict any better than this
- and we can get a lot further off.

## Prediction

Time to predict some future data using the two models i've created - Now at this point I was given some weather data and energy usae for 2023.
I formatted it in the exact same way as the last time giving this energy usage graph:

[]!(https://github.com/Sam-Preece/Projects/blob/master/img/Energy_usage_2023.png)

As you can see, it gets pretty weird - This was to show that not everything in data science will be nice although for a while i thought i had done something wrong and tried to fix it.
However if we only evaluate our models up to the point before it gets weird it shouldn't be that far off, Heres the two combined energy usage graphs:

["First model"]!(https://github.com/Sam-Preece/Projects/blob/master/img/First_combined_energy_usage.png)

["Linear regression"]!(https://github.com/Sam-Preece/Projects/blob/master/img/Second_combined_energy_usage.png)

And given these I calculated the error for the linear regression model: The CV RMSE: 28.72 %, The CV MAE: 22.86 % and the CV MBE:14.61 %

All in all I'm quite happy with how I did - not too far off even with the curveballs thrown in. The predictions are obviously not perfect
but I think for only having learnt and created this in three days it's not half bad.



