**_Concretely, this repo and [the associated bot](https://twitter.com/bot_2024) provide a host of possible 2024 maps based on demographics and recent political history that account for trends and their possible continuations or reversals, slight shifts in political coalitions, and differing national environments._**

Perhaps no election in recent memory was as strongly explained by demographics as the [2020 election was](https://centerforpolitics.org/crystalball/articles/demographics-and-expectations-analyzing-biden-and-trumps-performances/). It is this predictive power that makes county margins extremely correlated across the nation, and with this, we can better analyze the nature of political coalitions to understand what different electoral scenarios and victory maps may look like for each political party.

This model projects 2024 demographics from the available 2012, 2016, and 2020 demographic data. In terms of predictions and setup, the model takes in county-level demographic data for 2012, 2016, and 2020 and uses it to construct underlying demographic features for the electorate. It then uses a (simple!) neural network to carve the electorate up into a latent "demographic space" on a per-county basis before predicting the turnout and partisanship per demographic and use that to compute county partisanship. The model is trained on the swings in each county from 2008->2012, 2012->2016, and 2016->2020. Once we learn the swing model, we add in the 2016 partisanship of the county to get the overall 2024 margin per county. We perturb the model's predictions to get different maps for 2024, showcasing a variety of unique and plausible coalitions. It is important to note that in predicting 2024, we center the model around the partisanship per "hidden" demographic, but we use a weighted average of the 2012/2016/2020 turnouts, because we do not know what turnout will look like and have no specific reason to prefer one scenario over another among the three. 

For the more technically inclined readers, the way that the probabilistic sampling for the map generation is done is via the injection of gaussian noise into the predictor's feature weights (`w_i' = w_i * n, n ~ N(1, 0.1)`); this accounts for the possibility of 2024 coalitions differing from 2020 coalitions in some ways (e.g. Hispanic voters move more to the right, college whites move more to the left) while preserving state and county-level correlations. There is a significant amount of variance in the model's predictions (90% of the model's maps fall within a 16 point interval), and this is done primarily to showcase the variance inherent in such an uncertain scenario -- at the time of creation, no polls of predictive quality even exist for 2024, nor has the race even materialized, but by examining the ongoing realignment in the context of demographics, we may understand how the electorate will look in the context of the predicted 2024 demographics and the election. 

It should be noted that the bot is not predictive and is not meant to be an oracle on what will happen. It is only meant to be an interesting exercise in showcasing different plausible coalitions based on recent trends in both partisanship and turnout. It will tweet an equal rate of Republican and Democratic maps, regardless of the internal simulation results -- on odd days, we sample Republican maps, and on even days we sample Democratic ones. **This is not due to our belief of what is likely to happen, but is rather based on our desire to present a range of possible and plausible electoral scenarios across the board. It is not a predictive model, and it is not meant to be a predictive model for what will happen in 2024. It is meant to be a showcaser of possible coalitions that may emerge across a host of national environments.**

You can see state level margins for each scenario at [this spreadsheet](https://docs.google.com/spreadsheets/d/1GD9GahVdiuYDR82Ne7oV_qW2hKwbL7cuyhgD1AhbZ48/edit#gid=0).

*(2016 because we essentially predict the 2020 swings but with the 2024 electorate. That's a simple way of saying "If the 2020 election saw the 2016->2020 swings that it did among the electorate on a demographic level, but the electorate was the 2024 electorate, what would the election map resemble?" This is the scenario that our model is centered on.)
## Running the model

First, install orca,

```
conda install -c plotly plotly-orca
```

if the above does not work, try following the instructions [here](https://plotly.com/python/orca-management/).
Then, create a virtualenv and install the requirements

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```
