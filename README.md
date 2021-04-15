**_Concretely, this repo and [the associated bot](https://twitter.com/bot_2024) provide a host of possible 2024 maps based on demographics and recent political history that account for trends and their possible continuations or reversals, slight shifts in political coalitions, and differing national environments._**

Perhaps no election in recent memory was as strongly explained by demographics as the [2020 election was](https://centerforpolitics.org/crystalball/articles/demographics-and-expectations-analyzing-biden-and-trumps-performances/). It is this predictive power that makes county margins extremely correlated across the nation, and with this, we can better analyze the nature of political coalitions to understand what different electoral scenarios and victory maps may look like for each political party.

This model takes in county-level demographic data, along with 2012 and 2016 partisanship, and extracts important features (via Principal Component Analysis) that would help predict the 2020 margins. Once it is trained to predict the 2020 partisanship of a county, it is fed 2024 demographic projections (extrapolated from census estimates over the past decade) and outputs a plausible map for 2024 based on the projected demographics, prior electoral history, and probabilistic sampling.

For the more technically inclined readers, the way that the probabilistic sampling is done is via the injection of gaussian noise into the predictor's feature weights (`w_i' = w_i * n, n ~ N(1, 0.1)`); this accounts for the possibility of 2024 coalitions differing from 2020 coalitions in some ways (e.g. Hispanic voters move more to the right, college whites move more to the left) while preserving state and county-level correlations.

Because the bot is essentially fitting 2024 (projected) demographics to 2020 results and then probabilistically sampling, it will provide some Republican-leaning maps and some Democratic-leaning ones in near-identical proportions (54% Democratic, 46% Republican, and the slight Democratic lean is entirely due to the results of the 2020 election). **This is not due to our belief of what is likely to happen, but is rather based on our desire to present a range of possible and plausible electoral scenarios across the board. It is not a predictive model, and it is not meant to be a predictive model for what will happen in 2024. It is meant to be a showcaser of possible coalitions that may emerge across a host of national environments.**

You can see state level margins for each scenario at [this spreadsheet](https://docs.google.com/spreadsheets/d/1GD9GahVdiuYDR82Ne7oV_qW2hKwbL7cuyhgD1AhbZ48/edit#gid=0).


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
