## Introduction:
This project is aimed at analysing the impact of the changes made to improve the user experience of the website.

## Objectives:
1. To measure the daily overall clickthrough rate
2. To examine which results people tend to try first and how it changes day-to-day.
3. To measure the daily overall zero results rate and how it varies between the different groups.
4. To examine the correlation between number of results and session length.

## Installations:
#### Install relevant libraries for transformation, analysis and visualization
1. !pip install pandas
2. !pip install numpy
3. !pip install seabon
4. !pip install matplotlib
5. !pip install scipy

## Usage:

##### Loading Important Libraries

```python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


##### Data Exploration

data = pd.read_csv("events_log.csv")
data.head()

data.info()

data.shape

data.isnull().sum() # checking for number of missing values

data.select_dtypes(include=['object']).nunique() # checking for distinct values in each column

cnt_action = data['action'].value_counts()
cnt_action


## Data Transformation


data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d%H%M%S')  # Converting Date to Timestamp



## Analysis
# Question 1:What is the daily overall clickthrough rate? How does it vary between the groups?

# Steps:
# - Calculate the clickthrough rate (CTR) based on the input data

def overall_clickthrough_rate(data):
    search_events = data[data['action'] == 'searchResultPage'].groupby(data['timestamp'].dt.date).size()
    click_events = data[data['action'] == 'visitPage'].groupby(data['timestamp'].dt.date).size()
    overall_ctr = click_events / search_events
    return overall_ctr

daily_overall_ctr = overall_clickthrough_rate(data)
#print("Daily overall clickthrough rate:")
print(daily_overall_ctr)

# Visualise your result

plt.figure(figsize=(10, 6))
plt.plot(daily_overall_ctr.index, daily_overall_ctr.values, marker='o',linestyle='-')
plt.title('Daily Overall Clickthrough Rate')
plt.xlabel('Date')
plt.ylabel('Overall Clickthrough Rate')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Question 1b: How the CTR varies between the different Groups

def clickthrough_rate(data):

    search_events = data[data['action'] == 'searchResultPage'].groupby([data['timestamp'].dt.date, 'group']).size()
    click_events = data[data['action'] == 'visitPage'].groupby([data['timestamp'].dt.date, 'group']).size()
    ctr = click_events / search_events
    return ctr
daily_ctr_grp = clickthrough_rate(data)
print("Daily Clickthrough rate by Group:")
print(daily_ctr_grp)

# Variation between groups
ctr_grpA = daily_ctr_grp[:, 'a'].values
ctr_grpB = daily_ctr_grp[:, 'b'].values

ctr_grpA_mean = np.mean(ctr_grpA)
ctr_grpB_mean = np.mean(ctr_grpB)
print("Mean ctr_grpA:", ctr_grpA_mean)
print("Mean ctr_grpB:", ctr_grpB_mean)


if ctr_grpA_mean > ctr_grpB_mean:
    print("Answer: Group A has a higher clickthrough rate.")
elif ctr_grpB_mean > ctr_grpA_mean:
    print("Answer: Group B has a higher clickthrough rate.")
else:
    print("Answer: Both groups have the same clickthrough rate.")


# Visualise the CTR for each group per day

plt.figure(figsize=(10, 6))
plt.plot(daily_ctr_grp.index.levels[0], ctr_grpA, label='Group A')
plt.plot(daily_ctr_grp.index.levels[0], ctr_grpB, label='Group B')
plt.title('Clickthrough Rate Variance between Groups A and B')
plt.xlabel('Date')
plt.ylabel('Clickthrough Rate')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
#ax.set_facecolor('darkgrey') 
plt.show()


# Question 2: Which results do people tend to try first?

# Steps:
# - Calculate the count of first clicks on each result position from the input data

def first_click_results(data):
    first_clicks = data[data['action'] == 'visitPage']['result_position'].value_counts().sort_index()
    return first_clicks.head()
first_clicks_distribution = first_click_results(data)
print("Results people try first:")
print(first_clicks_distribution)


# - Visualise the distribution of first-click positions

plt.figure(figsize=(10, 6))
first_clicks_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of First Click Positions')
plt.xlabel('Result Position')
plt.ylabel('Number of First Clicks')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Question 2b: How does it change day-to-day? 

# Steps:
# - Calculate the count of first clicks for the top 3 result positions per day from the input data

def Top_3_click_results_per_day(data):
    Top_3_clicks_per_day = data[data['action'] == 'visitPage'].groupby([data['timestamp'].dt.date, 'result_position']).result_position.value_counts().unstack(fill_value=0)
    return Top_3_clicks_per_day.iloc[:, :3]
Top_3_clicks_distribution_per_day = Top_3_click_results_per_day(data)
print("Top 3 Results people try daily):")
print(Top_3_clicks_distribution_per_day)

# - Visualise the change in the Top 3 Click result positions by day

plt.figure(figsize=(10, 6))
Top_3_clicks_distribution_per_day.plot(marker='o', linestyle='-')
plt.title('Trend of Top 3 Click Result Positions Daily')
plt.xlabel('Date')
plt.ylabel('No of Top 3 Clicks')
plt.legend(title='Result Position')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Question 3: What is the daily overall zero results rate?

# Steps:
# - Calculate the ZR events per day from the input data

def overall_ZRR(data):
    search_events = data[data['action'] == 'searchResultPage'].groupby(data['timestamp'].dt.date).size()
    zero_results_events = data[data['n_results'] == 0].groupby(data['timestamp'].dt.date).size()
    overall_ZRR = zero_results_events / search_events
    return overall_ZRR

daily_overall_ZRR = overall_ZRR(data)
print("Daily overall ZRR:")
print(daily_overall_ZRR)

# - Visualise the Overall ZRR by day

plt.figure(figsize=(10, 6))
plt.plot(daily_overall_ZRR.index, daily_overall_ZRR.values, marker='o',linestyle='-')
plt.title('Daily Overall ZRR')
plt.xlabel('Date')
plt.ylabel('Overall ZRR')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Question 3b: How does it vary between the groups?

# Steps:
# - Calculate the ZR in the search events from the input data grouping partitioning by date and group 

def ZR_Rate(data):

    search_events = data[data['action'] == 'searchResultPage'].groupby([data['timestamp'].dt.date, 'group']).size()
    zero_results_events = data[data['n_results'] == 0].groupby([data['timestamp'].dt.date,'group']).size()
    ctr = zero_results_events / search_events
    return ctr
daily_zr_grp = ZR_Rate(data)
print("Daily ZR rate by Group:")
print(daily_zr_grp)


# - Create a mean ZRR distribution for each group to dtermine the variation
zrr_grpA = daily_zr_grp[:, 'a'].values
zrr_grpB = daily_zr_grp[:, 'b'].values

zrr_grpA_mean = np.mean(zrr_grpA)
zrr_grpB_mean = np.mean(zrr_grpB)
print("Mean ctr_grpA:", zrr_grpA_mean)
print("Mean ctr_grpB:", zrr_grpB_mean)


# - Visualise the ZRR between the 2 groups
plt.figure(figsize=(10, 6))
plt.plot(daily_zr_grp.index.levels[0], zrr_grpA, label='Group A')
plt.plot(daily_zr_grp.index.levels[0], zrr_grpB, label='Group B')
plt.title('ZRR Variance between Groups A and B')
plt.xlabel('Date')
plt.ylabel('ZRR')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
#ax.set_facecolor('darkgrey') 
plt.show()

# Question  3c: Is there a statistically significant difference?

# Steps:
# - Create independent two-sample t-tests and a p-value to determine statistical significant difference

zrr_grpA = daily_zr_grp[:, 'a'].values
zrr_grpB = daily_zr_grp[:, 'b'].values

t_stat, p_value = ttest_ind(zrr_grpA,zrr_grpB)
print("Statistical significance (p-value) between groups:")
print("p-value =", round(p_value,2))

if p_value < 0.05:
    print("There is a statistically significant difference between the ZRR of groups 'a' and 'b'.")
else:
    print("There is no statistically significant difference between the ZRR of groups 'a' and 'b'.")

# Question 4: Let session length be approximately the time between the first event and the last event in a session. Choose a variable from the dataset and describe its relationshipto session length by group. Can you identify any strong correlations?


# Steps:
# Calculate the relationship between the chekin variable and the session length variable

session_length = data.groupby(['session_id', 'group'])['timestamp'].agg(lambda x: (x.max() - x.min()).seconds)
variable_session_corr = data.groupby(['session_id', 'group'])['checkin'].mean().corr(session_length)
print("\nCorrelation between 'checkin' and session length by group:",round(variable_session_corr,2))

# - Create a DataFrame to store session length and checkin mean

session_length = data.groupby(['session_id', 'group'])['timestamp'].agg(lambda x: (x.max() - x.min()).seconds)
checkin_mean = data.groupby(['session_id', 'group'])['checkin'].mean()

session_checkin_df = pd.DataFrame({'session_length': session_length,'checkin_mean': checkin_mean}) 

# - Visualize the relationship between both variables

plt.figure(figsize=(10, 6))
sns.scatterplot(x='checkin_mean', y='session_length', hue='group',data=session_checkin_df)
plt.title('Relationship between Checkin Time and Session Length by Group')
plt.xlabel('Checkin Mean (seconds)')
plt.ylabel('Session Length (seconds)')
plt.legend(title='Group')
plt.grid(True)
plt.show()

