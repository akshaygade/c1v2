#!/usr/bin/env python
# coding: utf-8

# # **Capital One Data Challenge - Venkata Sai Akshay Gade**

# # **Problem Statement:**
# An Airline company is looking to enter the US domestic market and needs help to:
# 1. Identify the 10 busiest round trip routes in terms of number of round trip flights in the quarter.
# 
# 2. Identify the 10 most profitable round trip routes in the quarter.
# 
# 3. Suggest the 5 round trip routes that I recommend to invest in based on any factors that I choose.
# 
# 4. The number of round trip flights needed to breakeven on the upfront airplane cost for those 5 round trip routes.
# 
# 5. KPI’s that you recommend tracking in the future to measure the success of the round trip routes that you recommend.

# # 2.  Import Libraries & Load Datasets

# In[21]:


import pandas as pd
import os
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default = "notebook_connected"
import numpy as np
# Defining assumptions
max_seats = 200
baggage_fee = 70 # Per round trip
baggage_rate = 0.5

# Operational costs
cost_per_mile = 8
fixed_cost_per_mile = 1.18
total_cost_per_mile = 9.18
delay_cost_per_min = 75
med_airport_fee = 5000
lar_airport_fee = 10000


# In[22]:


# Get current working directory
cwd = os.getcwd()
print("1. Current working directory: \n"+cwd)

# Set directory path to access data files
data_directory=os.path.join(cwd, "data")
print("\n2. Data files directory: \n"+data_directory)

# Walk through the folder to print the files
for directory, folders, files in os.walk(data_directory):
    print("\n3. Data files:")
    for file in files:
        print(" -> "+file)


# In[23]:


# Read raw files into dataframes
tickets_raw = pd.read_csv(data_directory + "/Tickets.csv")
airport_codes_raw = pd.read_csv(data_directory + "/Airport_Codes.csv")
flights_raw = pd.read_csv(data_directory + "/Flights.csv", low_memory=False) # Set low_memory to false as the dataset's size is large
airline_metadata_raw = pd.read_excel(data_directory + "/Airline_Challenge_Metadata.xlsx")


# In[24]:


# Set copies to avoid overwriting the original dataframe
tickets_df = tickets_raw.copy()
airports_df = airport_codes_raw.copy()
flights_df = flights_raw.copy()
airline_metadata_df = airline_metadata_raw.copy()


# ### Helper Functions

# In[25]:


def plot_bar_chart(df, x_axis, y_axis, text,
    x_axis_title, y_axis_title, title,
    color="#636EFA", xaxis_tickangle=0):
    """
    This function creates an interactive bar chart using plotly express.
    Parameters:
        df (pandas dataframe): The dataframe to plot.
        x_axis (str): Column name for x-axis.
        y_axis (str): Column name for y-axis.
        text (str) : Values to annotate on the bars.
        x_axis_title (str): Label for the x-axis.
        y_axis_title (str): Label for the y-axis.
        title (str): Title of the chart.
        color (str): Color of the chart.
        xaxis_tickangle (int): The x-axis tick angle of the chart.
    Returns:
         An interactive bar chart.
    """
    figure = px.bar(
        df,
        x=x_axis,
        y=y_axis,
        text=text,
        title=title,
        labels={x_axis: x_axis_title, y_axis: y_axis_title},
        color_discrete_sequence=[color]
    )

    figure.update_traces(
        textposition='outside',
        textfont_color='black',
        cliponaxis=False
    )

    figure.update_layout(
        width=900,
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font=dict(size=18, color='black'),
        font=dict(color='black'),
        yaxis=dict(gridcolor='lightgrey'),
        margin=dict(t=80),
        xaxis_tickangle=xaxis_tickangle
    )

    return figure

def plot_interactive_pie(df, label, value, title, color=None):
    """
    Create an interactive pie chart using plotly express.
    Parameters:
        df (pandas DataFrame): DataFrame containing the data.
        label (str): Column name for pie slice labels.
        value (str): Column name for pie slice values.
        title (str): Title of the chart.
    Returns:
        An interactive pie chart
    """
    figure = px.pie(df, names=label, values=value, title=title, hole=0.2,color=color,color_discrete_sequence=px.colors.qualitative.Pastel)
    figure.update_traces(textposition='inside', textinfo='percent+label')
    figure.update_layout(width=900, height=500, plot_bgcolor='white', paper_bgcolor='white',
                      title_font=dict(size=18, color='black'), font=dict(color='black'))
    return figure

def plot_table(df,index=False):
    """Render a DataFrame as a light‐themed plotly table.
    Parameters:
         pandas.DataFrame
    Returns:
        plotly table
    """
    figure = ff.create_table(df,
                             index=index
                             )

    figure.update_layout(template="plotly_white")

    return figure
# Function to remove outliers using the IQR method
def remove_outliers(df, column):
    """
    Removes outliers from a DataFrame column using the IQR method.

    Parameters:
        df (pandas DataFrame): The input DataFrame
        column (str): The name of the numeric column to check for outliers

    Returns:
        pandas dataFrame: A filtered DataFrame with outliers removed based on the IQR rule.
    """
    # Make sure the column is not a str type
    df[column] = pd.to_numeric(df[column], errors='coerce')

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]

def airport_fee(airport_type):
    """
    Return the fixed airport usage fee based on airport size.

    Parameters:
        airport_type (str): The airport category, expected to be 'medium_airport' or 'large_airport'.
    Returns:
        5000 for medium airports, 10000 for large airports, otherwise 0.
    """
    if airport_type == 'medium_airport':
        return 5000
    elif airport_type == 'large_airport':
        return 10000
    else:
        return 0

def delay_cost_calc(delay):
    """
    Compute the additional cost incurred by flight delay beyond a free 15-minute window.

    Parameters:
        delay (int) : The delay duration in minutes.

    Returns:
        The delay penalty in dollars (float).
    """
    if delay > 15:
        return (delay - 15) * 75
    else:
        return 0


# # 3. Data Quality Checks & Cleaning

# **Objective:**
# Streamline our data to include only the records needed for route analysis, setting the stage for accurate, focused insights.
# 1. Airport Codes:
#     - Retain U.S. airports classified as medium or large.
#     - Drop any entries missing an airport code to preserve data quality.
# 2. Flights:
#     - Remove all canceled flights so we analyze only completed operations.
# 3. Tickets:
#     - Restrict to round-trip itineraries, since our focus is on round-trip route performance.

# ### 3.1.  High Level Summary

# In[26]:


def summarize_dfs(dfs, df_names):
    """
    Summarize a list of dataFrames by computing key metrics for initial analysis.

    Parameters:
        dfs (list of pandas dataFrames): The list of DataFrames to summarize.
        df_names (list of str): Optional names for the datasets.

    Returns:
        pandas dataFrame: Summary statistics table.
    """

    result_df = [] # Empty df to consolidate the results

    for index, df in enumerate(dfs):
        df_name = df_names[index] if df_names else None

        # Total Counts
        num_rows = len(df)
        num_cols = len(df.columns)

        # Missing Values - Rows that contain at least one missing value
        missing_records = df[df.isnull().any(axis=1)].shape[0]
        missing_percentage = (missing_records / num_rows * 100) if num_rows > 0 else 0

        # Duplicate Counts
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / num_rows) * 100 if num_rows > 0 else 0

        # Append each table's result to the empty list
        result_df.append([
            df_name, num_rows,duplicate_count, round(duplicate_percentage,2),  missing_records, round(missing_percentage,2),  num_cols
        ])

    return_df = pd.DataFrame(result_df, columns=[
        "Dataset Name", "# Total Rows", "# Duplicate Rows", "# Duplicate Rows (%)",  "# Missing Records", "# Missing Records (%)", "# Total Columns"])

    summary_df = return_df.T # Transpose the consolidated dataframe for conciseness
    summary_df.columns=summary_df.iloc[0] # Setting the first row as the dataframe header
    summary_df = summary_df[1:] # Remove the redundant first row
    return summary_df


# In[27]:


summary_table=summarize_dfs(dfs=[flights_df, tickets_df, airports_df], df_names=["Flights", "Tickets", "Airport Codes"])
plot_table(summary_table, index=True)


# **Observations:**
# From our initial exploration of the datasets. Here are the following insights:
# 1. Flights: 1.9 million total records - 4.5k (0.24%) records are duplicates,  1.75M (0.5%) values are missing.
# 
# 2. Tickets: 1.1 million records -  71k (6.16%) records are duplicates,  2.9k (0.02%) values are missing.
# 
# 3. Airport codes: 55k records -  101 (0.18%) records are duplicates,  86k (19.64%) values are missing.
# 

# ### 3.2 Flights Data

# In[28]:


flights_df= flights_raw.copy()


# In[29]:


flights_df.head()


# In[30]:


flights_df.info()


# #### Filter for relevant columns

# In[31]:


# removing unnecessary columns
flights_df= flights_df.drop(['TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID', 'ORIGIN_CITY_NAME','DEST_AIRPORT_ID','DEST_CITY_NAME'], axis=1)


# #### Remove Cancelled tickets from our analysis

# In[32]:


flights_df = flights_df[flights_df['CANCELLED'] == 0]


# #### Standardize the date format

# In[33]:


# FL_DATE contains mixed formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY') and standardization is necessary
flights_df['FL_DATE'] = pd.to_datetime(flights_df['FL_DATE'],format='mixed',errors='coerce').dt.strftime('%Y-%m-%d')
print("Standardized date sample values: ")
print(flights_df['FL_DATE'].astype(str).sample(5).tolist())


# #### Clean Distance Column
# 

# In[34]:


flights_df['DISTANCE_CLEAN'] = pd.to_numeric(flights_df['DISTANCE'], errors='coerce')
invalid_distance_values = flights_df[flights_df['DISTANCE_CLEAN'].isna()]['DISTANCE']
print("Invalid distance values:", invalid_distance_values.unique())


# In[35]:


# Drop rows with invalid distance
flights_df = flights_df[flights_df['DISTANCE_CLEAN'].notna()].copy()
flights_df['DISTANCE'] = flights_df['DISTANCE_CLEAN']
flights_df.drop(columns='DISTANCE_CLEAN', inplace=True)


# #### Remove and impute outliers

# In[36]:


# Assuming that there is no delay if the departure and arrival delays are null
flights_df.loc[flights_df['DEP_DELAY'].isna(), 'DEP_DELAY'] = 0
flights_df.loc[flights_df['ARR_DELAY'].isna(), 'ARR_DELAY'] = 0


# In[37]:


# Remove and impute outliers for these four columns
for col in ['DEP_DELAY', 'ARR_DELAY', 'AIR_TIME', 'DISTANCE']:
    flights_df = remove_outliers(flights_df, col)


# In[38]:


# Drop negative records
flights_df = flights_df[flights_df['DISTANCE'] >= 0]
flights_df = flights_df[flights_df['AIR_TIME'] >= 0]
flights_df = flights_df[flights_df['OCCUPANCY_RATE'].between(0,1)]


# #### Drop Duplicate Records

# In[39]:


# Since we know that 4.5k records are duplicates,let's drop them
flights_df = flights_df.drop_duplicates()


# In[40]:


top_10_busiest_airports = pd.DataFrame(flights_df[['ORIGIN']].value_counts()[:10].reset_index().sort_values(by='count', ascending=False))
colors=px.colors.sequential.Cividis
fig = plot_bar_chart(df=top_10_busiest_airports, x_axis='ORIGIN', y_axis='count',text='count',x_axis_title='Airport',
               y_axis_title='Number of Flights Taken',title='Top 10 Busiest Airports ')
fig.update_traces(marker_color=colors[:len(top_10_busiest_airports)])
fig.update_layout(title={'text': 'Top 10 Busiest Airports<br><sub>By number of departures in Q1 2019</sub>','x': 0.5,'xanchor': 'center'})
fig.show()


# ### 3.3 Tickets Data

# In[41]:


tickets_df=tickets_raw.copy()


# In[42]:


tickets_df.head()


# In[43]:


tickets_df.info()


# #### Filter for relevant columns

# In[44]:


tickets_df= tickets_df.drop(['ORIGIN_COUNTRY','YEAR','QUARTER'], axis=1)


# #### Excluding One Way Trips

# In[45]:


# Removing one way trips as our objective is to analyze only round trips
tickets_df = tickets_df[tickets_df.ROUNDTRIP == 1]


# #### Clean up Invalid Ticket Prices

# In[46]:


tickets_df['ITIN_FARE_CLEAN'] = pd.to_numeric(tickets_df['ITIN_FARE'], errors='coerce')
invalid_fare_values = tickets_df[tickets_df['ITIN_FARE_CLEAN'].isna()]['ITIN_FARE']
print("Invalid ticket fare values:", invalid_fare_values.unique())
print(f"Invalid ticket fare records: {len(invalid_fare_values)}")

# Clean up
tickets_df['ITIN_FARE'] = tickets_df['ITIN_FARE'].str.replace('$', '')
tickets_df['ITIN_FARE'] = tickets_df['ITIN_FARE'].astype(float)
tickets_df.drop(columns='ITIN_FARE_CLEAN', inplace=True)


# #### Remove outliers

# In[47]:


# checking for outliers in the ticket fare column
unique_prices = tickets_df[['ITIN_FARE']].drop_duplicates()
top_10_fares = unique_prices[['ITIN_FARE']].sort_values(by='ITIN_FARE', ascending=False).head(10).reset_index(drop=True)
top_10_fares['INDEX'] = top_10_fares.index + 1  # For labeling purposes

fig=plot_bar_chart(df=top_10_fares, x_axis='INDEX', y_axis='ITIN_FARE',text='ITIN_FARE',x_axis_title='Fares',
               y_axis_title='Ticket Price (USD)',title='Top 10 Flight Ticket Prices ',color='#EF553B')
fig.update_layout(title={'text': 'Top 10 Flight Ticket Prices<br><sub></sub>','x': 0.5,'xanchor': 'center'})


# In[48]:


print("Minimum Ticket Price: "+ str(tickets_df['ITIN_FARE'].min().round()) )
print("Average Ticket Price: "+ str(tickets_df['ITIN_FARE'].mean().round()) )
print("Maximum Ticket Price: "+ str(tickets_df['ITIN_FARE'].max().round()) )


# ##### The ticket price data contains extreme outliers, with values ranging from 0 - 38K despite an average of just 473, indicating outliers that need to be removed.

# In[49]:


# Used IQR method to retain statistically representative fares
q1_fares = tickets_df['ITIN_FARE'].quantile(0.25)
q3_fares = tickets_df['ITIN_FARE'].quantile(0.75)
IQR = q3_fares - q1_fares

stat_lower = q1_fares - 1.5 * IQR
stat_upper = q3_fares + 1.5 * IQR

# Business realistic caps
logical_lower = 50
logical_upper = 2000

# Combine both
lower_bound = max(stat_lower, logical_lower)
upper_bound = min(stat_upper, logical_upper)

# Filtering outliers
tickets_df = tickets_df[(tickets_df['ITIN_FARE'] >= lower_bound) & (tickets_df['ITIN_FARE'] <= upper_bound)]


# ##### Used a hybrid approach combining IQR-based outlier detection with business-informed caps: 50 as a minimum to exclude unrealistic low fares, and 2,000 as a maximum to filter out premium or erroneous high prices. This ensures the data reflects typical commercial airfare patterns without being skewed by anomalies.

# #### Drop duplicate records

# In[50]:


# Ensure fares and passengers are all positive numbers
tickets_df = tickets_df.loc[((tickets_df['ITIN_FARE']>0) & (tickets_df['PASSENGERS']>0))]

tickets_df = tickets_df.drop_duplicates()


# ### 3.4 Airports Data

# In[51]:


airports_df=airport_codes_raw.copy()


# In[52]:


airports_df.head()


# #### Filter Airport types

# In[53]:


airport_types_grouped = (airports_df.groupby('TYPE')['NAME'].nunique().
                         reset_index(name='COUNT').sort_values(by='COUNT', ascending=False))

fig=plot_bar_chart(df=airport_types_grouped,x_axis='TYPE',y_axis='COUNT',text='COUNT',x_axis_title='Airport Type',
               y_axis_title='Number of Airports',title='Number of Airports by Type in the U.S.')
fig.update_layout(title={'text': 'Number of Airports<br><sub>By Type</sub>','x': 0.5,'xanchor': 'center'})


# In[54]:


# Filter airport types to medium and large size and country is US
airports_df = airports_df[(airports_df['TYPE'].isin(['medium_airport','large_airport']) )
                          & (airports_df['ISO_COUNTRY']=="US")
                          & (airports_df['IATA_CODE'].notnull())]


# ##### For the purpose of this analysis, only U.S.based airports classified as Medium or Large in size were included.

# #### Drop Irrelevant Columns

# In[55]:


# We can drop the continent column as we are only dealing with US data.
airports_df=airports_df.drop(columns=['ELEVATION_FT', 'CONTINENT', 'COORDINATES'], axis=1)


# #### Drop Null Values

# In[56]:


airports_df.isnull().sum()


# In[57]:


airports_df = airports_df.dropna()


# #### Drop Duplicates

# In[58]:


airports_df = airports_df.drop_duplicates()


# ### Summary
# - 87 rows had non-numeric DISTANCE values like "Hundred" — dropped.
# - 22 rows had negative DISTANCE values (eg: -1947, -198) - dropped.
# - 51K Cancellation Tickets rows - dropped
# - FL_DATE is parsed to datetime and invalid formats are coerced to NaT
# - 75K rows are duplicates - dropped
# - Outliers are imputed using IQR method
# - Only Medium and Large sized airports are included
# - Excluded One way trips from our analysis

# # 4. Data Preparation

# **Objective:** <br>
# Transform and prepare our cleaned tables so every flight record carries the attributes needed for our analysis.
# 
# **Approach:** <br>
#     1. Aggregate Flights: Join by origin–destination to gather origin and destination aiport information associated to each flight route <br>
#     2. Aggregate Tickets: Apply the same grouping to derive consolidated ticket metrics (e.g., total fare, number of passengers) per route.  <br>
#     3. Final Join: Merge these two aggregated tables in a single operation, eliminating redundant records.

# In[59]:


# Create origin and destination airports to join separately
origin_airports = airports_df[['IATA_CODE','TYPE']].drop_duplicates().rename(columns={'IATA_CODE':'ORIGIN_IATA', 'TYPE':'ORIGIN_TYPE'})
destination_airports = airports_df[['IATA_CODE','TYPE']].drop_duplicates().rename(columns={'IATA_CODE':'DESTINATION_IATA', 'TYPE':'DESTINATION_TYPE'})

# Merge using ORIGIN column from flights, ORIGIN_IATA from origin_airports
flights_airports_merged = pd.merge(flights_df,origin_airports,left_on='ORIGIN', right_on='ORIGIN_IATA', how='inner')

# Merge again using DESTINATION column from flights_airport_merged to get destination information
flights_airports_merged = pd.merge(flights_airports_merged,destination_airports, left_on='DESTINATION', right_on='DESTINATION_IATA', how='inner')

# Clean up to remove redundant columns for clarity
flights_airports_merged.drop(columns=['ORIGIN_IATA', 'DESTINATION_IATA','CANCELLED'], inplace=True)

# Create route column to identify normalized route (regardless of direction)
flights_airports_merged['ROUTE'] = flights_airports_merged.apply(lambda x: '-'.join(sorted([x['ORIGIN'], x['DESTINATION']])),axis=1)

# Creating a column for the number of passengers occupied using occupancy rate and capacity of flight
flights_airports_merged['NUM_PASSENGERS'] = flights_airports_merged['OCCUPANCY_RATE'] * max_seats

# Delay Costs
flights_airports_merged['DEP_DELAY_COST'] = flights_airports_merged['DEP_DELAY'].apply(delay_cost_calc)
flights_airports_merged['ARR_DELAY_COST'] = flights_airports_merged['ARR_DELAY'].apply(delay_cost_calc)
flights_airports_merged['TOTAL_DELAY_COST'] = (flights_airports_merged['DEP_DELAY_COST']
                                                + flights_airports_merged['ARR_DELAY_COST'])

# Airport Costs
flights_airports_merged['ORIGIN_AIRPORT_COST'] = flights_airports_merged['ORIGIN_TYPE'].apply(airport_fee)
flights_airports_merged['DESTINATION_AIRPORT_COST'] = flights_airports_merged['DESTINATION_TYPE'].apply(airport_fee)
flights_airports_merged['TOTAL_AIRPORT_COST'] =  (flights_airports_merged['ORIGIN_AIRPORT_COST'] +
                                                   flights_airports_merged['DESTINATION_AIRPORT_COST'])

# Drop duplicates
flights_airports_merged=flights_airports_merged.drop_duplicates()


# In[60]:


# Aggregate flights data to get necessary metrics
flights_agg = flights_airports_merged.groupby(['ROUTE']).agg(
    DISTANCE_TOTAL=('DISTANCE', 'sum'),  # Total distance covered
    TOTAL_DELAY_COST=('TOTAL_DELAY_COST', 'sum'),  # Departure Delay Costs
    NUM_PASSENGERS=('NUM_PASSENGERS', 'sum'),  # Total Number of Passengers
    ARR_DELAY_AVG=('ARR_DELAY', 'mean'),  # Average arrival delay
    DEP_DELAY_AVG=('DEP_DELAY', 'mean'),  # Average departure delay
    ORIGIN=('ORIGIN', 'first'),  # Taking the first origin
    DESTINATION=('DESTINATION', 'first') , # Taking the first destination,
    ORIGIN_TYPE=('ORIGIN_TYPE', 'first'),  # Origin Airport Type
    DESTINATION_TYPE=('DESTINATION_TYPE', 'first'),  # Destination Airport Type
    TOTAL_AIRPORT_COST = ('TOTAL_AIRPORT_COST', 'sum'), # Total Airport Fees Cost
    OCCUPANCY_RATE_AVG = ('OCCUPANCY_RATE', 'mean'), # Average Occupancy Rate for each route
    TOTAL_TRIPS_TAKEN = ('FL_DATE', 'size') # Number of flights taken for each route
).reset_index()

# Aggregate tickets data to get fare information
tickets_df['ROUTE'] = tickets_df.apply(lambda x: '-'.join(sorted([x['ORIGIN'], x['DESTINATION']])), axis=1)
tickets_agg = tickets_df.groupby(['ROUTE']).agg(
    ITIN_FARE=('ITIN_FARE', 'mean'),  # Averaging fare prices
    PASSENGERS=('PASSENGERS', 'sum')  # Total number of  passengers
).reset_index()


# # 5. Data Analysis

# ## Question 1 - The 10 busiest round trip routes in terms of number of round trip flights in the quarter.

# **Approach**
# 1. **Data Preparation:** Filtered out cancelled flights and standardized routes to treat both directions (e.g., LAX-SFO & SFO-LAX) as a single round-trip.
# 
# 2. **Aggregation:** Counted the total number of flights per unique route and divided by two to calculate total round trips.
# 
# 3. **Ranking:** Sorted all routes by round trip volume and selected the top 10 records to identify the busiest routes.

# In[61]:


# Calculate the number of trips per route
routes_grouped = (flights_airports_merged.groupby('ROUTE').size().reset_index(name='TOTAL_FLIGHTS'))

# Convert to round-trip count
routes_grouped['ROUND_TRIP_FLIGHTS'] = (routes_grouped['TOTAL_FLIGHTS'] * 0.5).astype(int)

# Filter for top 10 routes
top_10_busiest_routes = routes_grouped.sort_values(by='ROUND_TRIP_FLIGHTS', ascending=False).head(10)[['ROUTE','ROUND_TRIP_FLIGHTS']]


# In[62]:


plot_table(top_10_busiest_routes)


# In[63]:


# Visual representation
colors = px.colors.qualitative.Pastel
fig = plot_bar_chart(df=top_10_busiest_routes, x_axis='ROUTE', y_axis='ROUND_TRIP_FLIGHTS',text='ROUND_TRIP_FLIGHTS'
                     ,x_axis_title='Airport',y_axis_title='Number of Flights Taken',title='Top 10 Busiest Airports ')
fig.update_traces(marker_color=colors[:len(top_10_busiest_routes)])
fig.update_layout(title={'text': 'Top 10 Busiest Routes<br><sub>By number of round trip flights in Q1 2019</sub>','x': 0.5,'xanchor': 'center'})
fig.show()


# ##### **Observations**
# 1. LAX–SFO stands out as the busiest route in Q1 2019, with over 3,100 round trip flights, significantly ahead of all others.
# 
# 2. Major hubs like LAX, LGA, and ATL appear multiple times, reinforcing their roles as high traffic routes in the domestic network.

# ## Question 2 - The 10 most profitable round trip routes

# **Objective:**
# Identify the top ten round-trip routes by total net profit for Q1-2019, excluding cancelled flights.
# 
# **Approach:**
# 1. **Compute Revenue & Cost Components:** <br>
#     - Use flight-level occupancy and standard fares to calculate ticket and baggage revenue per route.
#     - Derive variable (fuel, maintenance) and fixed (airport fees, delay penalties) cost elements per route.<br>
# 2. **Aggregate to Route Level** <br>
#     - Create a canonical ROUTE key and sum revenues and costs across both directions.
#     - Join aggregated ticket-fare statistics to validate revenue estimates where needed.<br>
# 3. **Rank & Select Top 10 Routes** <br>
#    - Calculate TOTAL_PROFIT (TOTAL_REVENUE – TOTAL_COSTS) for each route.
#    - Sort routes by descending TOTAL_PROFIT and pick the first ten for detailed analysis.

# **Assumptions considered:**
# 1. Fuel, Oil, Maintenance, Crew - 8 dollars per mile total
# 2. Depreciation, Insurance, Other - 1.18 dollars per mile total
# 3. Airport operational costs for the right to use the airports and related services are fixed at 5000 dollars for medium airports and 10,000 dollars for large airports. There is one charge for each airport where a flight lands
# 4. For each individual departure, the first 15 minutes of delays are free, otherwise each minute costs the airline 75 in added operational costs.
# 5. For each individual arrival, the first 15 minutes of delays are free, otherwise each minute costs the airline 75 in added operational costs
# 6. Each plane can accommodate up to 200 passengers and each flight has an associated occupancy rate provided in the Flights data set. Do not use the Tickets data set to determine occupancy.
# 7. Baggage fee is 35 dollars for each checked bag per flight. We expect 50% of passengers to check an average of 1 bag per flight. The fee is charged separately for each leg of a round trip flight, thus 50% of passengers will be charged a total of 70 dollars in baggage fees for a round trip flight.
# 8. Disregard seasonal effects on ticket prices

# In[64]:


# Merge the aggregated dataframes
merged_df = pd.merge(flights_agg, tickets_agg, on='ROUTE')

# Calculate Revenues
merged_df['TICKET_REVENUE'] = merged_df['NUM_PASSENGERS'] * merged_df['ITIN_FARE']
merged_df['BAGGAGE_REVENUE'] = merged_df['NUM_PASSENGERS'] * baggage_rate * baggage_fee

merged_df['TOTAL_REVENUE'] = merged_df['TICKET_REVENUE'] + merged_df['BAGGAGE_REVENUE']

# Calculate total cost and total revenue
merged_df['TOTAL_COST'] = (merged_df['DISTANCE_TOTAL'] * total_cost_per_mile
                          + merged_df['TOTAL_DELAY_COST']
                          + merged_df['TOTAL_AIRPORT_COST'])

merged_df['PROFIT'] = merged_df['TOTAL_REVENUE'] - merged_df['TOTAL_COST']


# In[65]:


top_10_profitable_routes = merged_df[['ROUTE','PROFIT','TOTAL_REVENUE','TOTAL_COST']].sort_values(by='PROFIT', ascending=False).head(10)
# Define columns to format and new names
columns_to_format = {
    'PROFIT': 'PROFITS',
    'TOTAL_COST': 'COSTS',
    'TOTAL_REVENUE': 'REVENUE'
}
# Apply formatting to millions with rounding
for orig_col, new_col in columns_to_format.items():
    top_10_profitable_routes[new_col] = top_10_profitable_routes[orig_col].apply(lambda x: f"${x/1e6:,.0f}M")
    top_10_profitable_routes[orig_col]= (top_10_profitable_routes[orig_col]/1000000).round(2)


# In[66]:


plot_table(top_10_profitable_routes[['ROUTE','REVENUE','COSTS','PROFITS']])


# In[67]:


fig = px.bar(top_10_profitable_routes[['ROUTE','PROFIT','TOTAL_COST','TOTAL_REVENUE']].sort_values('PROFIT', ascending = False), x="ROUTE", y=["TOTAL_REVENUE","TOTAL_COST", "PROFIT"], text_auto = True, title="Route Vs Profit ($Mil)")

fig.update_layout(title={'text': 'Top 10 Routes<br><sub>By Profitability in 2019 Q1</sub>','x': 0.5,'xanchor': 'center'},
                  width=950,height=500,showlegend=True,plot_bgcolor='white',paper_bgcolor='white',
                  title_font=dict(size=18, color='black'),font=dict(color='black'),yaxis=dict(gridcolor='lightgrey') )
fig.show()


# In[68]:


rev_vs_cost_df = top_10_profitable_routes.sort_values("PROFIT", ascending=True)[["ROUTE", "TOTAL_REVENUE", "TOTAL_COST","PROFITS"]].copy()
rev_vs_cost_df["TOTAL_COST"] *= -1
long_df = rev_vs_cost_df.melt(id_vars="ROUTE",value_vars=["TOTAL_REVENUE", "TOTAL_COST"],var_name="Metric",value_name="Amount"         ,
                              ).replace({"TOTAL_REVENUE": "Revenue", "TOTAL_COST": "Cost"})
long_df['label']=long_df.Amount.abs().round()
# Diverging bar chart
fig = (px.bar(long_df, y="ROUTE", x="Amount", color="Metric", orientation="h",color_discrete_map={"Revenue": "green", "Cost": "red"},
             text="label") )
fig.update_layout(title={'text': 'Top 10 Profitable Routes <br><sub>Revenue vs Cost</sub>','x': 0.5,'xanchor': 'center'},
                  barmode="relative", xaxis_title="Amount ($ Mil)", yaxis_title="Route",width=950,height=500,showlegend=True,
                  plot_bgcolor='white',paper_bgcolor='white',title_font=dict(size=18, color='black'),
                  font=dict(color='black'),yaxis=dict(gridcolor='lightgrey'))
fig.show()


# **Observations:**
# 
# - **Profit Leaders:** The top route (e.g., DCA–ORD) generates roughly \$145 M in Q1 net profit, accounting for over 15 % of total profit across all routes.
# - **Margin Strength:** Most of the top 10 routes deliver profit margins above 30 %, indicating that high revenue is paired with cost control.
# - **Cost Drivers:** Routes with longer stage lengths (e.g., LAX–SFO) incur higher per-flight costs yet still rank in the top 10 because of strong revenue.
# - **Timeliness Impact:** While average delays vary from -5 min to over 10 min, their overall effect on profitability remains secondary to revenue scale.

# ## Question 3 -  Top 5 Recommended Investments

# ##### **Objective:**
# 
# Identify five U.S. round-trip routes that maximise net profit and preserve the airline’s “On time, for you” reputation
# 
# ---
# 
# ##### **Approach — Profit-Timeliness-Demand Scorecard**
# 
# 1. <b>Baseline Filter</b> – apply a minimum business standard <i>before</i> scoring. This removes marginally late routes, ensuring management time is spent only on viable candidates.<br>
#     ▪ Average load-factor (occupancy) of at least <b>25 %</b>, ensuring the route is not critically under-filled.<br>
#     ▪ At least <b>100 round-trip rotations</b>, providing a statistically meaningful performance record.<br>
#     ▪ Mean departure-delay ≤ <b>45 minutes</b>, eliminating routes that would undermine an “on-time” brand position.</li>
# 
# 2. <b>Tri-Pillar Score (0 – 100%)</b> – convert three key KPIs into a single, comparable scale:<br>
#     ▪ <b>40% - Profit</b> – routes are benchmarked against the highest aggregate profit in the dataset.<br>
#     ▪ <b>30% - Delay (inverse)</b> – lower average delays earn a higher score, rewarding timeliness.<br>
#     ▪ <b>30% - Occupancy</b> – measures how well seats are filled without letting an occasional sell-out skew the score.
# 
# 3. <b> Ranking</b> – sum the three pillar scores and identify the top-five routes for investment consideration.
# 
# <b>Rationale:</b> <br>
# By filtering first, we eliminate routes that would fail operationally or reputationally.
# The tri-pillar index then fuses profit, punctuality, and demand into a single, transparent score-easy to identify the most viable investments

# In[69]:


"""
Rank routes with a three-pillar Profit–Timeliness–Demand (PTD) index and return the five highest-scoring candidates.
    The function:
    1. Filters routes that fail baseline business criteria
       – average load-factor below min_load (0.25)
       – fewer than min_round_trips (100) flown in the quarter
       – mean of departure + arrival delay above 45 minutes.

    2. Normalize each surviving route on three KPIs:
       Profit , Mean Delay and Occupancy

    3. Combines the normalised scores with user-supplied weights
       weight_profit, weight_delay, weight_occupancy to calculate a single PTD_SCORE.

    4. Returns the top five routes after dropping helper columns.
"""
def rank_routes_ptd(route_df,
                    weight_profit=0.40,
                    weight_delay=0.30,
                    weight_occupancy=0.30,
                    min_load=0.25,
                    min_round_trips=100,
                    max_avg_delay=45):
    """
    Parameters
    ----------
    route_df : pd.DataFrame
        Must contain the following columns
        PROFIT : total profit per route
        ARR_DELAY_AVG : mean arrival delay (min)
        DEP_DELAY_AVG : mean departure delay (min)
        OCCUPANCY_RATE_AVG : mean load-factor (0–1)
        TOTAL_TRIPS_TAKEN : legs or trips flown in the quarter.
    weight_profit, weight_delay, weight_occupancy : optional
        Weights assigned to the three pillars; should sum to 1.0.
    min_load : optional
        Minimum average load-factor required to pass the baseline filter.
    min_round_trips : optional
        Minimum number of trips required to pass the baseline filter.
    max_avg_delay : optional
        Maximum allowed mean delay (minutes) after combining departure and arrival delay.

    Returns
    -------
    pandas DataFrame
        The five highest-ranked routes, with helper columns removed.
    """

    # 1. Baseline screen
    screened = route_df[
        (route_df.OCCUPANCY_RATE_AVG >= min_load) &
        (route_df.TOTAL_TRIPS_TAKEN >= min_round_trips)].copy()

    screened["MEAN_DELAY"] = (screened.ARR_DELAY_AVG + screened.DEP_DELAY_AVG) / 2
    screened = screened[screened.MEAN_DELAY <= max_avg_delay]

    # 2 · Min–max normalisation helpers
    def minmax(series):
        return (series - series.min()) / (series.max() - series.min())

    screened["PROFIT_NORM"] = minmax(screened.PROFIT)
    screened["DELAY_NORM"]  = 1 - minmax(screened.MEAN_DELAY)   # lower delay = higher score
    screened["OCC_NORM"]    = minmax(screened.OCCUPANCY_RATE_AVG)

    # 3.  Composite PPD score
    screened["PTD_SCORE"] = (
        screened.PROFIT_NORM * weight_profit +
        screened.DELAY_NORM  * weight_delay  +
        screened.OCC_NORM    * weight_occupancy ).round(2)

    # 4.  Return top five and drop helper cols
    helper_cols = ["PROFIT_NORM", "DELAY_NORM", "OCC_NORM", "MEAN_DELAY"]
    top_five_rec_routes = (screened.sort_values("PTD_SCORE", ascending=False).head(5))
                           # .drop(columns=helper_cols))

    return top_five_rec_routes


# In[70]:


top_5_routes_to_invest =  rank_routes_ptd(merged_df)
plot_table( top_5_routes_to_invest[['ROUTE','PTD_SCORE']] )


# In[71]:


matrix = top_5_routes_to_invest.set_index("ROUTE")[["PROFIT_NORM", "DELAY_NORM", "OCC_NORM", "PTD_SCORE"]]

matrix.columns = ["Profit", "Timelinsess", "Occupancy", "PTD Score"]

fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale=px.colors.diverging.Geyser)
fig.update_layout(title={'text': 'Top-5 Recommended Routes Scorecard <br><sub></sub>','x': 0.5,'xanchor': 'center'},
                  width=950,height=500, plot_bgcolor='white',paper_bgcolor='white',
                  title_font=dict(size=18, color='black'),
                  font=dict(color='black'),yaxis=dict(gridcolor='lightgrey'))
fig.show()


# ##### **Observations**
# 1. **DCA – ORD** is the top earner in the group with high profit margin and runs a flights ahead of schedule.<br>
#    **Reason:** It posts the highest profit index and averages a 6.3 min departure delay,
#      so it combines cash strength with early push-backs.<br>
# 
# 2. **BOS – LGA** is our on-time champion and still flies roughly three-quarters full - an efficient, brand-friendly mix.<br>
#    **Reason:** It owns the best punctuality of -7.3 min average delay and highest occupancy score (0.49),
#      proving to be both timely and well-filled.<br>
# 
# 3. **ATL – CLT** generates strong cash flow; trimming a couple more minutes of delay would push its score even higher.<br>
#    **Reason:** Profit sits at 0.92 of the best route while average delay is only -4.7 min - small, targeted schedule tweaks could close the gap to #2
# 
# 4. **DCA – LGA** delivers steady earnings with consistently reliable on-time performance, making it a solid, low-risk choice.<br>
#    **Reason:** Profit (0.80) and punctuality -6.0 min are both comfortably above our investment thresholds,
#      giving balanced, dependable returns.
# 
# 5. **LAX – SFO** brings in substantial revenue but shows moderate delays; tightening the timetable could unlock more profit.<br>
#    **Reason:** With a profit index of 0.82 it’s a big earner, yet its -5.2 min delay lags the punctuality leaders - an obvious improvement lever
# 

# ##### **Insights**
# 1. Profit isn’t everything. BOS-LGA climbs from #8 to #2 on timeliness + demand despite a lower margin, underscoring the weight of service quality in the index.
# 
# 2. All five routes depart ahead of schedule on average, showing room to tighten block-times and free aircraft minutes for additional rotations.
# 
# 3. Occupancy factors sit > 72 % across the board, indicating resilient demand and limited risk of under utilisation.

# ## Question 4 -  Calculating the number of roundtrips it will take to breakeven between these five recommended routes

# To calculate the number of round-trip flights to breakeven, we need to divide the cost of the airplane with the expected profits we will earn with the 5 routes.

# In[72]:


upfront_cost = 90_000_000
top_5_routes_breakeven = top_5_routes_to_invest.copy()  # From Question 3
top_5_routes_breakeven['PROFIT_PER_ROUND_TRIP'] = top_5_routes_breakeven['PROFIT'] / top_5_routes_breakeven['TOTAL_TRIPS_TAKEN']
top_5_routes_breakeven['BREAKEVEN_TRIPS'] = upfront_cost / top_5_routes_breakeven['PROFIT_PER_ROUND_TRIP']
top_5_routes_breakeven['BREAKEVEN_TRIPS']=(top_5_routes_breakeven['BREAKEVEN_TRIPS']*0.5).astype(int)


# In[73]:


plot_table(top_5_routes_breakeven[['ROUTE','BREAKEVEN_TRIPS']].sort_values("BREAKEVEN_TRIPS", ascending=False))


# In[74]:


# Visual representation
colors = px.colors.qualitative.Pastel
fig1 = plot_bar_chart(df=top_5_routes_breakeven, x_axis='ROUTE', y_axis='BREAKEVEN_TRIPS',text='BREAKEVEN_TRIPS',
                     x_axis_title='ROUTE',y_axis_title='Number of Trips to Breakeven',title='Top 5 Recommendation Routes',)
fig1.update_traces(marker_color=colors[:len(top_5_routes_breakeven)])
fig1.update_layout(title={'text': 'Top 5 Recommendation Routes<br><sub>By Round-Trips to Breakeven</sub>','x': 0.5,'xanchor': 'center'})
fig1.show()
fig2 = plot_interactive_pie(top_5_routes_breakeven, label="ROUTE", value="BREAKEVEN_TRIPS", title="Breakeven Trips",color="BREAKEVEN_TRIPS")
fig2.update_layout(title={'text': 'Breakeven Trips<br><sub></sub>','x': 0.5,'xanchor': 'center'})
fig2.show()


# ##### **Observations:**
# 1. **LAX-SFO**: Highest number of round trips required to break even, indicating its high total distance and costs.
# 
# 2. **ATL-CLT:** Lowest number of round trips required to break even, indicating higher profit per flight.
# 
# 3. **BOS-LGA:** Moderate number of round trips to break even due to relatively lower profit margin per flight.
# 
# 4. **DCA-LGA** and **DCA-ORD**: Balanced in terms of profitability and breakeven points, showcasing efficient operations and revenue generation.

# ##### **Insights:**
# 1. **Wide Breakeven Spread**:  The gap between the fastest (ATL-CLT at 942 trips) and the slowest (LAX-SFO at 2.3k trips) is over 1.4k rotations—highlighting how distance and cost structure dramatically affect payback time.
# 
# 2. **East Coast Efficiency**:  Three of the five shortest breakeven routes (DCA-ORD, ATL-CLT, DCA-LGA) are on the East Coast, reflecting both higher yields and shorter stage lengths.
# 
# 3. **Midpoint Clustering**:  BOS-LGA (1.4k trips) and DCA-LGA (1k trips) sit near the chart’s median, suggesting moderate distance and balanced per-flight margins.

# ## Question 5 -  Key Performance Indicators (KPI’s) that you recommend tracking in the future to measure the success of the round trip routes that you recommend.
# 

# #### KPIs that are already covered previously in our analysis
# 1. **Total Profit per Route (Revenue-Cost)**
# 2. **Average Departure & Arrival Delay**
# 3. **Occupancy Rate**
# 4. **Number of Round-Trip Flights**
# 5. **Composite Profit-Punctuality-Demand (PTD) Score**

# #### 6 Key Additional KPIs to consider in the future
# 
# 1. **Cost per Available Seat-Mile (CASM)** <br>
#    **Measure:** Total operating cost (fuel, maintenance, crew, airport fees, delay penalties) divided by seats & distance flown. <br>
#    **Rationale & Action:** Rising CASM indicates unit costs are undermining margins. Investigate major cost drivers like fuel costs, ground turnaround times, or airport fees—and implement targeted efficiencies.
# 
# 2. **Seat Distribution** <br>
#    **Measure:** Proportion of business, premium and economy seats per each route. <br>
#    **Rationale & Action:** Changes in the share of seat type directly influence revenue per flight. Monitor shifts and reconfigure cabin layouts or adjust fare tiers to capitalise on premium demand or growing leisure traffic.
# 
# 3. **Promotions & Pricing Effectiveness** <br>
#    **Measure:** Change in bookings and load-factor resulting from discounts or special offers <br>
#    **Rationale & Action:** Well-targeted promotions can fill seats without damaging long-term fares. Tracking net yield impact ensures discounts drive profitable demand.
# 
# 4. **Customer Experience & Satisfaction** <br>
#    **Measure:** Post flight survey score or Net Promoter Score collected from passengers on each route. <br>
#    **Rationale & Action:** Service quality underpins repeat business and brand reputation. Correlating satisfaction with on-time performance and comfort metrics helps protect loyalty.
# 
# 5. **Profit Margin** <br>
#    **Measure:** Net profit as a percentage of total revenue for a given route. <br>
#    **Rationale & Action:** A route’s margin reveals its efficiency beyond raw revenue figures. Tracking net profit helps us spot hidden cost pressure points and take corrective action.
# 
# 6. **Delay Causation Analysis** <br>
#    **Measure:** Segmentation of departure and arrival delays by root cause (weather, technical, crew, ATC, etc.) <br>
#    **Rationale & Action:** Identify the dominant delay drivers on each route and deploy targeted remedies such as optimizing dispatch protocols, predictive maintenance, or slot improvements with air traffic control.

# #### **Conclusion**
# By tracking these KPIs, we gain clear visibility into cost efficiency, revenue potential, service quality, and operational consistency—empowering us to quickly identify and address issues, boost profitability, enhance reliability, and drive sustainable growth.
