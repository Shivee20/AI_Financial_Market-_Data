import kagglehub
ai_financial_and_market_data_path = kagglehub.dataset_download('rohitgrewal/ai-financial-and-market-data')

print('Data source import complete.')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.show()

df = pd.read_csv(r"C:\Users\Shivee\Desktop\Binomial\Fin_Market_Analysis\AI_Fin_Data.csv", parse_dates = ["Date"], dayfirst = True)
df
df.info()
df['Date'] = pd.to_datetime(df['Date'])
df.info()
df.head()

# Show the Companies Name
df['Company'].unique()
# Create a new column for 'Year' only
df['Year'] = df['Date'].dt.year
df.head()
df

# Show the years
df['Year'].unique()
df['Year'].nunique()
df
df['Event'].value_counts()
df[df['Event'] == 'GPT-4 release']
df.isnull().sum()
df.head()

print("Company's spending for R & D in $Bn")
RD = df.groupby('Company')['R&D_Spending_USD_Mn'].sum()/1000
RD
# Draw a Bar Plot to show the amount spent on R & D by the companies

plt.bar(RD.index, RD.values, color = ['cyan', 'black', 'magenta'])

plt.title( "R&D Spending by the companies")
plt.xlabel("Company")
plt.ylabel("Amount in USD_$Bn")
plt.show()
df.head()

print("Company's AI_Revenue_USD_Bn :")
rev = df.groupby('Company')['AI_Revenue_USD_Mn'].sum()/1000
rev
# Draw a Bar Plot to show the Revenues of the companies

plt.bar(rev.index, rev.values, color = ['cyan', 'black', 'magenta'], width = 0.2)

plt.title( "Revenue earned by the companies")
plt.xlabel("Company")
plt.ylabel("Amount in USD_$Bn")
plt.show()
# Bar plots to show expenditure & revenue of the companies

plt.figure(figsize = (10,4))

plt.subplot(1,2,1)

plt.bar(RD.index, RD.values, color = ['cyan', 'black', 'magenta'])

plt.title( "R&D Spending by the companies")
plt.xlabel("Company")
plt.ylabel("Amount in USD_$Bn")

plt.subplot(1,2,2)

plt.bar(rev.index, rev.values, color = ['cyan', 'black', 'magenta'], width = 0.2)

plt.title( "Revenue earned by the companies")
plt.xlabel("Company")
plt.ylabel("Amount in USD_$Bn")

plt.show()
df.head()
plt.figure(figsize = (10,5))

plt.plot(df['Date'], df['Stock_Impact_%'], color = 'green')

plt.title("Change in Stock value")
plt.xlabel("Date ('Year')")
plt.ylabel("Stock_Impact_%")
plt.show()
df.head()
data_openai =  df [df['Company'] == 'OpenAI']
data_google = df [df['Company'] == 'Google']
data_meta = df [df['Company'] == 'Meta']
data_meta
data_openai
data_google
plt.figure(figsize = (10,5))

plt.plot(data_openai['Date'], data_openai['Stock_Impact_%'], color = 'm')

plt.title("Change in Stock value of OpenAI")
plt.xlabel("Date")
plt.ylabel("Stock_Impact_%")
plt.show()
data_google
plt.figure(figsize = (10,5))

plt.plot( data_google['Date'], data_google['Stock_Impact_%'], color = 'c')
plt.title("Change in Stock value of Google")
plt.xlabel("Date")
plt.ylabel("Stock_Impact_%")
plt.show()
data_meta
plt.figure(figsize = (10,5))

plt.plot( data_meta['Date'], data_meta['Stock_Impact_%'], color = 'black' )
plt.title("Change in Stock value of Meta")
plt.xlabel("Date")
plt.ylabel("Stock_Impact_%")
plt.show()
df.head()
data_openai

# OpenAI's Events when Maximum Stock Impact was observed
data_openai.sort_values( by = 'Stock_Impact_%', ascending = False)
data_google
# Google's Events when Maximum Stock Impact was observed
data_google.sort_values( by = 'Stock_Impact_%', ascending = False )
data_meta
data_meta.sort_values(by = 'Stock_Impact_%', ascending = False )
df.head()
plt.figure(figsize = (10,5))
sns.scatterplot(x = 'Date', y = 'AI_Revenue_Growth_%', data = df, hue = 'Company')
plt.show()
df.sort_values(by = ['AI_Revenue_Growth_%'])
data_openai
plt.plot( data_openai['Date'], data_openai['AI_Revenue_Growth_%'], color = 'm')
plt.show()
data_google
# Google's AI Revenue Growth year-by-year

plt.plot( data_google['Date'], data_google['AI_Revenue_Growth_%'], color = 'c')
plt.show()
data_meta
# Meta's AI Revenue Growth year-by-year

plt.plot( data_meta['Date'], data_meta['AI_Revenue_Growth_%'], color = 'black')
plt.show()
sns.heatmap( df.corr(numeric_only = True) )
df.head()
spend = df.groupby('Year')['R&D_Spending_USD_Mn'].sum()
spend
# Showing the Amounnt spent on R & D

plt.plot(spend.index, spend.values, color = 'r')

plt.title("Combined R&D Spending Year-by-Year")
plt.xlabel("Year")
plt.ylabel("Amount in USD_$Mn")
plt.show()
revenue = df.groupby('Year')['AI_Revenue_USD_Mn'].sum()
revenue

# Showing the Revenue earned 
plt.plot( revenue.index, revenue.values, color = 'g')
plt.title( "Combined Revenue Earned Year-by-Year")
plt.xlabel("Year")
plt.ylabel("Amount in USD_$Mn")
plt.show()
plt.plot(spend.index, spend.values, color = 'r')
plt.plot( revenue.index, revenue.values, color = 'g')

plt.title( "Combined Expenditure vs Revenue Year-by-Year", fontsize = 12)
plt.xlabel("Year")
plt.ylabel("Amount in USD_$Mn")
plt.legend(['Expenditure', 'Revenue'])
plt.show()

# Pairplot to show the relations between the columns
sns.pairplot(df);
df.head()
# Showing the various Events
df.Event.value_counts()
# Checking for a particular event
df[ df.Event == 'TensorFlow open-source release']
tf = df.loc[ 3955 : 3975 ]
tf
# Showing the Impact with a line chart

plt.figure(figsize = (10,4))

plt.plot( tf['Date'], tf['Stock_Impact_%'], color = 'c')
plt.title("Comparison before and after the release of TensorFlow open-source")
plt.xlabel("Date")
plt.ylabel("Change in Stock %")
plt.show()
# Checking for a particular event
df[ df.Event == 'GPT-4 release']
gpt4 = df.loc[ 2984 : 3004]
gpt4
# Showing the Impact with a line chart

plt.figure(figsize = (10,4))

plt.plot( gpt4['Date'], gpt4['Stock_Impact_%'], color = 'm')

plt.title("Comparison before and after the release of GPT-4")
plt.xlabel("Date")
plt.ylabel("Change in Stock %")
plt.show()
# Daily Average impact on the Stocks of the companies
df.groupby('Company')['Stock_Impact_%'].mean()*100
# Daily Average Expenditure on R & D by the companies
df.groupby('Company')['R&D_Spending_USD_Mn'].mean()
# Maximum impact % on a company's stocks
df.groupby('Company')['Stock_Impact_%'].max()
df.head(2)

# Highest change in the index
stocks = df.groupby(['Year', 'Company'])['Stock_Impact_%'].max()
stocks
stocks.plot(kind = 'barh', color = ['r', 'black', 'm'])
plt.title("change in index")
plt.show()