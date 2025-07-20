import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

# Load the data
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB2\Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Set correct column names manually
date_col = 'Date'
close_col = 'Price'
chg_col = 'Chg%'

# Convert Date column to datetime
df[date_col] = pd.to_datetime(df[date_col])
df['DayOfWeek'] = df[date_col].dt.day_name()

# 1. Mean and Variance of Close Price
close_prices = df[close_col]
mean_price = statistics.mean(close_prices)
var_price = statistics.variance(close_prices)
print(f"Mean Close Price: ₹{mean_price:.2f}")
print(f"Variance of Close Price: ₹{var_price:.2f}")

# 2. Mean Close Price on Wednesdays
wed_close_prices = df[df['DayOfWeek'] == 'Wednesday'][close_col]
mean_wed = statistics.mean(wed_close_prices)
print(f"Mean Close Price on Wednesdays: ₹{mean_wed:.2f}")

# 3. Mean Close Price in April
april_prices = df[df[date_col].dt.month == 4][close_col]
mean_april = statistics.mean(april_prices)
print(f"Mean Close Price in April: ₹{mean_april:.2f}")

# 4. Probability of loss (Chg% < 0)
loss_prob = (df[chg_col] < 0).mean()
print(f"Probability of making a loss: {loss_prob:.2f}")

# 5. Probability of profit on Wednesday
wednesday_df = df[df['DayOfWeek'] == 'Wednesday']
profit_wed_prob = (wednesday_df[chg_col] > 0).mean()
print(f"Probability of profit on Wednesday: {profit_wed_prob:.2f}")

# 6. Conditional probability of profit given it's Wednesday
total_wed = len(wednesday_df)
profitable_wed = (wednesday_df[chg_col] > 0).sum()
cond_prob = profitable_wed / total_wed
print(f"Conditional Probability (Profit | Wednesday): {cond_prob:.2f}")

# 7. Scatter plot: Chg% vs DayOfWeek
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='DayOfWeek', y=chg_col, jitter=True)
plt.title('Chg% vs Day of Week')
plt.ylabel('Change %')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
