# Question 1: Explore and clean the data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("automobileEDA.csv", na_values="?")

print(data.head(10))

# Check the number of rows and columns.
print("Rows and columns:", data.shape)

# info shows column names, data types and the number of filled values.
data.info()

# Count the empty entries in each column.
print(data.isna().sum())

print("\n")

# Count rows that repeat all the values of an earlier row.
print("Duplicate rows:", data.duplicated().sum())

# Look at the records with no stroke measurement.
missing_stroke = data[data["stroke"].isna()]
print(missing_stroke[["make", "engine-type", "stroke"]])

print("\n")

# Look at the record with no horsepower category.
missing_category = data[data["horsepower-binned"].isna()]
print(missing_category[["make", "horsepower", "horsepower-binned"]])

# Check the horsepower ranges of the existing categories.
print(data.groupby("horsepower-binned")["horsepower"].min())

print("\n")

print(data.groupby("horsepower-binned")["horsepower"].max())

# Fill the one missing category as High because its horsepower is 262.
data["horsepower-binned"] = data["horsepower-binned"].fillna("High")

# Mark the four stroke values that do not apply to rotary engines.
data["stroke-not-applicable"] = data["stroke"].isna()

# Remove exact duplicate rows, if any. There are none in this file.
data = data.drop_duplicates()

print("\n")

print("Missing horsepower categories:", data["horsepower-binned"].isna().sum())
print("Stroke values marked not applicable:", data["stroke-not-applicable"].sum())

# Store the names of the numerical measurements in a Python list.
numeric_columns = [
    "normalized-losses", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price",
    "city-L/100km"
]

# Select those columns from the table.
numeric_data = data[numeric_columns]

# Find the lower and upper quartiles of each numerical column.
q1 = numeric_data.quantile(0.25)
q3 = numeric_data.quantile(0.75)

# The interquartile range is the distance between the two quartiles.
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

# Flag values below or above these limits. The | symbol means OR.
below_limit = numeric_data < lower_limit
above_limit = numeric_data > upper_limit
outlier_flags = below_limit | above_limit

# Count the flagged entries in each column.
# Print this result because Python scripts do not display it automatically.
print(outlier_flags.sum())

# Copy the price outlier flags into the main table.
data["price-outlier"] = outlier_flags["price"]

# Display the prices and makes of the flagged cars.
expensive_cars = data[data["price-outlier"]]
print(expensive_cars[["make", "engine-size", "horsepower", "price"]])

print("\n")

print(f"Lower price limit: {lower_limit['price']}")
print(f"Upper price limit: {upper_limit['price']}")

# Save the cleaned dataset.
# index=False prevents pandas from adding an extra row-number column.
data.to_csv("Turyahebwa_Data.csv", index=False)

print("Cleaned rows and columns:", data.shape)
print("Duplicate rows:", data.duplicated().sum())
print("Remaining blanks:", data.isna().sum().sum())





# Question 2: Descriptive statistics and distributions

# describe gives counts, averages, standard deviations and quartiles.
# T turns the table so each measurement occupies one row.
# round(2) displays numbers to two decimal places.
numeric_summary = data[numeric_columns].describe().T
# Print this result because Python scripts do not display it automatically.
print(numeric_summary.round(2))

# Draw a histogram for every numerical measurement.
data[numeric_columns].hist(bins=15, figsize=(14, 12), color="green", edgecolor="white")
plt.tight_layout()
plt.show()

# value_counts counts the records in each category.
print(data["make"].value_counts())

print("\n")
print(data["body-style"].value_counts())

print("\n")
print(data["drive-wheels"].value_counts())

print("\n")
print(data["aspiration"].value_counts())

# Count the other text categories.
print(data["num-of-doors"].value_counts())

print("\n")
print(data["engine-location"].value_counts())

print("\n")
print(data["engine-type"].value_counts())

print("\n")
print(data["num-of-cylinders"].value_counts())

print("\n")
print(data["fuel-system"].value_counts())

# Count horsepower groups and the categories stored as numerical codes.
print(data["horsepower-binned"].value_counts())

print("\n")
print(data["symboling"].value_counts())

print("\n")
print(data["diesel"].value_counts())

print("\n")
print(data["gas"].value_counts())

# Show how many cars belong to each body style.
data["body-style"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Number of cars by body style")
plt.xlabel("Body style")
plt.ylabel("Number of cars")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Show how many cars belong to each drive-wheel group.
data["drive-wheels"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Number of cars by drive wheels")
plt.xlabel("Drive wheels")
plt.ylabel("Number of cars")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# A horizontal bar chart keeps the make names readable.
data["make"].value_counts().plot(kind="barh", figsize=(8, 7), color="green")
plt.title("Number of cars by make")
plt.xlabel("Number of cars")
plt.ylabel("Make")
plt.tight_layout()
plt.show()





# Question 3: Group prices by car characteristics

# Put cars into drive-wheel groups and describe price within each group.
# Print this result because Python scripts do not display it automatically.
print(data.groupby("drive-wheels")["price"].describe().round(2))

# Calculate and plot the average price for each drive-wheel group.
drive_means = data.groupby("drive-wheels")["price"].mean()
drive_means.plot(kind="bar", color="skyblue")
plt.title("Average price by drive wheels")
plt.xlabel("Drive wheels")
plt.ylabel("Average price in dataset units")
plt.xticks(rotation=0)
plt.show()

# Describe price separately for each body style.
# Print this result because Python scripts do not display it automatically.
print(data.groupby("body-style")["price"].describe().round(2))

# Calculate and plot the average price for each body style.
body_means = data.groupby("body-style")["price"].mean()
body_means.plot(kind="bar", color="skyblue")
plt.title("Average price by body style")
plt.xlabel("Body style")
plt.ylabel("Average price in dataset units")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Group by both drive wheels and body style at the same time.
joint_means = data.groupby(["drive-wheels", "body-style"])["price"].mean()

# Sort the averages from highest to lowest.
# Print this result because Python scripts do not display it automatically.
print(joint_means.sort_values(ascending=False).round(2))



# 4(a) Numerical variables and price

# Calculate the correlation between every pair of numerical columns.
correlation_table = data[numeric_columns].corr()

# Keep the price column from that correlation table.
price_correlations = correlation_table["price"]

# Remove price itself because a column always correlates perfectly with itself.
price_correlations = price_correlations.drop("price")

# Show the correlations from the most positive to the most negative.
# Print this result because Python scripts do not display it automatically.
print(price_correlations.sort_values(ascending=False).round(3))

# Select price and graph its distribution.
price = data["price"]

# Calculate the three measures of central tendency.
price_mean_value = price.mean()
price_median_value = price.median()
price_mode_value = price.mode()[0]

# Set the chart size.
plt.figure(figsize=(6, 4))

# Draw the histogram.
# bins controls the number of bars.
# kde=True adds a smooth distribution curve.
sns.histplot(price, bins=10, kde=True, color="green", edgecolor="white", alpha=0.4)

# Add a red dashed line for the mean.
plt.axvline(price_mean_value, color="red", linestyle="--", label=f"Mean = {price_mean_value:.2f}")

# Add a blue dash-dot line for the median.
plt.axvline(price_median_value, color="blue", linestyle="-.", label=f"Median = {price_median_value:.2f}")

# Add an orange dotted line for the mode.
plt.axvline(price_mode_value, color="darkorange", linestyle=":", linewidth=2, label=f"Mode = {price_mode_value:.2f}")

# Add the title, axis labels and legend.
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Select engine size graph its distribution.
engine_size = data["engine-size"]

# Calculate the three measures of central tendency.
engine_size_mean_value = engine_size.mean()
engine_size_median_value = engine_size.median()
engine_size_mode_value = engine_size.mode()[0]

# Set the chart size.
plt.figure(figsize=(6, 4))

# Draw the histogram.
# bins controls the number of bars.
# kde=True adds a smooth distribution curve.
sns.histplot(engine_size, bins=10, kde=True, color="green", edgecolor="white", alpha=0.4)

# Add a red dashed line for the mean.
plt.axvline(engine_size_mean_value, color="red", linestyle="--", label=f"Mean = {engine_size_mean_value:.2f}")

# Add a blue dash-dot line for the median.
plt.axvline(engine_size_median_value, color="blue", linestyle="-.", label=f"Median = {engine_size_median_value:.2f}")

# Add an orange dotted line for the mode.
plt.axvline(engine_size_mode_value, color="darkorange", linestyle=":", linewidth=2, label=f"Mode = {engine_size_mode_value:.2f}")

# Add the title, axis labels and legend.
plt.title("Engine Size Distribution")
plt.xlabel("Engine size on the supplied scale")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Each dot compares engine size and price for one car.
data.plot(kind="scatter", x="engine-size", y="price", color="green")
plt.title("Engine size and price")
plt.xlabel("Engine size on the supplied scale")
plt.ylabel("Price")
plt.show()

# Select curb weight and graph its distribution.
curb_weight = data["curb-weight"]

# Calculate the three measures of central tendency.
curb_weight_mean_value = curb_weight.mean()
curb_weight_median_value = curb_weight.median()
curb_weight_mode_value = curb_weight.mode()[0]

# Set the chart size.
plt.figure(figsize=(6, 4))

# Draw the histogram.
# bins controls the number of bars.
# kde=True adds a smooth distribution curve.
sns.histplot(curb_weight, bins=10, kde=True, color="green", edgecolor="white", alpha=0.4)

# Add a red dashed line for the mean.
plt.axvline(curb_weight_mean_value, color="red", linestyle="--", label=f"Mean = {curb_weight_mean_value:.2f}")

# Add a blue dash-dot line for the median.
plt.axvline(curb_weight_median_value, color="blue", linestyle="-.", label=f"Median = {curb_weight_median_value:.2f}")

# Add an orange dotted line for the mode.
plt.axvline(curb_weight_mode_value, color="darkorange", linestyle=":", linewidth=2, label=f"Mode = {curb_weight_mode_value:.2f}")

# Add the title, axis labels and legend.
plt.title("Curb weight Distribution")
plt.xlabel("Curb weight on the supplied scale")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Compare the weight of the car with its price.
data.plot(kind="scatter", x="curb-weight", y="price", color="green")
plt.title("Curb weight and price")
plt.xlabel("Curb weight on the supplied scale")
plt.ylabel("Price")
plt.show()

# Select Horse power and graph its distribution.
horse_power = data["horsepower"]

# Calculate the three measures of central tendency.
horse_power_mean_value = horse_power.mean()
horse_power_median_value = horse_power.median()
horse_power_mode_value = horse_power.mode()[0]

# Set the chart size.
plt.figure(figsize=(6, 4))

# Draw the histogram.
# bins controls the number of bars.
# kde=True adds a smooth distribution curve.
sns.histplot(horse_power, bins=10, kde=True, color="green", edgecolor="white", alpha=0.4)

# Add a red dashed line for the mean.
plt.axvline(horse_power_mean_value, color="red", linestyle="--", label=f"Mean = {horse_power_mean_value:.2f}")

# Add a blue dash-dot line for the median.
plt.axvline(horse_power_median_value, color="blue", linestyle="-.", label=f"Median = {horse_power_median_value:.2f}")

# Add an orange dotted line for the mode.
plt.axvline(horse_power_mode_value, color="darkorange", linestyle=":", linewidth=2, label=f"Mode = {horse_power_mode_value:.2f}")

# Add the title, axis labels and legend.
plt.title("Horse Power Distribution")
plt.xlabel("Horse Power")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Compare engine power with price.
data.plot(kind="scatter", x="horsepower", y="price", color="green")
plt.title("Horsepower and price")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.show()

# Select Horse power and graph its distribution.
highway_mpg = data["highway-mpg"]

# Calculate the three measures of central tendency.
highway_mpg_mean_value = highway_mpg.mean()
highway_mpg_median_value = highway_mpg.median()
highway_mpg_mode_value = highway_mpg.mode()[0]

# Set the chart size.
plt.figure(figsize=(6, 4))

# Draw the histogram.
# bins controls the number of bars.
# kde=True adds a smooth distribution curve.
sns.histplot(highway_mpg, bins=10, kde=True, color="green", edgecolor="white", alpha=0.4)

# Add a red dashed line for the mean.
plt.axvline(highway_mpg_mean_value, color="red", linestyle="--", label=f"Mean = {highway_mpg_mean_value:.2f}")

# Add a blue dash-dot line for the median.
plt.axvline(highway_mpg_median_value, color="blue", linestyle="-.", label=f"Median = {highway_mpg_median_value:.2f}")

# Add an orange dotted line for the mode.
plt.axvline(highway_mpg_mode_value, color="darkorange", linestyle=":", linewidth=2, label=f"Mode = {highway_mpg_mode_value:.2f}")

# Add the title, axis labels and legend.
plt.title("Highway mpg Distribution")
plt.xlabel("Highway mpg")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Compare highway fuel economy with price.
data.plot(kind="scatter", x="highway-mpg", y="price", color="green")
plt.title("Highway fuel economy and price")
plt.xlabel("Highway miles per gallon")
plt.ylabel("Price")
plt.show()


# 4(b) Categorical variables and price

# A box plot compares the middle and spread of prices across groups.
data.boxplot(column="price", by="drive-wheels", figsize=(7, 5))
plt.title("Prices by drive wheels")
plt.suptitle("")
plt.xlabel("Drive wheels")
plt.ylabel("Price")
plt.show()

# Compare price distributions for the five body styles.
data.boxplot(column="price", by="body-style", figsize=(9, 5))
plt.title("Prices by body style")
plt.suptitle("")
plt.xlabel("Body style")
plt.ylabel("Price")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Compare average prices across several other categories.
print("AVERAGE PRICE BY MAKE")
print(data.groupby("make")["price"].mean().round(2))

print("\n")
print("AVERAGE PRICE BY ASPIRATION")
print(data.groupby("aspiration")["price"].mean().round(2))

print("\n")
print("AVERAGE PRICE BY HORSEPOWER CATEGORY")
print(data.groupby("horsepower-binned")["price"].mean().round(2))

print("\n")
print("AVERAGE PRICE BY NUMBER OF DOORS")
print(data.groupby("num-of-doors")["price"].mean().round(2))

# Examine the remaining categorical attributes using the same method.
print("AVERAGE PRICE BY ENGINE LOCATION")
print(data.groupby("engine-location")["price"].mean().round(2))

print("\n")
print("AVERAGE PRICE BY ENGINE TYPE")
print(data.groupby("engine-type")["price"].mean().round(2))

print("\n")
print("AVERAGE PRICE BY CYLINDER CATEGORY")
print(data.groupby("num-of-cylinders")["price"].mean().round(2))

print("\n")
print("AVERAGE PRICE BY FUEL SYSTEM")
print(data.groupby("fuel-system")["price"].mean().round(2))

print("\n")
print("AVERAGE PRICE BY RISK CODE")
print(data.groupby("symboling")["price"].mean().round(2))

print("\n")
print("AVERAGE PRICE BY DIESEL INDICATOR")
print(data.groupby("diesel")["price"].mean().round(2))


# 4(c) Two categorical variables

# A cross-tabulation counts each combination of two categories.
category_counts = pd.crosstab(data["body-style"], data["drive-wheels"])
# Print this result because Python scripts do not display it automatically.
print(category_counts)

# Show the combination counts as a stacked bar chart.
category_counts.plot(kind="bar", stacked=False, figsize=(9, 5))
plt.title("Drive wheels within each body style")
plt.xlabel("Body style")
plt.ylabel("Number of cars")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Group cars by both body style and drive wheels.
# count gives the number of cars; mean gives their average price.
combined_comparison = data.groupby(
    ["body-style", "drive-wheels"]
)["price"].agg(["count", "mean"])

# Show the number and average price for each combination.
# Print this result because Python scripts do not display it automatically.
print(combined_comparison.round(2))

# Put the average prices into separate drive-wheel columns.
average_prices = combined_comparison["mean"].unstack()

# Draw a grouped bar chart.
average_prices.plot(kind="bar", figsize=(10, 5))

plt.title("Average Price by Body Style and Drive Wheels")
plt.xlabel("Body style")
plt.ylabel("Average price")
plt.legend(title="Drive wheels")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()




# Question 5: Feature selection and recommendation

# Remove the two numerical variables excluded from this screening step.
candidate_correlations = price_correlations.drop(["stroke", "city-L/100km"])

# abs changes negative values to positive values for strength comparisons.
correlation_strength = candidate_correlations.abs()

# Keep features whose absolute correlation is at least 0.70.
selected_features = correlation_strength[correlation_strength >= 0.70]

# Put the strongest selected features first.
selected_features = selected_features.sort_values(ascending=False)
# Print this result because Python scripts do not display it automatically.
print(selected_features.round(3))

# Plot the strength of the five selected relationships.
selected_features.plot(kind="barh", color="green", figsize=(8, 4))
plt.title("Features selected by the correlation filter")
plt.xlabel("Absolute correlation with price")
plt.ylabel("Feature")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
