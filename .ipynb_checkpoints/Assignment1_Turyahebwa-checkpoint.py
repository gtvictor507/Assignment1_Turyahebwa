# -*- coding: utf-8 -*-
# Automobile price analysis: Victor Turyahebwa
# Run this script from the folder containing automobileEDA.csv.


# Next step
# pandas helps us read and work with tables.
import pandas as pd

# matplotlib helps us draw graphs.
import matplotlib.pyplot as plt

# Read the CSV file and store the table in a variable called data.
# A question mark, if present, will be treated as a missing value.
data = pd.read_csv("automobileEDA.csv", na_values="?")

# Show the first five rows.
print(data.head())


# Next step
# shape shows the number of rows and columns, in that order.
print("Rows and columns:", data.shape)

# info shows column names, data types and the number of filled values.
data.info()


# Next step
# Count the empty entries in each column.
print(data.isna().sum())

# Count rows that repeat all the values of an earlier row.
print("Duplicate rows:", data.duplicated().sum())


# Next step
# Look at the records with no stroke measurement.
missing_stroke = data[data["stroke"].isna()]
print(missing_stroke[["make", "engine-type", "stroke"]])

# Look at the record with no horsepower category.
missing_category = data[data["horsepower-binned"].isna()]
print(missing_category[["make", "horsepower", "horsepower-binned"]])


# Next step
# Check the horsepower ranges of the existing categories.
print(data.groupby("horsepower-binned")["horsepower"].min())
print(data.groupby("horsepower-binned")["horsepower"].max())

# Fill the one missing category as High because its horsepower is 262.
data["horsepower-binned"] = data["horsepower-binned"].fillna("High")

# Mark the four stroke values that do not apply to rotary engines.
data["stroke-not-applicable"] = data["stroke"].isna()

# Remove exact duplicate rows, if any. There are none in this file.
data = data.drop_duplicates()

print("Missing horsepower categories:", data["horsepower-binned"].isna().sum())
print("Stroke values marked not applicable:", data["stroke-not-applicable"].sum())


# Next step
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
print(outlier_flags.sum())


# Next step
# Copy the price outlier flags into the main table.
data["price-outlier"] = outlier_flags["price"]

# Display the prices and makes of the flagged cars.
expensive_cars = data[data["price-outlier"]]
print(expensive_cars[["make", "engine-size", "horsepower", "price"]])

print("Lower price limit:", lower_limit["price"])
print("Upper price limit:", upper_limit["price"])


# Next step
# Save the cleaned dataset using the required surname.
# index=False prevents pandas from adding an extra row-number column.
data.to_csv("Turyahebwa Data.csv", index=False)

print("Cleaned rows and columns:", data.shape)
print("Duplicate rows:", data.duplicated().sum())
print("Remaining blanks:", data.isna().sum().sum())


# Next step
# describe gives counts, averages, standard deviations and quartiles.
# T turns the table so each measurement occupies one row.
# round(2) displays numbers to two decimal places.
numeric_summary = data[numeric_columns].describe().T
print(numeric_summary.round(2))


# Next step
# Draw a histogram of car prices.
data["price"].plot(kind="hist", bins=20, color="steelblue", edgecolor="white")
plt.title("Distribution of car prices")
plt.xlabel("Price in dataset units")
plt.ylabel("Number of cars")
plt.show()


# Next step
# Draw a histogram for every numerical measurement.
data[numeric_columns].hist(bins=15, figsize=(14, 12), color="steelblue")
plt.tight_layout()
plt.show()


# Next step
# value_counts counts the records in each category.
print("MAKE")
print(data["make"].value_counts())

print("BODY STYLE")
print(data["body-style"].value_counts())

print("DRIVE WHEELS")
print(data["drive-wheels"].value_counts())

print("ASPIRATION")
print(data["aspiration"].value_counts())


# Next step
# Count the other text categories.
print("NUMBER OF DOORS")
print(data["num-of-doors"].value_counts())

print("ENGINE LOCATION")
print(data["engine-location"].value_counts())

print("ENGINE TYPE")
print(data["engine-type"].value_counts())

print("NUMBER OF CYLINDERS")
print(data["num-of-cylinders"].value_counts())

print("FUEL SYSTEM")
print(data["fuel-system"].value_counts())


# Next step
# Count horsepower groups and the categories stored as numerical codes.
print("HORSEPOWER CATEGORY")
print(data["horsepower-binned"].value_counts())

print("SYMBOLING: CODED RISK CATEGORY")
print(data["symboling"].value_counts())

print("DIESEL INDICATOR")
print(data["diesel"].value_counts())

print("GAS INDICATOR")
print(data["gas"].value_counts())


# Next step
# Show how many cars belong to each body style.
data["body-style"].value_counts().plot(kind="bar", color="steelblue")
plt.title("Number of cars by body style")
plt.xlabel("Body style")
plt.ylabel("Number of cars")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# Next step
# Show how many cars belong to each drive-wheel group.
data["drive-wheels"].value_counts().plot(kind="bar", color="teal")
plt.title("Number of cars by drive wheels")
plt.xlabel("Drive wheels")
plt.ylabel("Number of cars")
plt.xticks(rotation=0)
plt.show()


# Next step
# A horizontal bar chart keeps the make names readable.
data["make"].value_counts().plot(kind="barh", figsize=(8, 7), color="steelblue")
plt.title("Number of cars by make")
plt.xlabel("Number of cars")
plt.ylabel("Make")
plt.tight_layout()
plt.show()


# Next step
# Put cars into drive-wheel groups and describe price within each group.
print(data.groupby("drive-wheels")["price"].describe().round(2))


# Next step
# Calculate and plot the average price for each drive-wheel group.
drive_means = data.groupby("drive-wheels")["price"].mean()
drive_means.plot(kind="bar", color="teal")
plt.title("Average price by drive wheels")
plt.xlabel("Drive wheels")
plt.ylabel("Average price in dataset units")
plt.xticks(rotation=0)
plt.show()


# Next step
# Describe price separately for each body style.
print(data.groupby("body-style")["price"].describe().round(2))


# Next step
# Calculate and plot the average price for each body style.
body_means = data.groupby("body-style")["price"].mean()
body_means.plot(kind="bar", color="steelblue")
plt.title("Average price by body style")
plt.xlabel("Body style")
plt.ylabel("Average price in dataset units")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# Next step
# Group by both drive wheels and body style at the same time.
joint_means = data.groupby(["drive-wheels", "body-style"])["price"].mean()

# Sort the averages from highest to lowest.
print(joint_means.sort_values(ascending=False).round(2))


# Next step
# Calculate the correlation between every pair of numerical columns.
correlation_table = data[numeric_columns].corr()

# Keep the price column from that correlation table.
price_correlations = correlation_table["price"]

# Remove price itself because a column always correlates perfectly with itself.
price_correlations = price_correlations.drop("price")

# Show the correlations from the most positive to the most negative.
print(price_correlations.sort_values(ascending=False).round(3))


# Next step
# Each dot compares engine size and price for one car.
data.plot(kind="scatter", x="engine-size", y="price", color="steelblue")
plt.title("Engine size and price")
plt.xlabel("Engine size on the supplied scale")
plt.ylabel("Price in dataset units")
plt.show()


# Next step
# Compare the weight of the car with its price.
data.plot(kind="scatter", x="curb-weight", y="price", color="teal")
plt.title("Curb weight and price")
plt.xlabel("Curb weight on the supplied scale")
plt.ylabel("Price in dataset units")
plt.show()


# Next step
# Compare engine power with price.
data.plot(kind="scatter", x="horsepower", y="price", color="steelblue")
plt.title("Horsepower and price")
plt.xlabel("Horsepower")
plt.ylabel("Price in dataset units")
plt.show()


# Next step
# Compare highway fuel economy with price.
data.plot(kind="scatter", x="highway-mpg", y="price", color="teal")
plt.title("Highway fuel economy and price")
plt.xlabel("Highway miles per gallon")
plt.ylabel("Price in dataset units")
plt.show()


# Next step
# A box plot compares the middle and spread of prices across groups.
data.boxplot(column="price", by="drive-wheels", figsize=(7, 5))
plt.title("Prices by drive wheels")
plt.suptitle("")
plt.xlabel("Drive wheels")
plt.ylabel("Price in dataset units")
plt.show()


# Next step
# Compare price distributions for the five body styles.
data.boxplot(column="price", by="body-style", figsize=(9, 5))
plt.title("Prices by body style")
plt.suptitle("")
plt.xlabel("Body style")
plt.ylabel("Price in dataset units")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


# Next step
# Compare average prices across several other categories.
print("AVERAGE PRICE BY MAKE")
print(data.groupby("make")["price"].mean().round(2))

print("AVERAGE PRICE BY ASPIRATION")
print(data.groupby("aspiration")["price"].mean().round(2))

print("AVERAGE PRICE BY HORSEPOWER CATEGORY")
print(data.groupby("horsepower-binned")["price"].mean().round(2))

print("AVERAGE PRICE BY NUMBER OF DOORS")
print(data.groupby("num-of-doors")["price"].mean().round(2))


# Next step
# Examine the remaining categorical attributes using the same method.
print("AVERAGE PRICE BY ENGINE LOCATION")
print(data.groupby("engine-location")["price"].mean().round(2))

print("AVERAGE PRICE BY ENGINE TYPE")
print(data.groupby("engine-type")["price"].mean().round(2))

print("AVERAGE PRICE BY CYLINDER CATEGORY")
print(data.groupby("num-of-cylinders")["price"].mean().round(2))

print("AVERAGE PRICE BY FUEL SYSTEM")
print(data.groupby("fuel-system")["price"].mean().round(2))

print("AVERAGE PRICE BY RISK CODE")
print(data.groupby("symboling")["price"].mean().round(2))

print("AVERAGE PRICE BY DIESEL INDICATOR")
print(data.groupby("diesel")["price"].mean().round(2))


# Next step
# A cross-tabulation counts each combination of two categories.
category_counts = pd.crosstab(data["body-style"], data["drive-wheels"])
print(category_counts)


# Next step
# Show the combination counts as a stacked bar chart.
category_counts.plot(kind="bar", stacked=True, figsize=(9, 5))
plt.title("Drive wheels within each body style")
plt.xlabel("Body style")
plt.ylabel("Number of cars")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


# Next step
# Remove the two numerical variables excluded from this screening step.
candidate_correlations = price_correlations.drop(["stroke", "city-L/100km"])

# abs changes negative values to positive values for strength comparisons.
correlation_strength = candidate_correlations.abs()

# Keep features whose absolute correlation is at least 0.70.
selected_features = correlation_strength[correlation_strength >= 0.70]

# Put the strongest selected features first.
selected_features = selected_features.sort_values(ascending=False)
print(selected_features.round(3))


# Next step
# Plot the strength of the five selected relationships.
selected_features.plot(kind="barh", color="teal", figsize=(8, 5))
plt.title("Features selected by the correlation filter")
plt.xlabel("Absolute correlation with price")
plt.ylabel("Feature")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
