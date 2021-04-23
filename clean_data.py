# %% Libraries
import pandas as pd

## TRANSACTIONS

# %% Read CSV into dataframe
transactions = pd.read_csv(
    "DATA/BECODE TRANSACTIONS.csv",
    parse_dates=["TransactionDate"],
    index_col=["TransactionID"],
    low_memory=False,
)

# %% Fill nan discounts with 0
transactions["Discount"].fillna(0.0, inplace=True)

# %% Drop negative discounts
transactions = transactions[transactions["Discount"] >= 0]

# %% Negative values for quantity
transactions.loc[
    (transactions["Quantity"] < 0) & (transactions["Returned"] == 0), "Returned"
] = 1

# %% NAN values for customers: keep in separate dataframe to be sure and drop in main df
non_registered_customers = transactions[transactions["CustomerID"].isna()].drop(
    labels=["CustomerID"], axis=1
)
transactions.dropna(axis=0, subset=["CustomerID"], inplace=True)
transactions["CustomerID"] = transactions["CustomerID"].astype(int)

# %% Drop negative price and discounts
transactions = transactions[
    (transactions["ProductPrice"] >= 0) & (transactions["Discount"] >= 0)
]

# %%
transactions["ProductID"].replace(-1, 0, inplace=True)

# %% Save csv files
transactions.to_csv("DATA/transactions_cleaned.csv")
non_registered_customers.to_csv("DATA/transactions_no_custid_cleaned.csv")

## PRODUCTS
# %% read read_csv
product = pd.read_csv("./DATA/BECODE PRODUCTS.csv", sep=";", index_col="ProductID")

# %% Change brand to separate columns
product["Brand_premium"] = product["Brand"].apply(
    lambda brand: 1 if "Premium" in brand else 0
)
product["Brand_internal"] = product["Brand"].apply(
    lambda brand: 1 if "Inter" in brand else 0
)
product.drop(["Brand"], axis=1, inplace=True)

# %% fill unknown values
product.fillna("Unknown", inplace=True)

# %% get product types column
def product_categories(x):
    if x == "BABIES":
        return "babies"
    elif x == "KIDS":
        return "kids"
    elif x == "WOMEN":
        return "women"
    else:
        return "unknown"


product["ProductCategory"] = (product["ProductName"].str.split(" ").str[-1]).apply(
    product_categories
)
product.drop(["ProductName"], axis=1, inplace=True)
# %% save csv
product.to_csv("DATA/products_cleaned.csv")


## CUSTOMERS
# %% load csv
customers = pd.read_csv(
    "DATA/BECODE CUSTOMERS.csv",
    sep=";",
    index_col="CustomerID",
    parse_dates=["CR_DATE"],
)

# %% Change column names to English
customers.columns = ["municipality", "distance", "creation_date", "age", "gender"]

# %% Gender
customers["is_male"] = (customers["gender"] == "M").astype(int)
customers.drop("gender", axis=1, inplace=True)

# %% save to csv
customers.to_csv("DATA/customers_cleaned.csv")

# %%
