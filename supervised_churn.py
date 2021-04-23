# %%
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

## convert timedeltas to float (days)
def to_days(duration):
    return duration.days + duration.seconds / 86400.0


# %% merge datasets
print("· Reading transations…")
transactions = pd.read_csv(
    "DATA/transactions_cleaned.csv", parse_dates=["TransactionDate"],
)
# %%  filtering out B2B customer
print("· Filterings out some transactions")
transactions = transactions.loc[
    transactions["Quantity"] > 0
]  # only positives purchases
transactions_filter = transactions[
    ["CustomerID", "ProductPrice", "Quantity", "TransactionDate"]
].copy()
transactions_filter["TransactionDate"] = pd.to_datetime(
    transactions_filter["TransactionDate"]
).dt.year
yearly = transactions_filter.groupby(["CustomerID", "TransactionDate"])[
    ["ProductPrice", "Quantity"]
].agg({"ProductPrice": "sum", "Quantity": "sum"})
yearly = yearly.groupby("CustomerID").agg("mean")
btoc_customers = (
    yearly.loc[(yearly["ProductPrice"] < 6000) | (yearly["Quantity"] < 200)]
).index
transactions = transactions.loc[transactions["CustomerID"].isin(btoc_customers)]


# %% Gruoping together the transactions by customer
print("· Customer grouping and aggregations")
transactions_per_customer = transactions.groupby("CustomerID")[
    ["TransactionDate", "ProductPrice", "TransactionID", "Discount"]
].agg(
    {
        "ProductPrice": ["sum", "mean", "count"],
        "TransactionDate": [
            "min",
            "max",
            "nunique",
            lambda dates: max(dates) - min(dates) + timedelta(days=1),
            lambda dates: np.median(np.diff(np.sort(np.array(list(dates)))))
            if len(list(dates)) > 1
            else timedelta(days=0),
            lambda dates: np.max(np.diff(np.sort(np.array(list(dates)))))
            if len(list(dates)) > 1
            else timedelta(days=0),
            lambda dates: np.min(np.diff(np.sort(np.array(list(dates)))))
            if len(list(dates)) > 1
            else timedelta(days=0),
        ],
        "TransactionID": "nunique",
        "Discount": ["sum", lambda x: int(x.astype(bool).sum())],
    }
)

transactions_per_customer.columns = [
    "total_price",
    "price_per_transaction",
    "number_transactions",
    "first_day",
    "last_day",
    "number_of_days",
    "period",
    "median_days_between",
    "max_days_between",
    "min_days_between",
    "unique_transactions",
    "total_discount",
    "number_discounts",
]

# %% Some date calculations
tqdm.pandas(desc="· Some date calculations")
transactions_per_customer["avg_days_between"] = transactions_per_customer[
    "period"
] / transactions_per_customer["number_of_days"].progress_apply(
    lambda x: max(x - 1, 1)
) - timedelta(
    days=1
)
transactions_per_customer["since_last"] = (
    transactions["TransactionDate"].max() - transactions_per_customer["last_day"]
)
transactions_per_customer["since_first"] = (
    transactions["TransactionDate"].max() - transactions_per_customer["first_day"]
)
transactions_per_customer["transactions_over_time"] = transactions_per_customer[
    "number_transactions"
] / transactions_per_customer["period"].progress_apply(to_days)
transactions_per_customer["revenue_over_time"] = transactions_per_customer[
    "total_price"
] / transactions_per_customer["period"].progress_apply(to_days)
#%% join with yearly mean information
yearly.columns = ["ProductPrice_yearly_mean", "Quantity_yearly_mean"]
transactions_per_customer = transactions_per_customer.join(yearly)

# %%
print("· Reading customer data and merge.")
customers = pd.read_csv(
    "DATA/customers_cleaned.csv", index_col="CustomerID", parse_dates=["creation_date"]
)
customers.drop("municipality", axis=1, inplace=True)
transactions_per_customer = transactions_per_customer.join(customers)


# %% rank customers in 10%, 30% and 60% largest
## calculate subset sizes
print("· Ranking the customers")
N = transactions_per_customer.shape[0]
top10_n = round(N * 0.10)
top30_n = round(N * 0.30)
top60_n = round(N * 0.60)
## sort customers by total rev
transactions_per_customer.sort_values(by="total_price", ascending=False, inplace=True)
## create new empty col
transactions_per_customer["customer_rank"] = 0
## set rank 1,2,3 ate each col
transactions_per_customer.iloc[:top10_n] = transactions_per_customer.iloc[
    :top10_n
].assign(customer_rank="top10")
transactions_per_customer.iloc[top10_n:top30_n] = transactions_per_customer.iloc[
    top10_n:top30_n
].assign(customer_rank="top30")
transactions_per_customer.iloc[top30_n:] = transactions_per_customer.iloc[
    top30_n:
].assign(customer_rank="top60")


# %% label customer churn
## based on usual buying rate
tqdm.pandas(desc="· Calculating churn…")
transactions_per_customer["churned"] = transactions_per_customer.progress_apply(
    lambda row: "churned"
    if 1.5 * row["avg_days_between"] < row["since_last"]
    else "not_churned",
    axis=1,
)
## make exception for recent (new?) customers
## → they haven't had the chance to churn yet
transactions_per_customer.loc[
    (transactions_per_customer["since_last"] < timedelta(days=60))
    & (transactions_per_customer["number_of_days"] == 1),
    "churned",
] = "not_churned"

# %% Convert every column to numeric values and normalize if necessary
## remove index
print("· Cleaning the final results")
transactions_per_customer.reset_index(inplace=True)
## drop unneccesary cols
transactions_per_customer.drop(
    ["first_day", "last_day", "period", "creation_date"], axis=1, inplace=True
)

transactions_per_customer.set_index("CustomerID", inplace=True)
# %% Product based features
print("- Maybe some more features...")
products = pd.read_csv("DATA/products_cleaned.csv", index_col="ProductID")
# %% Season information
summer_list = ["S10", "S11", "S12", "S13", "S14", "S15", "S16"]
winter_list = ["W10", "W11", "W12", "W13", "W14", "W15", "S16"]
basics_list = ["BASICS"]
unknown_list = ["unknown"]
# 1=summer / 2 = winter / 3= basic / 4 = unknown
products["Season"] = products["Collection"].apply(
    lambda x: "summer"
    if x in summer_list
    else (
        "winter"
        if x in winter_list
        else ("basics" if x in basics_list else ("unknown" if x in unknown_list else 0))
    )
)
# add productId =0 for unknown product transactions
product_unknown = pd.Series(
    data={
        "Collection": "unknown",
        "Brand_premium": "unknown",
        "Brand_internal": "unknown",
        "ProductCategory": "unknown",
        "Season": "unknown",
    },
    name=0,
)
products = products.append(product_unknown, ignore_index=False)
# %% get features per customer
transactions_for_products = transactions.copy()
transactions_products = transactions_for_products.join(
    products, on="ProductID", how="left"
)
transactions_products_perCustomer = (
    transactions_products.groupby(
        [
            "CustomerID",
            "Season",
            "ProductCategory",
            "Brand_premium",
            "Brand_internal",
            "Collection",
        ]
    )["Quantity"]
    .agg("sum")
    .to_frame(name="count")
)
# %% get features max values
print("· Still running")
max_product_perCustomer = transactions_products_perCustomer.loc[
    transactions_products_perCustomer.groupby("CustomerID")["count"].idxmax()
]
max_product_perCustomer.reset_index(inplace=True)
max_product_perCustomer.set_index("CustomerID", inplace=True)
# %%
max_product_perCustomer.columns = [
    "most_purchased_season",
    "most_purchased_product_type",
    "most_Brand_premium",
    "most_Brand_internal",
    "most_purchased_colection",
    "count_idxmax",
]


# %%
# transactions_per_customer["period"] = transactions_per_customer["period"].apply(
#     to_days,
# )
transactions_per_customer[
    [
        "avg_days_between",
        "median_days_between",
        "max_days_between",
        "min_days_between",
        "since_first",
    ]
] = transactions_per_customer[
    [
        "avg_days_between",
        "median_days_between",
        "max_days_between",
        "min_days_between",
        "since_first",
    ]
].applymap(
    to_days,
)

# %%  add product based feautures
transactions_per_customer = transactions_per_customer.join(max_product_perCustomer)

# %% save new dataset in file
print("· Saving to file")
transactions_per_customer.to_csv("DATA/customer_churn_for_clustering.csv")

# %%
