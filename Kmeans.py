import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

##
# Imput : csv file from supervised_churn.py
df = pd.read_csv("DATA/customer_churn_for_clustering.csv")
# Selection of relevant features for the clustering
customer_metrics = df[
    [
        "total_price",
        "price_per_transaction",
        "unique_transactions",
        "number_discounts",
        "avg_days_between",
        "ProductPrice_yearly_mean",
        "Quantity_yearly_mean",
        "distance",
        "age",
        "is_male",
        "customer_rank",
        "churned",
        "most_purchased_season",
        "most_purchased_product_type",
        "most_Brand_premium",
        "most_Brand_internal",
    ]
]
# Reset Index so we dont get back NaN clusters

# Kmeans /
def run_kmeans_rank(m, n):
    kmeans = KMeans(n_clusters=3, init="k-means++").fit(n)
    cluster_labels = pd.Series(kmeans.labels_, name="cluster")
    n = m.join(cluster_labels.to_frame())
    return n


#
# Per RANK
rank_one = customer_metrics.loc[customer_metrics["customer_rank"] == "top10"]
rank_two = customer_metrics.loc[customer_metrics["customer_rank"] == "top30"]
rank_three = customer_metrics.loc[customer_metrics["customer_rank"] == "top60"]
rank_one.reset_index(inplace=True, drop=True)
rank_two.reset_index(inplace=True, drop=True)
rank_three.reset_index(inplace=True, drop=True)
# Dummies for categorical data
rank_one_dummies = pd.get_dummies(rank_one)
rank_two_dummies = pd.get_dummies(rank_two)
rank_three_dummies = pd.get_dummies(rank_three)
# drop unnecesary column
rank_one_dummies.drop(["customer_rank_top10"], axis=1, inplace=True)
rank_two_dummies.drop(["customer_rank_top30"], axis=1, inplace=True)
rank_three_dummies.drop(["customer_rank_top60"], axis=1, inplace=True)
# RANK1
print("- Clustering TOP10 Customers")
scaler = StandardScaler()
scaler.fit(rank_one_dummies)
rank_one_dummies_scale = scaler.transform(rank_one_dummies)
rank_one_dummies_scale_df = pd.DataFrame(
    rank_one_dummies_scale, columns=rank_one_dummies.columns
)
rank_one_clusters = run_kmeans_rank(rank_one, rank_one_dummies_scale_df)
# RANK2
print("- Clustering TOP30 Customers")
scaler = StandardScaler()
scaler.fit(rank_one_dummies)
rank_two_dummies_scale = scaler.transform(rank_two_dummies)
rank_two_dummies_scale_df = pd.DataFrame(
    rank_two_dummies_scale, columns=rank_one_dummies.columns
)
rank_two_clusters = run_kmeans_rank(rank_two, rank_two_dummies_scale_df)
# RANK3
print("- Clustering TOP60 Customers")
scaler = StandardScaler()
scaler.fit(rank_one_dummies)
rank_three_dummies_scale = scaler.transform(rank_three_dummies)
rank_three_dummies_scale_df = pd.DataFrame(
    rank_three_dummies_scale, columns=rank_three_dummies.columns
)
rank_three_clusters = run_kmeans_rank(rank_three, rank_three_dummies_scale_df)
# Put cluster per rank together
rank_one_clusters["cluster"].replace(
    {0: "Cluster A", 1: "Cluster B", 2: "Cluster C"}, inplace=True
)
rank_two_clusters["cluster"].replace(
    {0: "Cluster D", 1: "Cluster E", 2: "Cluster F"}, inplace=True
)
rank_three_clusters["cluster"].replace(
    {0: "Cluster G", 1: "Cluster H", 2: "Cluster I"}, inplace=True
)
clusters_per_rank = (
    rank_one_clusters.append(rank_two_clusters, ignore_index=True)
).append(rank_three_clusters, ignore_index=True)

clusters_per_rank = clusters_per_rank.join(df[["CustomerID"]])
clusters_per_rank.set_index("CustomerID", inplace=True)
# Save
clusters_per_rank.to_csv("DATA/clusters_per_rank.csv")
print("Done")
