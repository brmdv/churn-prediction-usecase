# Churn prediction for retail data

In this group project we do basic churn prediction for an unnamed company in the fashion industry, working together with [Delaware](https://www.delaware.pro/).

It is a **learning challenge**, held over **two weeks**. Our group consisted of [brmdv](https://github.com/brmdv), [medokhodeery](https://github.com/medokhodeery/medokhodeery), [danielmendoza4213 ](https://github.com/danielmendoza4213).

## The Mission

> An important retail company is interested in analyzing its client database to increase revenue. They are also focused on gathering new clients and increasing customer’s loyalty on a competitive fashion industry.
>
> The churn rate is increasing, the CEO urges the marketing team to start a marketing campaign for client retention.
>
> Your **mission**:
>
> - Predict those clients with more propensity to stop buying at the retail store.
> - Find possible groups of clients and define their characteristics. This will help the marketing team to design custom-made campaigns to attract new customers and increase customer retention.
> - Explore the impact on revenue and customer’s loyalty.
> - What should be the marketing strategy leading to higher revenue and higher customer retention?
> - Build a dashboard with data insights and KPIs.

## Files

- _clean_data_: Initial data cleaning, like removing invalid rows, unnecessary columns,…
- _supervised_churn.py_: Collects and merges the data, based on the customers. It also calculates some new features.

### Data sources

The data was anonimously scrambled provided to us by Delaware. For obvious privacy reasons, it is not included in this repository. The data was very messy and needed a lot of clean-up.

### Data cleaning

Assumptions :

- We considered only the transactions with a positive "Quantity" and with a defined "Product ID". This means that we do not consider transactions where the products were returned and transactions made by unregistered customers

#### Churn labeling

- The labeling of churn was based on 3 paramenters:
  - Median time between purchases
  - How far behind are they at the end of the dataset?
  - Set some tolerance with an arbitrary factor
    - 1.5 × median days between purchases < days since last ⇒ churned
- Plus an exception for recent (new) customers

## Usage

1. Run clean_data.py with the 3 oroginal databases
2. Run supervised_churn.py to create a dataframe for:
   2.1 Churn Predictions
   2.2 Clustering - > Run Kmeans.py

3. Outputs:
   3.1 Dataframe with features per CustomerId plus Churned or Not_Churned
   3.2 Dataframe with features per CustomerId plus Cluster group

## Results

Prensentation : Ressources\Retail.pdf
