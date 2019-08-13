#!/usr/bin/env python
# -*- coding: utf-8 -*-


import featuretools as ft



# ===============================================================================
# Representing data with EntitySets
# 	- An EntitySet is a collection of entities and the relationships between them
# ===============================================================================
data = ft.demo.load_mock_customer()

# transactions dataframe
transactions_df = data["transactions"]\
	.merge(data["sessions"])\
	.merge(data["customers"])
transactions_df.sample(10)

# products dataframe
products_df = data["products"]
print(products_df)


# create an EntitySet
es = ft.EntitySet(id = "customer_data")
es = es.entity_from_dataframe(entity_id = "transactions",
							  dataframe = transactions_df,
							  index = "transaction_df",
							  time_index = "transaction_time",
							  variable_types = {
								  "product_id": ft.variable_types.Categorical,
								  "zip_code": ft.variable_types.ZIPCode
							  })