import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('store_data.csv', header=None)

records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i, j]) for j in range(0, 20)])

association_rules = apriori(
    records, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)
print(association_results)


for r in association_results:
    pair = r[0]
    if len(pair) > 1:
        print("rule: ", " -> ".join([x for x in pair]))
        print("Support = ", r[1])
        print("Confidence = ", r.ordered_statistics[0].confidence)
        print("Lift = ", r.ordered_statistics[0].lift)
        print("============================================")
