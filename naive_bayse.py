from operator import index
import pandas as pd
import numpy as np

df = pd.read_csv('adult.csv', index_col=False, header=None)
df = df[:100]

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.columns = col_names

targer_set = 'income'
feature_set = ['age', 'workclass', 'education', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'hours_per_week', 'native_country']
targetcount = df[targer_set].value_counts().tolist()


feature_list = {}
likelyhood = []
feature_count = []

for i in feature_set:
    # make a pivot table with feature and target with columns as the target and rows as unique values from feature set
    table = pd.crosstab(df[i], df[targer_set])
    likelyhood.append(table.to_numpy())
    feature_list[i] = table.index.to_list()

likelyhood_prob = []
for i in likelyhood:
    likelyhood_prob.append(np.divide(i, targetcount))

print(likelyhood_prob)

valuesss = df[targer_set].unique()


def naive_bayes(**kwargs):
    prob_yes = 1
    prob_no = 1
    j = 0
    p_yes = targetcount[0]/sum(targetcount)
    p_no = targetcount[1]/sum(targetcount)
    for i, value in kwargs.items():
        if feature_set[j] == i:
            prob_yes *= likelyhood_prob[feature_set.index(
                i)][feature_list[i].index(value)][0]
            prob_no *= likelyhood_prob[feature_set.index(
                i)][feature_list[i].index(value)][1]
            j += 1
        elif j < len(feature_set):
            j += 1
        else:
            j = 0

    prob_yes *= p_yes
    prob_no *= p_no

    if prob_yes > prob_no:
        return valuesss[0]
    else:
        return valuesss[1]


print(naive_bayes(age=34, workclass=' Local-gov', education=' Bachelors',
                  marital_status=' Married-civ-spouse', occupation=' Protective-serv', relationship=' Husband'))

print(naive_bayes(age=39, workclass=' State-gov', education=' Bachelors', marital_status=' Never-married', occupation=' Adm-clerical',
                  relationship=' Not-in-family', race=' White', sex=' Male', hours_per_week=40, native_country=' United-States'))
