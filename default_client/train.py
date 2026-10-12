from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("german_credit_data.csv")
# print(df.info())
#print(df.describe(include='all'))
#print(df.head())
# print(df['Saving accounts'].unique())
# print(df['Checking account'].unique())
# print(df['Purpose'].nunique())

from cust_preprocessing import preprocessing, CustomEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

X = df.drop(columns=['Risk'])
y = df['Risk'].map({'good':0, 'bad':1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

prep = FunctionTransformer(preprocessing)

cod = ColumnTransformer(
    transformers= [('one', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Housing'])],
    remainder= 'passthrough',
    verbose_feature_names_out= False,
).set_output(transform='pandas')

pipka = Pipeline([
    ('clean', prep),
    ('cod', cod),
    ('enc', CustomEncoder(columns=['Purpose'])),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', random_state=42))
])
pipka.fit(X_train, y_train)

import joblib
joblib.dump(pipka, 'prod_scoring.pkl')

from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
true_prob = pipka.predict_proba(X_test)[:, 1]
treshold = 0.3
y_pred = (true_prob >= treshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
print(f'Правильно предсказала хорошего: {tn} | Предсказала плохого, а он хороший: {fp}')
print(f'Предсказала хорошего, а он плохой: {fn} | Правильно предсказала плохого: {tp}')

score1 = recall_score(y_test, y_pred)
score2 = roc_auc_score(y_test, true_prob)
print(f'recall: {score1}, roc-auc: {score2}')


# print(X_test.iloc[[0]])
# print(true_prob[0])

# data_tr = pipka[:-1].transform(X_train)
# df_tr = pd.DataFrame(data_tr)
# print(df_tr.describe(include="all"))

# import matplotlib.pyplot as plt
# import seaborn as sns
# cor = train_enc.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(cor, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1,vmax=1)
# plt.show()

# import shap
# explainer = shap.LinearExplainer(pipka, X_train)
# sh_values = explainer(X_test)
# print(sh_values)
# plt.figure(figsize=(10, 6))
# shap.plots.beeswarm(sh_values)
#shap.plots.waterfall(sh_values[13])