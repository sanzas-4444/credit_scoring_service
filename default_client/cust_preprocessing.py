import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df_cl = df.copy()
    if 'Unnamed: 0' in df_cl.columns:
        df_cl = df_cl.drop('Unnamed: 0', axis=1)
    df_cl = df_cl.fillna('none')
    df_cl['Sex'] = df_cl['Sex'].map({'male':1, 'female':0})
    cat1, cat2 = ['none', 'little', 'moderate', 'quite rich', 'rich'], ['none', 'little', 'moderate', 'rich']
    ord = OrdinalEncoder(
        categories=[cat1, cat2], handle_unknown='error'
        )
    df_cl[['Saving accounts', 'Checking account']] = ord.fit_transform(
        df_cl[['Saving accounts', 'Checking account']]
        )
    df_cl['credit_month'] = df_cl['Credit amount']/df_cl['Duration']
    return df_cl

# функция для кодирвоания Purpose (заменяем на частоту) 
class CustomEncoder:
    def __init__(self, columns):
        self.columns = columns
        self.maps = {}
    def fit(self, X, y=None):
        for col in self.columns:
            self.maps[col] = X[col].value_counts(normalize=True)
        return self
    def transform(self, X):
        X_enc = X.copy()
        for col in self.columns:
            X_enc[col + '_enc'] = X_enc[col].map(self.maps[col]).fillna(0)
            X_enc = X_enc.drop(columns=[col])
        return X_enc