from sklearn.preprocessing import MaxAbsScaler
import pandas as pd

X = pd.DataFrame([[8, -2, 3], [2, 25, 0], [4, 0, -2]])
scaler_maxabs = MaxAbsScaler()
df_maxabs = scaler_maxabs.fit_transform(X)
# df_maxabs = pd.DataFrame(df_maxabs, columns=X.columns)

print(df_maxabs)
