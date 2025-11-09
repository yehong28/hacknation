import pandas as pd
df = pd.read_csv('./archive/Resume/Resume.csv')
# Basic inspection
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print("Unique Categories:", df['Category'].nunique())
print("Category Distribution:\n", df['Category'].value_counts())