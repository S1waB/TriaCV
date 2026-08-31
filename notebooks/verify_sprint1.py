import pandas as pd
df = pd.read_csv('C:/Users/monta/Videos/TraiCV/data/processed/clean_resumes.csv')
print('Lignes:', len(df))
print('Categories:', df['Category'].nunique())
print('Nulls:', int(df.isnull().sum().sum()))
print(df[['Category','Clean_Word_Count']].head(3).to_string())
print(df['Category'].value_counts().to_string())
