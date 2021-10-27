import pandas as pd


consonant_path = 'data/consonant.csv'
vowel_path = 'data/vowel.csv' 
dfs = pd.read_csv(consonant_path, sep='\t', header=None)
dfs1 = pd.read_csv(vowel_path, sep='\t', header=None)
print(dfs.head())

print(dfs1.head())