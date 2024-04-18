#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import const
from utils import *

# Get data from Postgres
df = fetch_data_from_db(const.sql_query)

#%%

# Apply data convertion
df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)

#%%

# Defines variables
categorical_variables = ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto']
numerical_variables = ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem']

#%%
for column in categorical_variables:
    df[column].value_counts().plot(kind='bar', figsize=(10,6))
    plt.title(f'Distribuição de {column}')
    plt.ylabel('Contagem')
    plt.xlabel(column)
    plt.xticks(rotation=45)
    plt.show()
#%%
for column in numerical_variables:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x=column)
    plt.title(f'Boxplot de {column}')
    plt.show()

    df[column].hist(bins=20, figsize=(10,6))
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.show()

    print(f'Resumo estatístico de {column} \n' , df[column].describe(), '\n')

#%% 

nulls_per_column = df.isnull().sum()
print(nulls_per_column)

# %%
