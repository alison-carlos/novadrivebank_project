#%%

import pandas as pd
from datetime import datetime
import numpy as np
import random as python_random
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import tensorflow as tf

from utils import *
import const

#%%

# Reproducibility
seed = 41
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

#%%
# Raw data
df = fetch_data_from_db(const.sql_query)

#%%
# Data convertion
df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)

#%%
# Null treatment
fn_replace_nulls(df)

#%%
# Fix typing errors

valid_professions = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador', 'Dentista', 'Engenheiro', 'Médico', 'Programador']
fn_fix_typing_errors(df, 'profissao', valid_professions)

#%%
# Treat outliers
df = fn_treat_outliers(df, 'tempoprofissao', 0, 70)
df = fn_treat_outliers(df, 'idade', 0, 110)

#%%
# Feature Engineering
df['proporcaosolicitadototal'] = df['valorsolicitado'] / df['valortotalbem']
df['proporcaosolicitadototal'] = df['proporcaosolicitadototal'].astype(float)

#%%
# Split data in test and train
X = df.drop('classe', axis=1)
y = df['classe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#%%
# Apply StandardScaler
X_test = fn_save_scalers(X_test, ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem', 'proporcaosolicitadototal'])
X_train = fn_save_scalers(X_train, ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem', 'proporcaosolicitadototal']) 

#%%
# Codification

mapping = {'ruim' : 0, 'bom' : 1}
y_train = np.array([mapping[item] for item in y_train])
y_test = np.array([mapping[item] for item in y_test])

X_train = fn_save_encoders(X_train, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])
X_test = fn_save_encoders(X_test, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])

#%%
# Attribute Selection
model = RandomForestClassifier()

# RFE instance
selector = RFE(model, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# Transform data
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

joblib.dump(selector, './objects/selector.joblib')

#%% 
# Creating the RNA (Rede Neural Artificial)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#%%
# Setting the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#%%
# Compiling the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# %%
# Training the model
model.fit(
    X_train,
    y_train,
    validation_split=0.2,# Uses 20% of the data to validation
    epochs=500, # Max number of epochs
    batch_size=10,
    verbose=1
)
# %%

model.save('my_model.keras')
# %%
# Prediction

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

#%%
# Evaluating the model
print(f'Avaliação do Modelo nos Dados de Teste:')
model.evaluate(X_test, y_test)

# %%
print(f'\nRelatório de Classificação:')
print(classification_report(y_test, y_pred))
# %%
