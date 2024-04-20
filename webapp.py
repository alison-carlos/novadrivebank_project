import streamlit as st
import requests
import yaml

st.title('Avaliação de crédito')

with open('config.yml', 'r') as open_file:
    config = yaml.safe_load(open_file)
    url = config['url_api']['url']

professions = ['Advogado', 'Arquiteto', 'Cientista de Dados', 
              'Contador', 'Dentista', 'Empresário', 'Engenheiro', 'Médico', 'Programador']
residence_types = ['Alugada', 'Outros', 'Própria']
education_levels = ['Ens.Fundamental', 'Ens.Médio', 'PósouMais', 'Superior']
scores = ['Baixo', 'Bom', 'Justo', 'MuitoBom']
marital_status = ['Casado', 'Divorciado', 'Solteiro', 'Víuvo']
products = ['AgileXplorer', 'DoubleDuty', 'EcoPrestige', 'ElegantCruise', 
            'SpeedFury', 'TrailConqueror', 'VoyageRoarmer', 'WorkMaster']


with st.form(key='prediction_form'):
    profession = st.selectbox('Profissão', professions)
    profession_time = st.number_input('Tempo de profissão (em anos)', min_value=0, max_value=80, step=1)
    monthly_income = st.number_input('Renda mensal', min_value=0.0, value=0.0, step=1000.0)
    residence_type = st.selectbox('Tipo de residência', residence_types)
    education_level = st.selectbox('Escolaridade', education_levels)
    score = st.selectbox('Score', scores)
    age = st.number_input('Idade', min_value=18, max_value=80, value=18, step=1)
    dependents = st.number_input('Dependente(s)', min_value=0, value=0, step=1)
    marital_status = st.selectbox('Estado Civil', marital_status)
    product = st.selectbox('Produto', products)
    requested_value = st.number_input('Valor solicitado', min_value=0.0, value=0.0, step=10000.0)
    product_value = st.number_input('Valor total do bem', min_value=0.0, value=0.0, step=10000.0)

    submit_button = st.form_submit_button(label='Enviar')

if submit_button:
    data = {
        "profissao": [profession],
        "tempoprofissao": [profession_time],
        "renda": [monthly_income],
        "tiporesidencia": [residence_type],
        "escolaridade": [education_level],
        "score": [score],
        "idade": [age],
        "dependentes": [dependents],
        "estadocivil": [marital_status],
        "produto": [product],
        "valorsolicitado": [requested_value],
        "valortotalbem": [product_value],
        "proporcaosolicitadototal": [requested_value / product_value]
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        prediction = response.json()
        probability = prediction[0][0] * 100
        class_ = 'Bom' if probability > 50 else 'Ruim'
        st.success(f'Probabilidade: {probability:.2f}%')
        st.success(f'Classe: {class_}')
    else:
        st.error(f'Erro ao realizar previsão. \n {response.status_code}')