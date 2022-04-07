#carregando as bibliotecas
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model
from xgboost import XGBClassifier

var_model = "model"
var_dataset = "health.csv"

#carregando o modelo treinado.
model = load_model(var_model)

#carregando o conjunto de dados.
dataset = pd.read_csv(var_dataset)

# título
st.title("Health Insurance Cross Sell")

# subtítulo
st.markdown("Predição de probabilidade de adesão ao seguro de automóvel")

# imprime o conjunto de dados usado
st.dataframe(dataset.drop("turnover",axis=1).head())

st.sidebar.subheader("Defina os atributos do empregado para predição de probabilidade de adesão ao seguro de automóvel")

# mapeando dados do usuário para cada atributo
DanoAnterior = st.sidebar.number_input("Damage", value=dataset["Damage"].mean())
IdadeAutomovel= st.sidebar.number_input("V_age", value=dataset["V_age"].mean())
Regiao = st.sidebar.number_input("Region", value=dataset["Region"].mean())
CanalVendas = st.sidebar.number_input("Channel", value=dataset["Channel"].mean())
Habilitado = st.sidebar.number_input("License", value=dataset["License"].mean())
TempoAssociado = st.sidebar.number_input("Vinatge", value=dataset["Vintage"].mean())
PremioAnual = st.sidebar.number_input("Rate", value=dataset["Rate"].mean())
Feminino = st.sidebar.number_input("Gender_Female", value=dataset["Gender_Female"].mean())
Masculino = st.sidebar.number_input("Gender_Male", value=dataset["Gender_Male"].mean())

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["Damage"] = [Damage]
    data_teste["V_age"] = [V_age]    
    data_teste["Region"] = [Region]
    data_teste["Channel"] = [Channel]
    data_teste["License"] = [License]
    data_teste["Vintage"] = [Vintage]
    data_teste["Rate"] = [Rate]
    data_teste["Gender_Female"] = [Gender_Female]
    data_teste["Gender_Male"] = [Gender_Male]
    
    #imprime os dados de teste.
    st.subheader("Predição do modelo:")

    #realiza a predição.
    result = predict_model(model, data=data_teste)
    
    #recupera os resultados.
    classe = result["Label"][0]
    prob = result["Score"][0]*100
    
    if classe==1:
        st.write("A probabilidade de adesão ao seguro de automóvel é de: {0:.2f}%".format(prob))
    
