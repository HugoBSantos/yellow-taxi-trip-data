import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import numpy as np

st.set_page_config(
    page_title="NYC Taxi Analytics",
    page_icon="🚕",
    layout="wide"
)

@st.cache_data
def read_gold_data():
    
    gold_path = Path.cwd() / "data" / "gold" / "yellow_tripdata_2015-01.parquet"
    lf = pl.scan_parquet(gold_path)
    
    lf = lf.select([
        "tpep_pickup_datetime", "trip_distance", "total_amount", 
        "hour_of_day", "day_of_week", "vendor_id", "payment_type", "extra"
    ])
    
    return lf.collect().sample(500000)


def page_1():
    
    st.header("🧹 1. Contexto e Tratamento de Dados")
    df = read_gold_data()
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Amostra Analisada", f"{len(df):,}")
        col2.metric("Período", "Janeiro 2015")
        col3.metric("Cidade", "New York (Manhattan)")

        st.subheader("Tratamento Realizado (Camada Silver -> Gold)")
        st.markdown("""
        - **Remoção de Outliers:** Filtramos distâncias > 10 milhas e valores de tarifa inconsistentes.
        - **Geofencing:** Coordenadas limitadas ao perímetro de NYC.
        - **Tratamento de Nulos:** Colunas críticas com valores ausentes foram removidas.
        - **Tipagem:** Conversão de strings para datetime e mapeamento de IDs para nomes reais.
        """)
        st.write("Amostra dos dados limpos:")
        st.dataframe(df.head(10), use_container_width=True)


def page_2():
    
    st.header("🔍 2. Análise Exploratória (EDA)")
    df = read_gold_data()
    
    tab1, tab2, tab3 = st.tabs(["Distribuições", "Sazonalidade", "Correlações"])
    
    with tab1:
        st.subheader("Distribuição do Valor Total")
        fig = px.histogram(df, x="total_amount", nbins=50, color_discrete_sequence=['#FFD700'])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Volume de Corridas por Hora")
        vendas_hora = df.group_by("hour_of_day").count().sort("hour_of_day")
        fig = px.line(vendas_hora, x="hour_of_day", y="count", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Matriz de Correlação")
        # Apenas colunas numéricas para a correlação
        num_cols = df.select([pl.col(pl.NUMERIC_DTYPES)])
        corr_matrix = num_cols.corr().to_pandas()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)


def page_3():
    
    st.header("🤖 3. Modelagem de Regressão Linear")
    df = read_gold_data()
    
    st.write("Objetivo: Prever a **Taxa Extra (Rush Hour)** com base na **Hora do Dia**.")

    # Preparação rápida do modelo
    X = df.select("hour_of_day").to_numpy()
    y = df.select("extra").to_numpy()
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Métricas
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    c1, c2 = st.columns(2)
    c1.metric("R² (Precisão)", f"{r2:.4f}")
    c2.metric("Erro Médio (MAE)", f"${mae:.2f}")

    # Simulador interativo
    st.divider()
    st.subheader("🔮 Simulador de Previsão")
    input_hora = st.slider("Selecione a Hora do Dia", 0, 23, 17)
    previsao = model.predict([[input_hora]])[0][0]
    
    st.success(f"Para às **{input_hora}h**, a taxa extra estimada é de **${max(0, previsao):.2f}**")


def page_4():
    
    st.title("4. Testes de Hipóteses (Validação)")
    
    df = read_gold_data()
    st.dataframe(df.head())


def page_5():
    
    st.title("5. Propostas de Soluções")
    
    df = read_gold_data()
    st.dataframe(df.head())


if __name__ == "__main__":
    
    pages = {
        "Etapa 1: Dados": [
            st.Page(page_1, title="Limpeza e Preparação", icon="🧹"),
            st.Page(page_2, title="Exploração (EDA)", icon="🔍")
        ],
        "Etapa 2: Estatística": [
            st.Page(page_3, title="Modelagem Estatística", icon="🤖"),
            st.Page(page_4, title="Testes de Hipóteses", icon="🧪"),
        ],
        "Conclusão": [
            st.Page(page_5, title="Propostas de Soluções", icon="💡")
        ]
    }
    
    pg = st.navigation(pages)
    pg.run()
    