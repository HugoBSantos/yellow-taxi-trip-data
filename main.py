import streamlit as st
import polars as pl
import plotly.express as px
from pathlib import Path

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
    
    st.title("2. Análise Exploratória (Insights)")
    
    df = read_gold_data()
    st.dataframe(df.head())


def page_3():
    
    st.title("3. Modelagem Estatística (Regressão Linear)")
    
    df = read_gold_data()
    st.dataframe(df.head())


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
    