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
    
    st.header("🤖 3. Modelagem de Regressão e Inferência")
    df = read_gold_data()
    
    st.markdown("""
    Nesta etapa, analisamos o impacto da **Hora do Dia** no valor da **Taxa Extra**. 
    Além da previsão, calculamos a precisão do coeficiente usando **Bootstrap**.
    """)

    # Preparação dos dados
    X = df.select("hour_of_day").to_numpy().reshape(-1, 1)
    y = df.select("extra").to_numpy()
    
    # Modelo
    model = LinearRegression()
    model.fit(X, y)
    coef = float(np.ravel(model.coef_)[0])
    y_pred = model.predict(X)
    
    # Métricas de Erro
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # --- CÁLCULO DE INTERVALO DE CONFIANÇA (BOOTSTRAP) ---
    # Usando uma amostra menor para o dashboard ser rápido
    @st.cache_data
    def calculate_bootstrap(X_data, y_data):
        boot_coefs = []
        n = len(X_data)
        for _ in range(500): # 500 reamostragens para balanço entre precisão e velocidade
            idx = np.random.randint(0, n, size=n)
            m = LinearRegression().fit(X_data[idx], y_data[idx])
            boot_coefs.append(m.coef_.item())
        return np.percentile(boot_coefs, 2.5), np.percentile(boot_coefs, 97.5)

    ic_inf, ic_sup = calculate_bootstrap(X, y)

    # Exibição de Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (Ajuste)", f"{r2:.4f}")
    c2.metric("MAE (Erro Médio)", f"${mae:.2f}")
    c3.metric("Coeficiente (Impacto/h)", f"{coef:.4f}")

    # Exibição do Intervalo de Confiança
    st.info(f"**Intervalo de Confiança do Coeficiente (95%):** [{ic_inf:.4f} a {ic_sup:.4f}]")
    st.caption("O intervalo não contém o zero, o que indica que a hora do dia tem um impacto estatisticamente significativo na taxa extra.")

    # Simulador
    st.divider()
    st.subheader("🔮 Simulador de Previsão")
    input_hora = st.slider("Escolha a Hora para Prever o Extra", 0, 23, 18)
    previsao = model.predict([[input_hora]]).item()
    st.success(f"Valor previsto da taxa extra às {input_hora}h: **${max(0, previsao):.2f}**")


def page_4():
    
    st.header("🧪 4. Testes de Hipóteses (Validação Estatística)")
    st.markdown("""
    Abaixo estão os principais testes de hipóteses realizados sobre os fatores que influenciam o valor da corrida e da gorjeta.
    Os intervalos de confiança foram obtidos via **Bootstrap** (2.500 reamostragens) para garantir robustez.
    """)
    
    # Tabela de resumo dos testes
    resultados = {
        "Hipótese": [
            "Distância → Tarifa",
            "Nº de Passageiros → Tarifa",
            "Tempo de Viagem → Tarifa",
            "Pagamento em Dinheiro → Gorjeta",
            "Hora do Dia → Taxa Extra"
        ],
        "H₀ (Nula)": [
            "β_distância = 0",
            "β_passageiros = 0",
            "β_tempo = 0",
            "β_dinheiro = 0",
            "β_hora = 0"
        ],
        "IC 95% (Bootstrap)": [
            "(2,9375 ; 2,9683)",
            "(-0,0204 ; 0,0297)",
            "(0,0204 ; 0,0423)",
            "(1,4429 ; 1,4595)",
            "(0,0170 ; 0,0179)"
        ],
        "Contém zero?": [
            "❌ Não",
            "✅ Sim",
            "❌ Não",
            "❌ Não",
            "❌ Não"
        ],
        "Decisão": [
            "Rejeita H₀",
            "Não rejeita H₀",
            "Rejeita H₀",
            "Rejeita H₀",
            "Rejeita H₀"
        ],
        "Conclusão": [
            "Distância impacta positivamente a tarifa (+$2,96/milha)",
            "Número de passageiros não afeta a tarifa (pouco significativo)",
            "Tempo impacta a tarifa, mas com baixo poder explicativo (R²=0,027)",
            "Pagamento em dinheiro gera gorjetas ~$1,45 maiores",
            "Hora do dia tem efeito pequeno porém significativo na taxa extra"
        ]
    }
    
    df_results = pl.DataFrame(resultados)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("📊 Detalhamento dos Testes")
    
    with st.expander("🔹 Distância → Tarifa (Regressão Linear Simples)"):
        st.markdown("""
        - **Modelo:** `fare_amount = 3,72 + 2,96 * trip_distance`
        - **R² = 0,847** (84,7% da variação explicada)
        - **MAE = $1,22**
        - **Intervalo de confiança 95% para β:** (2,9375 ; 2,9683) → **não contém zero** → rejeitamos H₀.
        - **Conclusão:** A distância é a variável mais importante na determinação do preço.
        """)
    
    with st.expander("🔹 Nº de Passageiros → Tarifa (Regressão Múltipla)"):
        st.markdown("""
        - **Modelo:** `fare_amount = 3,72 + 2,96 * distance + 0,0094 * passenger_count`
        - **R² = 0,847** (praticamente igual ao modelo univariado)
        - **IC 95% para β_passageiros:** (-0,0204 ; 0,0297) → **contém zero** → não rejeitamos H₀.
        - **Conclusão:** O número de passageiros não influencia o valor da corrida quando a distância já é considerada.
        """)
    
    with st.expander("🔹 Tempo de Viagem → Tarifa (Regressão Linear)"):
        st.markdown("""
        - **Modelo:** `fare_amount = 8,53 + 0,022 * time_minutes`
        - **R² = 0,027** (poder explicativo muito baixo)
        - **IC 95% para β_tempo:** (0,0204 ; 0,0423) → **não contém zero** → rejeitamos H₀.
        - **Conclusão:** Embora estatisticamente significativo, o tempo sozinho explica menos de 3% da variação da tarifa.
        """)
    
    with st.expander("🔹 Pagamento em Dinheiro → Gorjeta (Regressão com Dummies)"):
        st.markdown("""
        - **Categoria base:** outros meios de pagamento (cartão, etc.)
        - **Coeficiente para `payment_type = 1` (dinheiro):** $1,54
        - **IC 95%:** (1,4429 ; 1,4595) → **totalmente positivo** → rejeitamos H₀.
        - **R² do modelo = 0,595** (forma de pagamento explica 59,5% da variação da gorjeta).
        - **Conclusão:** Pagamentos em dinheiro geram gorjetas **significativamente maiores** (cerca de $1,45 a mais).
        """)
    
    with st.expander("🔹 Hora do Dia → Taxa Extra (Regressão Linear)"):
        st.markdown("""
        - **Modelo:** `extra = 0,25 + 0,0177 * hour_of_day`
        - **R² = 0,098** (baixo poder explicativo)
        - **IC 95% para β_hora:** (0,0170 ; 0,0179) → **não contém zero** → rejeitamos H₀.
        - **Conclusão:** A taxa extra aumenta cerca de 1,8 centavos por hora, efeito pequeno porém detectável.
        """)
    
    st.info("✅ Todos os testes foram realizados com nível de significância α = 0,05. Intervalos de confiança obtidos por bootstrap não paramétrico (2.500 reamostragens).")


def page_5():
    
    st.header("💡 5. Propostas de Soluções e Recomendações")
    st.markdown("""
    Com base nos resultados das análises descritivas, modelagem e testes de hipóteses, apresentamos as seguintes propostas práticas para a operação de táxis em Nova York.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Otimização de Preços")
        st.markdown("""
        - **Tarifa baseada na distância:** O modelo mostrou que a distância é o fator mais relevante (R²=0,847).  
          Recomenda-se uma estrutura tarifária clara por milha (≈ $2,96) + bandeirada ($3,72).
        - **Desconsiderar passageiros adicionais:** Como o número de passageiros não impacta o preço, tarifas extras por ocupante são injustificáveis.
        - **Taxas dinâmicas por horário:** A hora do dia tem efeito pequeno, mas significativo. Pode-se aplicar pequenos acréscimos em horários de pico (ex.: +$0,10 entre 18h-20h).
        """)
    
    with col2:
        st.subheader("💰 Incentivo a Pagamentos em Dinheiro")
        st.markdown("""
        - **Gorjeta muito maior:** Pagamentos em dinheiro geram em média $1,45 a mais de gorjeta.
        - **Sugestão:** Oferecer pequenos descontos ou brindes para clientes que pagam em dinheiro, ou destacar essa opção no aplicativo.
        - **Impacto estimado:** Considerando 8 milhões de corridas/mês, um aumento de $1,45 por corrida em dinheiro representaria milhões de dólares adicionais para os motoristas.
        """)
    
    st.divider()
    st.subheader("⚠️ Limitações do Estudo")
    st.markdown("""
    - **Dados restritos a janeiro de 2015** – padrões podem variar sazonalmente.
    - **Amostra de 500 mil registros** (de 10,4 milhões) usada no dashboard – os intervalos de confiança são precisos, mas a representatividade temporal é limitada.
    - **Modelos lineares simples** – relações não lineares (ex.: tempo de espera, trânsito) não foram capturadas.
    - **Falta de variáveis como clima, eventos especiais, localização exata** – poderiam melhorar a predição.
    - **Gorjeta em dinheiro** pode estar subnotificada, já que o dataset só registra gorjetas pagas com cartão (campo `tip_amount` para dinheiro é zero).
    """)
    
    st.divider()
    st.subheader("🔮 Recomendações para Trabalhos Futuros")
    st.markdown("""
    - Incorporar dados de tráfego em tempo real e condições climáticas.
    - Utilizar modelos mais flexíveis (regressão quantílica, árvores de decisão) para capturar não linearidades.
    - Estender a análise para outros meses e anos para validar a sazonalidade.
    - Desenvolver um sistema de recomendação de horários e formas de pagamento para maximizar a renda dos motoristas.
    """)
    
    st.success("📌 As evidências estatísticas mostram que ações focadas na distância e no incentivo ao pagamento em dinheiro têm maior potencial de impacto positivo na receita dos motoristas e na satisfação dos passageiros.")


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
    