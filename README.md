# NYC Taxi Analytics

Projeto de análise estatística e ciência de dados aplicado ao dataset **NYC Yellow Taxi Trip Data — January 2015**.  
O objetivo é explorar os dados, aplicar limpeza e engenharia de variáveis, construir modelos de regressão linear, validar hipóteses estatísticas e apresentar os resultados em um dashboard interativo desenvolvido com **Streamlit**.

## Objetivo do projeto

Este repositório foi construído para atender aos requisitos da lauda do trabalho, com foco em:

- exploração descritiva dos dados;
- tratamento de inconsistências e valores ausentes;
- engenharia de variáveis para enriquecer a análise;
- modelagem com regressão linear;
- validação estatística com testes de hipóteses e intervalos de confiança;
- entrega dos resultados em um dashboard interativo;
- proposição de recomendações práticas baseadas nos insights obtidos.

## Dataset

A análise utiliza uma amostra do dataset público **NYC Yellow Taxi Trip Data**, referente a **janeiro de 2015**.  
No projeto, os dados passam por um pipeline em camadas:

- **Bronze**: dados brutos obtidos do Kaggle;
- **Silver**: dados limpos, filtrados e preparados para análise;
- **Gold**: dados enriquecidos com nomes legíveis e colunas finalizadas para consumo no dashboard.

## Estrutura do projeto

```
yellow-taxi-trip-data/  
│  
├── data/  
│ ├── bronze/ # Dados brutos (ingestão)  
│ ├── silver/ # Dados tratados parcialmente  
│ └── gold/ # Dados finais prontos para análise (parquet)  
│  
├── notebooks/  
│ ├── 01_eda.ipynb # Análise exploratória dos dados  
│ └── 02_modelling.ipynb # Modelagem estatística e testes de hipótese  
│  
├── app/  
│ └── main.py # Dashboard interativo (Streamlit)  
│  
├── requirements.txt # Dependências do projeto  
├── README.md # Documentação do projeto  
└── .gitignore
```

### Descrição das Camadas de Dados

- **Bronze:** Dados brutos, sem tratamento.
- **Silver:** Dados limpos parcialmente (remoção de inconsistências e nulos).
- **Gold:** Dados refinados, com features prontas para análise e modelagem.

## Pipeline de dados

### 1) Ingestão
O notebook `01_eda.ipynb` faz o download do arquivo bruto com `kagglehub` e salva em `data/bronze/`.

### 2) Limpeza e preparação
A camada **Silver** aplica as seguintes regras de negócio:

- `passenger_count` deve estar entre **1 e 4**;
- `trip_distance` deve ser **menor ou igual a 10 milhas**;
- coordenadas de embarque e desembarque devem ficar dentro do perímetro de NYC:
  - longitude entre **-74.3 e -73.7**;
  - latitude entre **40.5 e 40.9**;
- `rate_code_id` deve estar entre **1 e 6**;
- `fare_amount` deve estar entre **0 e 55**;
- `extra` deve estar entre **0 e 2**;
- `mta_tax` deve ser **maior ou igual a 0**;
- `tip_amount` deve estar entre **0 e 5**;
- `tolls_amount` deve estar entre **0 e 1**;
- `total_amount` deve ser **maior ou igual a 0**;
- linhas com nulos são removidas;
- as colunas de data/hora são convertidas para datetime;
- as corridas são restringidas ao período de **01/01/2015 a 31/01/2015**.

### 3) Engenharia de variáveis
Após a limpeza, o projeto cria as seguintes variáveis derivadas:

- `trip_duration_minutes` — duração da corrida em minutos;
- `hour_of_day` — hora do embarque;
- `day_of_week` — dia da semana do embarque;
- `speed_mph` — velocidade média estimada em milhas por hora.

### 4) Camada Gold
Na camada **Gold**, os códigos numéricos são transformados em rótulos legíveis:

- `vendor_id`
- `rate_code_id`
- `payment_type`

Essa camada é a base consumida pelo dashboard.

## Regras de negócio para análise e modelagem

As análises e os modelos seguem as seguintes premissas:

### Relações investigadas
- **Distância → tarifa**
- **Tempo de viagem → tarifa**
- **Número de passageiros → tarifa**
- **Forma de pagamento → tarifa**
- **Forma de pagamento → gorjeta**
- **Hora do dia → taxa extra**

### Regressão linear
Os modelos usam regressão linear para estimar relações entre variáveis explicativas e variáveis de resposta.

### Validação estatística
Os coeficientes são validados com:

- **intervalos de confiança por bootstrap**;
- **nível de significância de 5%**;
- decisão de rejeitar ou não rejeitar a hipótese nula com base na presença do zero no intervalo.

### Simulação/predição
O projeto também inclui previsões pontuais, como:

- estimativa da taxa extra por hora do dia;
- estimativa da tarifa com base na distância;
- comparação de comportamentos por tipo de pagamento.

## Principais análises realizadas

### EDA
O notebook `01_eda.ipynb` cobre:

- diagnóstico inicial da base;
- análise de distribuições;
- verificação de outliers;
- análise das coordenadas geográficas;
- limpeza e padronização dos dados;
- criação de variáveis derivadas;
- persistência das camadas Silver e Gold.

### Modelagem e hipóteses
O notebook `02_modelling.ipynb` cobre:

- correlação entre distância e preço;
- correlação entre tempo e preço;
- análise por número de passageiros;
- análise por tipo de pagamento;
- regressão linear para tarifa;
- regressão linear para gorjeta;
- regressão linear para taxa extra;
- testes de hipóteses para os coeficientes estimados;
- intervalos de confiança via bootstrap.

## Dashboard

O arquivo `main.py` implementa um dashboard com Streamlit dividido em cinco seções:

1. **Limpeza e preparação dos dados**
2. **Análise exploratória**
3. **Modelagem estatística**
4. **Testes de hipóteses**
5. **Propostas de soluções**

### Recursos visuais e interativos
- histogramas;
- gráficos de linha;
- matriz de correlação;
- mapas de embarque e desembarque;
- tabela-resumo dos testes estatísticos;
- simulador com slider para prever a taxa extra por hora.

## Resultados e conclusões do projeto

O projeto aponta que:

- **a distância é a variável mais forte na explicação da tarifa**;
- **o tempo de viagem também apresenta efeito estatisticamente significativo**, embora com baixo poder explicativo isolado;
- **pagamentos em dinheiro aparecem associados a gorjetas maiores** no conjunto analisado;
- **a hora do dia tem efeito pequeno, mas detectável, sobre a taxa extra**.

Esses achados sustentam recomendações práticas para precificação, incentivo a formas de pagamento e análise operacional de corridas.

## Requisitos de instalação

O projeto foi desenvolvido em Python e utiliza as bibliotecas listadas em `requirements.txt`.

### Dependências principais
- `kagglehub`
- `polars`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `plotly`
- `streamlit`

## Como instalar

### 1. Clonar o repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Criar e ativar um ambiente virtual
```bash
python -m venv .venv
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Gerar os dados processados
Execute os notebooks na ordem abaixo para criar as camadas do projeto:

1. `01_eda.ipynb`
2. `02_modelling.ipynb`

Ao final, a estrutura esperada é:

```text
data/
├── bronze/
├── silver/
└── gold/
```

## Como executar o dashboard

Depois que o arquivo Gold estiver disponível, inicie a aplicação:

```bash
streamlit run main.py
```

## Observações importantes

- O dashboard carrega o arquivo `data/gold/yellow_tripdata_2015-01.parquet`.
- A visualização principal utiliza uma amostra de até **500 mil registros** para manter o desempenho da interface.
- Nos notebooks, os intervalos de confiança são estimados com **bootstrap não paramétrico**.
- Há uma diferença de implementação entre o notebook e o dashboard: no dashboard, o bootstrap do coeficiente da hora usa **500 reamostragens** para manter a resposta rápida; nos notebooks, a documentação estatística considera **2.500 reamostragens**.

## Limitações

- A análise se concentra em **apenas um mês** de dados.
- O estudo usa **modelos lineares simples**, que não capturam relações não lineares mais complexas.
- Variáveis como **clima, trânsito, eventos especiais e geolocalização mais detalhada** não foram incorporadas.
- Parte das conclusões está condicionada às regras de filtragem aplicadas na camada Silver.

## Próximos passos

- expandir a análise para mais meses e anos;
- testar modelos mais flexíveis;
- incluir variáveis externas, como clima e trânsito;
- enriquecer o dashboard com filtros adicionais;
- aprofundar a comparação entre horários, regiões e tipos de pagamento.

## Licença

Projeto acadêmico desenvolvido para fins de estudo e apresentação.
