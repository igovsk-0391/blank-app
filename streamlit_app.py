import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página (deve ser a primeira chamada do Streamlit)
st.set_page_config(
    page_title="IA de Triagem Médica",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================

@st.cache_data
def load_or_generate_data():
    """
    Simula o dataset TriageGeist para o protótipo.
    Para usar seu arquivo CSV real, substitua o conteúdo desta função por:
    return pd.read_csv('triagegeist.csv')
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Gerando dados sintéticos baseados em parâmetros vitais
    data = {
        'Idade': np.random.randint(1, 90, n_samples),
        'Pressao_Sistolica': np.random.normal(120, 20, n_samples),
        'Frequencia_Cardiaca': np.random.normal(80, 15, n_samples),
        'Temperatura': np.random.normal(36.8, 0.8, n_samples),
        'Saturacao_O2': np.random.normal(97, 3, n_samples),
        'Nivel_Dor': np.random.randint(0, 11, n_samples),
        'Glasgow': np.random.choice([15, 14, 13, 12, 9, 8], n_samples, p=[0.8, 0.05, 0.05, 0.05, 0.03, 0.02])
    }
    df = pd.DataFrame(data)
    
    # Clipando valores para limites realistas
    df['Saturacao_O2'] = df['Saturacao_O2'].clip(70, 100)
    df['Pressao_Sistolica'] = df['Pressao_Sistolica'].clip(70, 220)
    
    # Criando a variável alvo (Destino: 0 = UBS, 1 = Hospital/UPA)
    # Regra lógica para o modelo aprender: sinais vitais alterados = Hospital
    df['Destino'] = np.where(
        (df['Saturacao_O2'] < 92) | 
        (df['Frequencia_Cardiaca'] > 120) | 
        (df['Frequencia_Cardiaca'] < 50) |
        (df['Pressao_Sistolica'] > 180) |
        (df['Glasgow'] < 14) |
        (df['Nivel_Dor'] >= 8), 
        1, 0
    )
    
    # Adicionando um pouco de ruído para o modelo não ficar 100% perfeito (realismo)
    ruido = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    df['Destino'] = np.abs(df['Destino'] - ruido)
    
    return df

@st.cache_resource
def train_model(df):
    """Treina o modelo Random Forest com os dados fornecidos."""
    X = df.drop('Destino', axis=1)
    y = df['Destino']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X, accuracy, y_test, y_pred

# Carregar dados e treinar modelo
df = load_or_generate_data()
model, X_features, accuracy, y_test, y_pred = train_model(df)

# ==========================================
# 2. INTERFACE DE NAVEGAÇÃO (SIDEBAR)
# ==========================================

st.sidebar.title("⚕️️ Menu de Navegação")
pagina = st.sidebar.radio("Selecione a página:", ["Triagem de Pacientes", "Dashboard & Performance da IA"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Protótipo de Triagem Inteligente**\n\n"
    "Este modelo utiliza Machine Learning para sugerir o encaminhamento adequado "
    "entre Atenção Básica (UBS) e Urgência/Emergência (Hospital)."
)

# ==========================================
# 3. PÁGINA 1: FORMULÁRIO DE TRIAGEM
# ==========================================

if pagina == "Triagem de Pacientes":
    st.title("Triagem Inteligente de Pacientes")
    st.write("Insira os sinais vitais e informações do paciente para receber a recomendação de direcionamento.")
    
    with st.form("triage_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            idade = st.number_input("Idade", min_value=0, max_value=120, value=30)
            pas = st.number_input("Pressão Sistólica (mmHg)", min_value=50, max_value=250, value=120)
            fc = st.number_input("Freq. Cardíaca (bpm)", min_value=30, max_value=220, value=80)
            
        with col2:
            temp = st.number_input("Temperatura (°C)", min_value=30.0, max_value=43.0, value=36.5, step=0.1)
            sat = st.number_input("Saturação O2 (%)", min_value=50, max_value=100, value=98)
            
        with col3:
            dor = st.slider("Nível de Dor (0-10)", 0, 10, 0)
            glasgow = st.selectbox("Escala de Glasgow", [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3], index=0)
            
        submit_button = st.form_submit_button(label="Realizar Triagem")
        
    if submit_button:
        # Preparar dados para predição
        input_data = pd.DataFrame([[idade, pas, fc, temp, sat, dor, glasgow]], 
                                  columns=X_features.columns)
        
        # Realizar predição e probabilidade
        prediction = model.predict(input_data)[0]
        probabilidade = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.subheader("Resultado da Triagem")
        
        if prediction == 0:
            st.success("🟢 **Recomendação: Unidade Básica de Saúde (UBS)**")
            st.write(f"**Confiança da IA:** {probabilidade[0]*100:.1f}%")
            st.write("*Motivo principal:* Sinais vitais estáveis. O paciente não apresenta critérios de urgência/emergência imediatos. O caso pode ser acompanhado na Atenção Básica.")
        else:
            st.error("🔴 **Recomendação: Hospital / UPA (Urgência/Emergência)**")
            st.write(f"**Confiança da IA:** {probabilidade[1]*100:.1f}%")
            st.write("*Motivo principal:* Foram detectadas alterações em sinais vitais críticos (ex: saturação, dor intensa ou alteração de consciência) que exigem avaliação médica imediata e recursos hospitalares.")

# ==========================================
# 4. PÁGINA 2: DASHBOARD & PERFORMANCE
# ==========================================

elif pagina == "Dashboard & Performance da IA":
    st.title("Visão Geral do Dataset e Performance do Modelo")
    
    # Métricas principais
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros (Dataset)", len(df))
    col2.metric("Acurácia do Modelo", f"{accuracy*100:.2f}%")
    
    casos_hospital = len(df[df['Destino'] == 1])
    col3.metric("Casos Históricos Hospitalares", f"{(casos_hospital/len(df))*100:.1f}%")
    
    st.markdown("---")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Importância das Variáveis (O que a IA mais avalia?)")
        # Gráfico de Feature Importance
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': X_features.columns, 'Importância': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importância', ascending=True)
        
        fig_imp = px.bar(feat_imp_df, x='Importância', y='Feature', orientation='h',
                         color='Importância', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with row1_col2:
        st.subheader("Distribuição de Encaminhamentos")
        # Gráfico de Pizza do Destino
        dist_df = df['Destino'].map({0: 'UBS (Baixa Complexidade)', 1: 'Hospital (Alta Complexidade)'}).value_counts().reset_index()
        dist_df.columns = ['Destino', 'Contagem']
        
        fig_pie = px.pie(dist_df, values='Contagem', names='Destino', 
                         color='Destino', color_discrete_map={'UBS (Baixa Complexidade)':'#2ca02c', 'Hospital (Alta Complexidade)':'#d62728'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Amostra do Dataset TriageGeist")
    st.dataframe(df.head(15), use_container_width=True)
