import os
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# Configuração da página no Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistema de Triagem e Encaminhamento Hospital/UBS",
    page_icon="🏥",
    layout="wide"
)

# Estilização CSS e Banner
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .banner-aviso {
        background-color: #fff3cd; color: #856404; padding: 12px;
        border-radius: 6px; border: 1px solid #ffeeba; text-align: center;
        font-weight: bold; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="banner-aviso">⚠ PROTÓTIPO ACADÊMICO — USO SIMULADO PARA DEMONSTRAÇÃO E APRESENTAÇÃO INSTITUCIONAL</div>',
    unsafe_allow_html=True
)

st.title("🏥 Sistema de Triagem e Vínculo Hospital ↔ UBS")
st.subheader("Redirecionamento Inteligente de Casos de Baixa Urgência")

# -----------------------------------------------------------------------------
# Carregamento dos Datasets (com Cache)
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    if os.path.exists('chief_complaints.csv'):
        df_complaints = pd.read_csv('chief_complaints.csv')
    else:
        df_complaints = pd.DataFrame({
            'queixa': [
                "Sintomas Gripais / Coriza",
                "Dor de cabeça leve / Tensão",
                "Renovação de Receita / Atestado",
                "Dor no Peito / Falta de Ar",
                "Trauma com Suspeita de Fratura",
                "Febre Alta Persistentemente acima de 39°C"
            ]
        })

    if os.path.exists('patient_history.csv'):
        df_patients = pd.read_csv('patient_history.csv')
    else:
        df_patients = pd.DataFrame([
            {"nome": "João da Silva", "idade": 35, "fc": 78, "pas": 120, "spo2": 98, "temp": 36.6, "dor": 2, "queixa": "Sintomas Gripais / Coriza"},
            {"nome": "Maria Oliveira", "idade": 62, "fc": 115, "pas": 160, "spo2": 91, "temp": 38.5, "dor": 8, "queixa": "Dor no Peito / Falta de Ar"}
        ])

    if os.path.exists('ubs_data.csv'):
        df_ubs = pd.read_csv('ubs_data.csv')
    else:
        df_ubs = pd.DataFrame([
            {"id": "UBS01", "nome": "UBS Central", "lat": -16.6800, "lon": -49.2550, "dist_km": 1.2, "espera_min": 15, "vagas_hoje": 8, "tipo": "UBS"},
            {"id": "UBS02", "nome": "UBS Jardim América", "lat": -16.6950, "lon": -49.2700, "dist_km": 2.8, "espera_min": 10, "vagas_hoje": 12, "tipo": "UBS"},
            {"id": "UBS03", "nome": "UBS Vila Nova", "lat": -16.6700, "lon": -49.2400, "dist_km": 3.5, "espera_min": 25, "vagas_hoje": 5, "tipo": "UBS"},
            {"id": "HOSP01", "nome": "Hospital Municipal (Pronto Socorro)", "lat": -16.6860, "lon": -49.2640, "dist_km": 0.0, "espera_min": 180, "vagas_hoje": 0, "tipo": "Hospital"}
        ])

    return df_complaints, df_patients, df_ubs

df_complaints, df_patients, UBS_DATA = load_datasets()

queixas_options = df_complaints['queixa'].dropna().unique().tolist() if 'queixa' in df_complaints.columns else df_complaints.iloc[:, 0].tolist()

# -----------------------------------------------------------------------------
# Interface Principal
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 📋 Ficha de Entrada do Paciente")
    
    usar_historico = st.checkbox("Carregar dados de paciente cadastrado no dataset")
    
    paciente_sel = None
    if usar_historico and not df_patients.empty:
        nomes_pacientes = df_patients['nome'].tolist() if 'nome' in df_patients.columns else [f"Paciente {i+1}" for i in range(len(df_patients))]
        idx_paciente = st.selectbox("Selecione o Paciente do Dataset", range(len(nomes_pacientes)), format_func=lambda x: nomes_pacientes[x])
        paciente_sel = df_patients.iloc[idx_paciente]

    val_nome = str(paciente_sel['nome']) if paciente_sel is not None and 'nome' in paciente_sel else "João da Silva"
    val_idade = int(paciente_sel['idade']) if paciente_sel is not None and 'idade' in paciente_sel else 35
    val_fc = int(paciente_sel['fc']) if paciente_sel is not None and 'fc' in paciente_sel else 78
    val_pas = int(paciente_sel['pas']) if paciente_sel is not None and 'pas' in paciente_sel else 120
    val_spo2 = int(paciente_sel['spo2']) if paciente_sel is not None and 'spo2' in paciente_sel else 98
    val_temp = float(paciente_sel['temp']) if paciente_sel is not None and 'temp' in paciente_sel else 36.6
    val_dor = int(paciente_sel['dor']) if paciente_sel is not None and 'dor' in paciente_sel else 2
    
    queixa_default_idx = 0
    if paciente_sel is not None and 'queixa' in paciente_sel and paciente_sel['queixa'] in queixas_options:
        queixa_default_idx = queixas_options.index(paciente_sel['queixa'])

    nome = st.text_input("Nome do Paciente", value=val_nome)
    idade = st.number_input("Idade", min_value=0, max_value=120, value=val_idade)
    
    st.markdown("**Sinais Vitais & Sintomas**")
    c1, c2 = st.columns(2)
    with c1:
        fc = st.number_input("Frequência Cardíaca (bpm)", 40, 200, val_fc)
        pas = st.number_input("PA Sistólica (mmHg)", 70, 220, val_pas)
        spo2 = st.number_input("Saturação de O₂ (%)", 70, 100, val_spo2)
    with c2:
        temp = st.number_input("Temperatura (°C)", 35.0, 42.0, val_temp, step=0.1)
        dor = st.slider("Nível de Dor (0 a 10)", 0, 10, val_dor)
        queixa = st.selectbox("Queixa Principal (Base do Dataset)", queixas_options, index=queixa_default_idx)

    btn_analisar = st.button("🔍 ANALISAR E CLASSIFICAR", type="primary", use_container_width=True)

with col2:
    st.markdown("### 🎯 Resultado da Triagem")
    
    if btn_analisar or 'classificado' not in st.session_state:
        st.session_state['classificado'] = True
        
        is_critico = (spo2 < 92) or (pas > 180) or (fc > 130) or ("Peito" in queixa) or ("Trauma" in queixa) or (dor >= 8)
        
        if is_critico:
            nivel = "Nível 2 — Muito Urgente (Laranja)" if dor >= 8 else "Nível 1 — Emergência (Vermelho)"
            st.error(f"**PRIORIDADE ALTA:** {nivel}")
            st.warning("🚨 **ENCAMINHAMENTO:** Manter atendimento no **Hospital Municipal (Ficha de Emergência)**.")
            st.metric(label="Tempo Estimado para Atendimento Hospitalar", value="Imediato / Atendimento Prioritário")
        else:
            nivel = "Nível 4 — Pouco Urgente (Verde)" if dor > 3 else "Nível 5 — Não Urgente (Azul)"
            st.success(f"**PRIORIDADE LEVE:** {nivel}")
            st.info("💡 **RECOMENDAÇÃO:** Encaminhar para Unidade Básica de Saúde (UBS). Fila de espera no Hospital Municipal estimada em **3 horas**.")
            
            ubss_validas = UBS_DATA[UBS_DATA['tipo'] == 'UBS'].copy()
            
            if not ubss_validas.empty:
                ubss_validas['score'] = (ubss_validas['vagas_hoje'] * 2) - (ubss_validas['dist_km'] * 1.5) - (ubss_validas['espera_min'] * 0.5)
                melhor_ubs = ubss_validas.sort_values(by='score', ascending=False).iloc[0]
                
                st.markdown(f"#### 🏥 UBS Recomendada: **{melhor_ubs['nome']}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Distância", f"{melhor_ubs['dist_km']} km")
                m2.metric("Tempo de Espera", f"~{melhor_ubs['espera_min']} min")
                m3.metric("Encaixes Hoje", f"{melhor_ubs['vagas_hoje']} vagas")

    st.markdown("### 🗺️ Rede de Atendimento Próxima")
    
    lat_centro = UBS_DATA['lat'].mean() if 'lat' in UBS_DATA.columns else -16.6860
    lon_centro = UBS_DATA['lon'].mean() if 'lon' in UBS_DATA.columns else -49.2640
    
    m = folium.Map(location=[lat_centro, lon_centro], zoom_start=13)
    
    for _, row in UBS_DATA.iterrows():
        color = "red" if row['tipo'] == 'Hospital' else "green"
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"{row['nome']} - Espera: {row['espera_min']} min",
            tooltip=row['nome'],
            icon=folium.Icon(color=color, icon="plus-sign" if row['tipo'] == 'Hospital' else "home")
        ).add_to(m)
        
    st_folium(m, width=650, height=280)
