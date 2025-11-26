import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
import altair as alt
from collections import Counter

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Previsão Presidenciais 2026 PT",
    page_icon="🇵🇹",
    layout="wide" # Mudei para WIDE para caberem as tabelas ao lado dos gráficos
)

# Título
st.title("🇵🇹 Simulador Presidenciais 2026")
st.markdown("##### Baseado em **Monte Carlo** (10.000 simulações) com dados da Wikipédia em tempo real.")

# ==========================================
# 1. CONFIGURAÇÕES & DADOS
# ==========================================
NUM_SIMULACOES = 10000
MARGEM_ERRO = 0.03
URL_WIKIPEDIA = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2026_Portuguese_presidential_election"

KEYWORDS = {
    'Gouveia': 'Almirante G. Melo',
    'Melo': 'Almirante G. Melo',
    'Mendes': 'Marques Mendes',
    'Ventura': 'André Ventura',
    'Seguro': 'António J. Seguro',
    'Costa': 'António Costa',
    'Cotrim': 'Cotrim Figueiredo',
    'Martins': 'Catarina Martins',
    'Filipe': 'António Filipe',
    'Santos': 'Pedro Nuno Santos',
    'Rio': 'Rui Rio',
    'Mortágua': 'Mariana Mortágua',
    'Raimundo': 'Paulo Raimundo'
}

# ==========================================
# 2. MOTOR DE DADOS
# ==========================================
@st.cache_data(ttl=3600)
def obter_dados():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(URL_WIKIPEDIA, headers=headers)
        dfs = pd.read_html(StringIO(r.text), header=0)
        
        df_final = None
        cols_map = {}
        
        for df in dfs:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(map(str, col)).strip() for col in df.columns]
            df.columns = [str(c) for c in df.columns]
            
            matches = {}
            for col in df.columns:
                for key, nome_real in KEYWORDS.items():
                    if key in col and nome_real not in matches.values():
                        matches[col] = nome_real
            
            if len(matches) >= 3:
                df_final = df
                cols_map = matches
                break
        
        if df_final is None: return None

        df_limpo = df_final[list(cols_map.keys())].rename(columns=cols_map)
        
        def limpar(val):
            if pd.isna(val): return 0.0
            s = str(val).strip().lower()
            if s in ['—', '-', '?', 'nan', 'tba']: return 0.0
            s = re.sub(r'\[.*?\]', '', s)
            s = re.sub(r'[a-zA-Z]', '', s)
            s = s.replace('%', '').strip()
            try: return float(s)
            except: return 0.0

        for col in df_limpo.columns:
            df_limpo[col] = df_limpo[col].apply(limpar)
            
        df_limpo = df_limpo.loc[(df_limpo.sum(axis=1) > 10)]
        return df_limpo.head(15)

    except Exception as e:
        st.error(f"Erro: {e}")
        return None

# ==========================================
# 3. LÓGICA E SIMULAÇÃO
# ==========================================
def calcular_medias_ponderadas(df):
    pesos = np.linspace(1.0, 0.4, len(df))
    medias = {}
    for col in df.columns:
        valores = df[col].values
        if np.sum(valores) > 0:
            media_pond = np.average(valores, weights=pesos)
        else:
            media_pond = 0.0
        medias[col] = media_pond
        
    series_medias = pd.Series(medias)
    soma = series_medias.sum()
    return (series_medias / soma) * 100 if soma > 0 else series_medias

def correr_simulacao(medias_norm):
    candidatos = medias_norm.index.tolist()
    vitorias = {c: 0 for c in candidatos}
    segunda_volta = []
    
    my_bar = st.progress(0)
    
    medias_array = medias_norm.values
    simulacoes = np.random.normal(loc=medias_array, scale=MARGEM_ERRO*100, size=(NUM_SIMULACOES, len(candidatos)))
    simulacoes = np.maximum(0, simulacoes)

    totais = simulacoes.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        percentagens = np.where(totais > 0, (simulacoes / totais) * 100, 0)

    ordem_indices = np.argsort(percentagens, axis=1)
    top1_idx = ordem_indices[:, -1]
    top2_idx = ordem_indices[:, -2]
    top1_perc = percentagens[np.arange(NUM_SIMULACOES), top1_idx]
    
    for i in range(NUM_SIMULACOES):
        nome_vencedor = candidatos[top1_idx[i]]
        if top1_perc[i] > 50.0001:
            vitorias[nome_vencedor] += 1
        else:
            nome_segundo = candidatos[top2_idx[i]]
            segunda_volta.append(tuple(sorted([nome_vencedor, nome_segundo])))
            
        if (i + 1) % 2500 == 0:
            my_bar.progress((i + 1) // 100)
            
    my_bar.empty()
    return vitorias, segunda_volta

# ==========================================
# 4. FRONTEND (VISUALIZAÇÃO)
# ==========================================
df_dados = obter_dados()

if df_dados is not None:
    st.subheader("📊 Médias das Sondagens (Time Decay)")
    medias_finais = calcular_medias_ponderadas(df_dados)
    
    # --- MOSTRAR MÉDIAS (Layout Top 8) ---
    todos_candidatos = medias_finais.sort_values(ascending=False)
    col_kpi1 = st.columns(4)
    for i, (cand, val) in enumerate(todos_candidatos.head(4).items()):
        col_kpi1[i].metric(label=cand, value=f"{val:.1f}%")
        
    if len(todos_candidatos) > 4:
        col_kpi2 = st.columns(4)
        resto = todos_candidatos.iloc[4:8]
        for i, (cand, val) in enumerate(resto.items()):
            if i < 4: col_kpi2[i].metric(label=cand, value=f"{val:.1f}%")

    st.markdown("---")
    
    # --- BOTÃO ---
    if st.button('🎲 Correr Simulação Monte Carlo (10.000 Eleições)', type="primary"):
        v1, v2 = correr_simulacao(medias_finais)
        
        st.header("🏆 Resultados Detalhados da Previsão")
        
        # === SECÇÃO 1: VITÓRIA À 1ª VOLTA ===
        st.subheader("1. Probabilidade de Vitória Direta (1ª Volta)")
        
        # Criar Tabela de Dados
        df_vitoria = pd.DataFrame(list(v1.items()), columns=['Candidato', 'Vitorias'])
        df_vitoria['Probabilidade (%)'] = (df_vitoria['Vitorias'] / NUM_SIMULACOES) * 100
        df_vitoria = df_vitoria.sort_values('Probabilidade (%)', ascending=False)
        df_vitoria = df_vitoria[df_vitoria['Probabilidade (%)'] > 0] # Remove quem tem 0% absoluto
        
        c1, c2 = st.columns([2, 1]) # Coluna da esquerda maior (gráfico), direita menor (tabela)
        
        with c1:
            if not df_vitoria.empty:
                chart = alt.Chart(df_vitoria).mark_bar().encode(
                    x=alt.X('Probabilidade (%)', title='Probabilidade (%)'),
                    y=alt.Y('Candidato', sort='-x', title=None),
                    color=alt.value("#2ecc71"),
                    tooltip=['Candidato', 'Probabilidade (%)']
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Nenhum candidato conseguiu vencer à 1ª volta nas 10.000 simulações.")
                
        with c2:
            st.write("**Dados Detalhados:**")
            if not df_vitoria.empty:
                st.dataframe(
                    df_vitoria[['Candidato', 'Probabilidade (%)']].style.format({"Probabilidade (%)": "{:.2f}%"}),
                    hide_index=True
                )
            else:
                st.write("Probabilidade < 0.01% para todos.")

        # === SECÇÃO 2: SEGUNDA VOLTA ===
        st.divider()
        st.subheader("2. Duelos de 2ª Volta (Cenários Mais Prováveis)")
        
        if v2:
            contagem = Counter(v2).most_common(10) # TOP 10 Cenários
            
            df_2v = pd.DataFrame(contagem, columns=['Par', 'Qtd'])
            df_2v['Probabilidade (%)'] = (df_2v['Qtd'] / len(v2)) * 100
            df_2v['Duelo'] = df_2v['Par'].apply(lambda x: f"{x[0]} vs {x[1]}")
            
            col_chart, col_table = st.columns([1.5, 1])
            
            with col_chart:
                # Gráfico Top 5
                chart2 = alt.Chart(df_2v.head(5)).mark_bar().encode(
                    x=alt.X('Probabilidade (%)'),
                    y=alt.Y('Duelo', sort='-x', title=None),
                    color=alt.value("#ff4b4b"),
                    tooltip=['Duelo', 'Probabilidade (%)']
                )
                st.altair_chart(chart2, use_container_width=True)
                
            with col_table:
                st.write("**Top 10 Cenários Possíveis:**")
                st.dataframe(
                    df_2v[['Duelo', 'Probabilidade (%)']].style.format({"Probabilidade (%)": "{:.1f}%"}),
                    hide_index=True,
                    height=300
                )
        else:
            st.warning("Não há 2ª volta (vitória esmagadora na 1ª).")

    with st.expander("Ver dados brutos da Wikipédia"):
        st.dataframe(df_dados)

else:
    st.error("Erro ao carregar dados.")

# ==========================================
# RODAPÉ
# ==========================================
st.write("")
st.markdown("---")

st.subheader("IMPORTANTE - Bicas")
try:
    # Tenta carregar imagem local se existir, senão usa link
    st.image("meme.jpg", caption="Quem não votar Cotrim é gayyyyy", width=400)
except:
    st.image("https://i.imgflip.com/994j20.jpg", caption="Meme de recurso", width=400)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 20px;'>
        🛠️ Desenvolvido pelo <b>GOAT Francisco Gonçalves</b> 🐐 <br>
        🤖 Powered by <i>Python, Streamlit & Monte Carlo Mathematics</i>
    </div>
    """, 
    unsafe_allow_html=True
)