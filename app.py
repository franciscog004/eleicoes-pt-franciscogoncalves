import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import matplotlib.pyplot as plt
from io import StringIO

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Previsão Presidenciais 2026", layout="centered")

st.title("🇵🇹 Simulador Presidenciais 2026")
st.markdown("Baseado em **Monte Carlo** (10.000 simulações) com dados da Wikipédia em tempo real.")

# ==========================================
# 1. CONFIGURAÇÕES
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
    'Cotrim': 'Cotrim Figueiredo',
    'Martins': 'Catarina Martins',
    'Filipe': 'António Filipe',
    'Santos': 'Pedro Nuno Santos'
}

# ==========================================
# 2. MOTOR DE DADOS
# ==========================================
@st.cache_data(ttl=3600) # Guarda em cache por 1 hora para não estar sempre a ler a Wiki
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
            s = str(val).strip()
            if s in ['—', '-', '?', 'nan']: return 0.0
            s = re.sub(r'\[.*?\]', '', s)
            s = re.sub(r'[a-zA-Z]', '', s)
            s = s.replace('%', '').strip()
            try: return float(s)
            except: return 0.0

        for col in df_limpo.columns:
            df_limpo[col] = df_limpo[col].apply(limpar)
            
        df_limpo = df_limpo.loc[(df_limpo.sum(axis=1) > 10)]
        
        # COMPLEXIDADE EXTRA: Pega nas 10 últimas, mas vamos dar pesos
        return df_limpo.head(10)

    except Exception as e:
        st.error(f"Erro ao ler dados: {e}")
        return None

# ==========================================
# 3. LÓGICA COMPLEXA (Weighted Average)
# ==========================================
def calcular_medias_ponderadas(df):
    # Criar pesos: A mais recente (índice 0) vale 1.0, a seguinte 0.9, etc.
    # Isto simula o "Time Decay" (sondagens velhas valem menos)
    pesos = np.linspace(1.0, 0.5, len(df)) # Decresce de 1.0 até 0.5
    
    medias = {}
    for col in df.columns:
        valores = df[col].values
        # Média ponderada = Soma(Valor * Peso) / Soma(Pesos)
        media_pond = np.average(valores, weights=pesos)
        medias[col] = media_pond
        
    # Normalizar para 100%
    series_medias = pd.Series(medias)
    return (series_medias / series_medias.sum()) * 100

def correr_simulacao(medias_norm):
    candidatos = medias_norm.index.tolist()
    vitorias = {c: 0 for c in candidatos}
    segunda_volta = []
    
    progresso = st.progress(0)
    
    # Otimização com Numpy (vetorizado) para ser rápido no site
    # Geramos uma matriz gigante [10000, n_candidatos] de uma vez
    means = medias_norm.values
    # Gerar 10.000 eleições de uma vez
    simulacoes = np.random.normal(loc=means, scale=MARGEM_ERRO*100, size=(NUM_SIMULACOES, len(candidatos)))
    simulacoes = np.maximum(0, simulacoes) # Remover negativos
    
    # Calcular totais de cada linha para saber %
    totais = simulacoes.sum(axis=1)[:, np.newaxis]
    percentagens = (simulacoes / totais) * 100
    
    # Quem ganhou cada simulação?
    # argsort devolve os índices ordenados. Pegamos nos 2 últimos (os maiores)
    ordem = np.argsort(percentagens, axis=1)
    vencedores_idx = ordem[:, -1] # O maior
    segundos_idx = ordem[:, -2]   # O segundo maior
    
    vencedores_val = percentagens[np.arange(NUM_SIMULACOES), vencedores_idx]
    
    # Contabilizar
    for i in range(NUM_SIMULACOES):
        idx_vencedor = vencedores_idx[i]
        nome_vencedor = candidatos[idx_vencedor]
        
        if vencedores_val[i] > 50.0:
            vitorias[nome_vencedor] += 1
        else:
            idx_segundo = segundos_idx[i]
            nome_segundo = candidatos[idx_segundo]
            par = tuple(sorted([nome_vencedor, nome_segundo]))
            segunda_volta.append(par)
            
    progresso.progress(100)
    return vitorias, segunda_volta

# ==========================================
# 4. INTERFACE GRÁFICA (FRONTEND)
# ==========================================
df = obter_dados()

if df is not None:
    st.subheader("📊 Médias Ponderadas (Time Decay)")
    st.caption("As sondagens mais recentes têm mais peso no cálculo.")
    
    medias = calcular_medias_ponderadas(df)
    
    # Mostrar colunas com métricas bonitas
    col1, col2, col3, col4 = st.columns(4)
    top_4 = medias.sort_values(ascending=False).head(4)
    cols = [col1, col2, col3, col4]
    
    for i, (cand, val) in enumerate(top_4.items()):
        cols[i].metric(label=cand, value=f"{val:.1f}%")
    
    if st.button('🎲 Correr Simulação Monte Carlo'):
        v1, v2 = correr_simulacao(medias)
        
        st.divider()
        st.subheader("🏆 Resultados da Previsão")
        
        # 1. VITÓRIA DIRETA
        prob_vitoria = {k: (v/NUM_SIMULACOES)*100 for k, v in v1.items() if v > 0}
        if any(p > 0.5 for p in prob_vitoria.values()):
            st.success("Há probabilidade de vitória à 1ª volta!")
            st.bar_chart(prob_vitoria)
        else:
            st.info("Probabilidade de vitória à 1ª volta é inferior a 0.5% para todos os candidatos.")
            
        # 2. SEGUNDA VOLTA
        st.subheader("⚔️ Cenários de 2ª Volta")
        from collections import Counter
        contagem = Counter(v2).most_common(5)
        
        df_2v = pd.DataFrame(contagem, columns=['Cenario', 'Qtd'])
        df_2v['Probabilidade (%)'] = (df_2v['Qtd'] / len(v2)) * 100
        df_2v['Duelo'] = df_2v['Cenario'].apply(lambda x: f"{x[0]} vs {x[1]}")
        
        # Gráfico bonito
        st.altair_chart(
            st.bar_chart(df_2v.set_index('Duelo')['Probabilidade (%)'], color="#ff4b4b").data
        )
        
        # Tabela detalhada
        st.table(df_2v[['Duelo', 'Probabilidade (%)']].head())

    with st.expander("Ver dados brutos da Wikipédia"):
        st.dataframe(df)

else:
    st.error("Não foi possível carregar os dados.")