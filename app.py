import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO

# ==========================================
# CONFIGURAÇÃO DA PÁGINA (TÍTULO E ÍCONE)
# ==========================================
st.set_page_config(
    page_title="Previsão Presidenciais 2026 PT",
    page_icon="🇵🇹",
    layout="centered"
)

# Título Principal no Site
st.title("🇵🇹 Simulador Presidenciais 2026")
st.markdown("##### Baseado em **Monte Carlo** (10.000 simulações) com dados da Wikipédia em tempo real.")

# ==========================================
# 1. DEFINIÇÕES
# ==========================================
NUM_SIMULACOES = 10000
MARGEM_ERRO = 0.03
# Usamos a Wikipédia em Inglês porque a tabela de dados é mais limpa e consistente
URL_WIKIPEDIA = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2026_Portuguese_presidential_election"

# Mapeamento de palavras-chave para nomes bonitos
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
# 2. MOTOR DE DADOS (SCRAPING & LIMPEZA)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="A carregar sondagens da Wikipédia...")
def obter_dados():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(URL_WIKIPEDIA, headers=headers)
        dfs = pd.read_html(StringIO(r.text), header=0)
        
        df_final = None
        cols_map = {}
        
        # Procura a tabela correta na página
        for df in dfs:
            # Achatar cabeçalhos duplos se existirem
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(map(str, col)).strip() for col in df.columns]
            df.columns = [str(c) for c in df.columns]
            
            # Verifica se encontra candidatos conhecidos nas colunas
            matches = {}
            for col in df.columns:
                for key, nome_real in KEYWORDS.items():
                    if key in col and nome_real not in matches.values():
                        matches[col] = nome_real
            
            # Se tiver pelo menos 3, assumimos que é a tabela certa
            if len(matches) >= 3:
                df_final = df
                cols_map = matches
                break
        
        if df_final is None: return None

        # Filtrar colunas e limpar dados
        df_limpo = df_final[list(cols_map.keys())].rename(columns=cols_map)
        
        def limpar(val):
            if pd.isna(val): return 0.0
            s = str(val).strip().lower()
            if s in ['—', '-', '?', 'nan', 'tba']: return 0.0
            s = re.sub(r'\[.*?\]', '', s) # Remove notas [a]
            s = re.sub(r'[a-zA-Z]', '', s) # Remove letras soltas
            s = s.replace('%', '').strip()
            try: return float(s)
            except: return 0.0

        for col in df_limpo.columns:
            df_limpo[col] = df_limpo[col].apply(limpar)
            
        # Remove linhas inválidas (soma < 10%)
        df_limpo = df_limpo.loc[(df_limpo.sum(axis=1) > 10)]
        
        # Pega nas 15 últimas sondagens para ter histórico suficiente
        return df_limpo.head(15)

    except Exception as e:
        st.error(f"Erro técnico ao ler dados: {e}")
        return None

# ==========================================
# 3. CÁLCULOS (MÉDIAS & SIMULAÇÃO)
# ==========================================
def calcular_medias_ponderadas(df):
    # Time Decay: Sondagens mais recentes valem mais
    # Cria pesos de 1.0 até 0.4
    pesos = np.linspace(1.0, 0.4, len(df))
    
    medias = {}
    for col in df.columns:
        valores = df[col].values
        # Evita erro se uma coluna estiver toda a zeros
        if np.sum(valores) > 0:
            media_pond = np.average(valores, weights=pesos)
        else:
            media_pond = 0.0
        medias[col] = media_pond
        
    # Normalizar para 100%
    series_medias = pd.Series(medias)
    soma = series_medias.sum()
    if soma > 0:
        return (series_medias / soma) * 100
    else:
        return series_medias # Retorna tudo a zero se não houver dados

def correr_simulacao(medias_norm):
    candidatos = medias_norm.index.tolist()
    vitorias = {c: 0 for c in candidatos}
    segunda_volta = []
    
    # Barra de progresso para dar feedback visual
    my_bar = st.progress(0)

    # --- OTIMIZAÇÃO NUMPY (Para ser rápido) ---
    medias_array = medias_norm.values
    # Gera 10.000 cenários de uma só vez (Matriz gigante)
    simulacoes = np.random.normal(loc=medias_array, scale=MARGEM_ERRO*100, size=(NUM_SIMULACOES, len(candidatos)))
    simulacoes = np.maximum(0, simulacoes) # Remove votos negativos

    # Calcula percentagens para cada simulação
    totais = simulacoes.sum(axis=1)[:, np.newaxis]
    # Evita divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        percentagens = np.where(totais > 0, (simulacoes / totais) * 100, 0)

    # Encontra vencedores e segundos lugares
    # argsort dá os índices do menor para o maior. Pegamos nos últimos dois.
    ordem_indices = np.argsort(percentagens, axis=1)
    top1_idx = ordem_indices[:, -1]
    top2_idx = ordem_indices[:, -2]
    
    # Vê a percentagem do vencedor em cada simulação
    top1_perc = percentagens[np.arange(NUM_SIMULACOES), top1_idx]
    
    # --- CONTABILIZAÇÃO ---
    for i in range(NUM_SIMULACOES):
        nome_vencedor = candidatos[top1_idx[i]]
        
        if top1_perc[i] > 50.0001: # Margem mínima para evitar empates float
            vitorias[nome_vencedor] += 1
        else:
            nome_segundo = candidatos[top2_idx[i]]
            # Guarda o par ordenado alfabeticamente para contar cenários iguais
            segunda_volta.append(tuple(sorted([nome_vencedor, nome_segundo])))
            
        # Atualiza barra de progresso a cada 2500 iterações
        if (i + 1) % 2500 == 0:
            my_bar.progress((i + 1) // 100)
            
    my_bar.empty() # Limpa a barra no fim
    return vitorias, segunda_volta

# ==========================================
# 4. INTERFACE DO SITE (FRONTEND)
# ==========================================
df_dados = obter_dados()

if df_dados is not None and not df_dados.empty:
    st.subheader("📊 Médias das Sondagens (Time Decay)")
    st.caption("Média ponderada das últimas 15 sondagens. As mais recentes têm mais peso.")
    
    medias_finais = calcular_medias_ponderadas(df_dados)
    
    # --- MOSTRAR MAIS CANDIDATOS (Top 8 em duas linhas) ---
    todos_candidatos = medias_finais.sort_values(ascending=False)
    
    # Linha 1 (Top 4)
    cols1 = st.columns(4)
    for i, (cand, val) in enumerate(todos_candidatos.head(4).items()):
        cols1[i].metric(label=cand, value=f"{val:.1f}%")
        
    # Linha 2 (Do 5º ao 8º lugar - Onde o Cotrim deve aparecer)
    if len(todos_candidatos) > 4:
        cols2 = st.columns(4)
        resto = todos_candidatos.iloc[4:8]
        for i, (cand, val) in enumerate(resto.items()):
             # Verifica se existe coluna disponível (para não dar erro se houver poucos candidatos)
            if i < 4:
                cols2[i].metric(label=cand, value=f"{val:.1f}%")

    st.write("") # Espaço
    
    # --- BOTÃO DE SIMULAÇÃO ---
    if st.button('🎲 Correr Simulação Monte Carlo (10k)', type="primary"):
        v1, v2 = correr_simulacao(medias_finais)
        
        st.divider()
        st.header("🏆 Resultados da Previsão")
        
        # 1. RESULTADOS DA 1ª VOLTA
        st.subheader("Probabilidade de Vitória à 1ª Volta (>50%)")
        prob_vitoria = {k: (v/NUM_SIMULACOES)*100 for k, v in v1.items() if v > 0}
        
        if any(p > 0.5 for p in prob_vitoria.values()):
            # Mostra gráfico se alguém tiver hipótese realista
            st.bar_chart(prob_vitoria, color="#2ecc71")
        else:
            # Mostra aviso se for tudo muito baixo
            st.info("ℹ️ A probabilidade de qualquer candidato vencer logo à 1ª volta é estatisticamente nula (<0.5%) com os dados atuais.")
            
        # 2. RESULTADOS DA 2ª VOLTA
        st.subheader("⚔️ Cenários Mais Prováveis de 2ª Volta")
        st.caption("Se ninguém tiver 50%, estes são os duelos finais mais prováveis.")
        
        from collections import Counter
        if v2:
            contagem = Counter(v2).most_common(5) # Top 5 cenários
            
            df_2v = pd.DataFrame(contagem, columns=['Cenario_Tuple', 'Qtd'])
            df_2v['Probabilidade (%)'] = (df_2v['Qtd'] / len(v2)) * 100
            df_2v['Duelo Final'] = df_2v['Cenario_Tuple'].apply(lambda x: f"{x[0]} vs {x[1]}")
            
            # Gráfico de barras horizontal bonito
            import altair as alt
            chart = alt.Chart(df_2v).mark_bar(color='#ff4b4b').encode(
                x=alt.X('Probabilidade (%)', title='Probabilidade (%)'),
                y=alt.Y('Duelo Final', sort='-x', title=None),
                tooltip=['Duelo Final', alt.Tooltip('Probabilidade (%)', format='.1f')]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
            
        else:
            st.warning("Não há dados suficientes para simular uma 2ª volta.")

    # Expander para os curiosos verem os dados brutos
    with st.expander("Ver tabela de dados brutos da Wikipédia"):
        st.dataframe(df_dados)

else:
    st.error("❌ Não foi possível carregar dados da Wikipédia. O formato da tabela pode ter mudado.")

# ==========================================
# RODAPÉ E MEME
# ==========================================
st.write("")
st.write("")
st.markdown("---")

# SECÇÃO DO MEME
# Podes trocar o link abaixo por qualquer link de imagem da internet (ex: imgur)
st.subheader("IMPORTANTE")
st.image(
    "meme.jpg", # <- TROCA ESTE LINK PELO TEU MEME!
    caption="Quem não votar Cotrim é gayyyyyy",
    width=400 # Podes ajustar o tamanho
)

# ASSINATURA DO GOAT
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