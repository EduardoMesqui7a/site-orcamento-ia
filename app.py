import io
import re
import tempfile
from typing import List, Optional

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from unidecode import unidecode

st.set_page_config(page_title="Orçamento IA - VSN", layout="wide")

MODELO_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
PESO_SEMANTICO = 0.70
PESO_FUZZY = 0.20
PESO_REGRAS = 0.10


@st.cache_resource
def carregar_modelo():
    return SentenceTransformer(MODELO_EMBEDDING)


def normalizar_texto(texto: str) -> str:
    if texto is None or pd.isna(texto):
        return ""

    texto = str(texto).strip().lower()
    texto = unidecode(texto)

    substituicoes = {
        "fck": "resistencia caracteristica",
        "mpa": "megapascal",
        "concreto armado": "concreto estrutural armado",
        "concreto simples": "concreto sem armadura",
        "divisoria": "parede divisoria vedacao compartimentacao interna",
        "drywall": "parede leve em gesso acartonado",
        "alvenaria": "parede de alvenaria vedacao",
        "parede": "vedacao parede fechamento",
        "aco": "aco armadura",
        "armacao": "armadura aco",
        "forma": "forma forma madeira compensado",
        "tubo": "tubulacao",
        "tubos": "tubulacao",
        "eletroduto": "tubulacao eletrica conduite",
        "conduite": "tubulacao eletrica eletroduto",
        "piso": "pavimentacao revestimento piso",
        "bloco": "alvenaria bloco",
        "reboco": "argamassa revestimento",
        "chapisco": "argamassa aderencia",
        "escavacao": "movimento de terra escavacao",
        "aterro": "movimento de terra aterro compactacao",
        "lastro": "camada de regularizacao lastro",
    }

    for de, para in substituicoes.items():
        texto = texto.replace(de, para)

    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def score_regras(busca_norm: str, descricao_norm: str) -> float:
    score = 0.0

    numeros_relevantes = ["5", "8", "10", "12", "15", "20", "25", "30", "35", "40", "50"]
    for numero in numeros_relevantes:
        if numero in busca_norm and numero in descricao_norm:
            score += 0.10

    pares = [
        ("concreto", "concreto"),
        ("armado", "armado"),
        ("argamassa", "argamassa"),
        ("alvenaria", "alvenaria"),
        ("divisoria", "divisoria"),
        ("drywall", "drywall"),
        ("piso", "piso"),
        ("tubulacao", "tubulacao"),
        ("eletrica", "eletrica"),
        ("hidraulica", "hidraulica"),
        ("escavacao", "escavacao"),
        ("aterro", "aterro"),
        ("forma", "forma"),
        ("aco", "aco"),
        ("vedacao", "vedacao"),
        ("bloco", "bloco"),
        ("porta", "porta"),
        ("janela", "janela"),
    ]

    for termo_busca, termo_desc in pares:
        if termo_busca in busca_norm and termo_desc in descricao_norm:
            score += 0.08

    if "divisoria" in busca_norm:
        if any(x in descricao_norm for x in ["drywall", "alvenaria", "parede", "vedacao"]):
            score += 0.20

    if "concreto" in busca_norm and "megapascal" in busca_norm:
        if "concreto" in descricao_norm and any(x in descricao_norm for x in ["megapascal", "resistencia caracteristica"]):
            score += 0.20

    return min(score, 1.0)


def carregar_excel(uploaded_file, nome_aba: Optional[str], header_index: int) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file)
    aba = nome_aba if nome_aba else xls.sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=aba, header=header_index)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def gerar_embeddings(df_base: pd.DataFrame, coluna_texto_base: str, modelo):
    textos = df_base[coluna_texto_base].fillna("").astype(str).tolist()
    textos_norm = [normalizar_texto(t) for t in textos]
    embeddings = modelo.encode(textos_norm, convert_to_tensor=True, normalize_embeddings=True)
    return textos_norm, embeddings


def buscar_melhor_item(texto_busca: str, df_base: pd.DataFrame, coluna_texto_base_norm: str, embeddings, modelo):
    busca_norm = normalizar_texto(texto_busca)
    if not busca_norm:
        return None

    emb_busca = modelo.encode([busca_norm], convert_to_tensor=True, normalize_embeddings=True)
    scores_semanticos = util.cos_sim(emb_busca, embeddings)[0].cpu().numpy()

    melhor_idx = None
    melhor_score = -1.0
    melhor_det = None

    for i, row in df_base.iterrows():
        texto_base_norm = row[coluna_texto_base_norm]
        score_sem = float(scores_semanticos[i])
        score_fuzzy = fuzz.token_set_ratio(busca_norm, texto_base_norm) / 100.0
        score_reg = score_regras(busca_norm, texto_base_norm)
        score_final = PESO_SEMANTICO * score_sem + PESO_FUZZY * score_fuzzy + PESO_REGRAS * score_reg

        if score_final > melhor_score:
            melhor_score = score_final
            melhor_idx = i
            melhor_det = {
                "score_final": round(score_final, 4),
                "score_semantico": round(score_sem, 4),
                "score_fuzzy": round(score_fuzzy, 4),
                "score_regras": round(score_reg, 4),
            }

    if melhor_idx is None:
        return None

    return melhor_idx, melhor_det


def processar_preenchimento(
    df_base: pd.DataFrame,
    df_destino: pd.DataFrame,
    coluna_busca_destino: str,
    colunas_base_retorno: List[str],
    colunas_destino_preencher: List[str],
    coluna_texto_base: str,
    score_minimo: float,
):
    modelo = carregar_modelo()

    df_base_proc = df_base.copy()
    df_destino_proc = df_destino.copy()

    df_base_proc[coluna_texto_base] = df_base_proc[coluna_texto_base].fillna("").astype(str)
    df_base_proc["__texto_base_norm__"] = df_base_proc[coluna_texto_base].apply(normalizar_texto)

    _, embeddings = gerar_embeddings(df_base_proc, coluna_texto_base, modelo)

    score_col = "IA_SCORE"
    match_col = "IA_DESCRICAO_ENCONTRADA"
    idx_col = "IA_LINHA_BASE"

    if score_col not in df_destino_proc.columns:
        df_destino_proc[score_col] = None
    if match_col not in df_destino_proc.columns:
        df_destino_proc[match_col] = None
    if idx_col not in df_destino_proc.columns:
        df_destino_proc[idx_col] = None

    total = len(df_destino_proc)
    progresso = st.progress(0)
    status = st.empty()

    for i in range(total):
        busca = df_destino_proc.at[i, coluna_busca_destino] if coluna_busca_destino in df_destino_proc.columns else None
        if busca is None or str(busca).strip() == "":
            progresso.progress((i + 1) / max(total, 1))
            continue

        res = buscar_melhor_item(
            texto_busca=str(busca),
            df_base=df_base_proc,
            coluna_texto_base_norm="__texto_base_norm__",
            embeddings=embeddings,
            modelo=modelo,
        )

        if res is None:
            progresso.progress((i + 1) / max(total, 1))
            continue

        idx_match, det = res
        if det["score_final"] < score_minimo:
            df_destino_proc.at[i, score_col] = det["score_final"]
            df_destino_proc.at[i, match_col] = "Confiança baixa"
            df_destino_proc.at[i, idx_col] = int(idx_match) + 2
            progresso.progress((i + 1) / max(total, 1))
            continue

        for col_base, col_dest in zip(colunas_base_retorno, colunas_destino_preencher):
            df_destino_proc.at[i, col_dest] = df_base_proc.at[idx_match, col_base]

        df_destino_proc.at[i, score_col] = det["score_final"]
        df_destino_proc.at[i, match_col] = df_base_proc.at[idx_match, coluna_texto_base]
        df_destino_proc.at[i, idx_col] = int(idx_match) + 2

        status.info(f"Processando linha {i + 1} de {total}")
        progresso.progress((i + 1) / max(total, 1))

    progresso.empty()
    status.empty()
    return df_destino_proc


def dataframe_para_excel_bytes(df: pd.DataFrame, nome_aba: str = "Resultado") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=nome_aba)
    output.seek(0)
    return output.getvalue()


st.title("Busca Semântica para Orçamento")
st.caption("Importe a base de dados e a planilha a preencher, escolha as colunas e gere o arquivo preenchido.")

with st.sidebar:
    st.header("Configurações")
    score_minimo = st.slider("Score mínimo para preencher", 0.0, 1.0, 0.35, 0.01)
    header_base = st.number_input("Linha do cabeçalho da base", min_value=1, value=3, step=1)
    header_dest = st.number_input("Linha do cabeçalho da planilha a preencher", min_value=1, value=1, step=1)
    st.markdown("Sugestão: se a base tem cabeçalho na linha 3 do Excel, informe 3.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Base de dados")
    arquivo_base = st.file_uploader("Importar base de dados", type=["xlsx", "xlsm", "xls"], key="base")

with col2:
    st.subheader("2. Planilha a preencher")
    arquivo_destino = st.file_uploader("Importar planilha de destino", type=["xlsx", "xlsm", "xls"], key="destino")

if arquivo_base and arquivo_destino:
    try:
        xls_base = pd.ExcelFile(arquivo_base)
        xls_dest = pd.ExcelFile(arquivo_destino)

        col3, col4 = st.columns(2)
        with col3:
            aba_base = st.selectbox("Aba da base", options=xls_base.sheet_names)
        with col4:
            aba_dest = st.selectbox("Aba da planilha a preencher", options=xls_dest.sheet_names)

        df_base = carregar_excel(arquivo_base, aba_base, int(header_base) - 1)
        df_destino = carregar_excel(arquivo_destino, aba_dest, int(header_dest) - 1)

        st.divider()
        st.subheader("3. Mapeamento das colunas")

        c1, c2 = st.columns(2)
        with c1:
            coluna_texto_base = st.selectbox(
                "Coluna da base usada para comparação semântica",
                options=df_base.columns.tolist(),
                index=df_base.columns.tolist().index("DESCRIÇÃO") if "DESCRIÇÃO" in df_base.columns else 0,
            )
        with c2:
            coluna_busca_destino = st.selectbox(
                "Coluna da planilha de destino usada como busca",
                options=df_destino.columns.tolist(),
                index=df_destino.columns.tolist().index("G") if "G" in df_destino.columns else 0,
            )

        st.markdown("### Colunas da base que deseja obter")
        colunas_base_retorno = st.multiselect(
            "Selecione as colunas da base",
            options=df_base.columns.tolist(),
            default=[c for c in ["R$ CAPEX/NOVO", "CÓDIGO", "FONTE", "UNID", "SEM BDI"] if c in df_base.columns],
        )

        st.markdown("### Colunas da planilha de destino que receberão os dados")
        st.caption("A ordem deve corresponder exatamente à ordem escolhida na base.")

        colunas_destino_preencher = []
        for i, col_base in enumerate(colunas_base_retorno, start=1):
            escolha = st.selectbox(
                f"Destino para '{col_base}'",
                options=df_destino.columns.tolist(),
                key=f"dest_{i}_{col_base}",
            )
            colunas_destino_preencher.append(escolha)

        if len(colunas_base_retorno) != len(colunas_destino_preencher):
            st.error("A quantidade de colunas da base e de destino precisa ser a mesma.")
        elif len(set(colunas_destino_preencher)) != len(colunas_destino_preencher):
            st.error("Você repetiu colunas de destino. Cada coluna de destino deve ser usada apenas uma vez.")
        else:
            st.divider()
            st.subheader("4. Prévia")
            p1, p2 = st.columns(2)
            with p1:
                st.write("Base")
                st.dataframe(df_base.head(10), use_container_width=True)
            with p2:
                st.write("Planilha a preencher")
                st.dataframe(df_destino.head(10), use_container_width=True)

            if st.button("Processar preenchimento", type="primary"):
                resultado = processar_preenchimento(
                    df_base=df_base,
                    df_destino=df_destino,
                    coluna_busca_destino=coluna_busca_destino,
                    colunas_base_retorno=colunas_base_retorno,
                    colunas_destino_preencher=colunas_destino_preencher,
                    coluna_texto_base=coluna_texto_base,
                    score_minimo=score_minimo,
                )

                st.success("Processamento concluído.")
                st.dataframe(resultado.head(50), use_container_width=True)

                excel_bytes = dataframe_para_excel_bytes(resultado, nome_aba=aba_dest)
                st.download_button(
                    label="Baixar planilha preenchida",
                    data=excel_bytes,
                    file_name="planilha_preenchida.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
else:
    st.info("Importe os dois arquivos para habilitar o mapeamento e o preenchimento automático.")

st.divider()
st.markdown(
    """
### Preencha passo a passo para habilitar a próxima etapa
```
"""
)


