from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from unidecode import unidecode

from conceptual_reranker import decide_best_candidate_with_llm
from llm_service import LLMDecisionConfig
from taxonomy_catalog import (
    CLASS_ALIASES,
    DOMAIN_BY_FAMILY,
    FAMILY_ALIASES,
    FAMILY_NEGATIVE_CONTEXT,
    FAMILY_REQUIRED_ANY,
    MATERIAL_ALIASES,
    PRIORITY_ORDER,
    SERVICE_ADMIN_MARKERS,
    SUBFAMILY_ALIASES,
)

logger = logging.getLogger("site_orcamento_ia.motor")

MOTOR_BUSCA = "tfidf-char-ngrams"
PESO_SEMANTICO = 0.55
PESO_FUZZY = 0.20
PESO_REGRAS = 0.25
TOP_K_PADRAO = 50
TOP_K_RERANK_TECNICO = 200
TOP_K_RECALL_AMPLIADO = 800
SCORE_MINIMO_CONFIAVEL = 0.42
GAP_MINIMO_CONFIAVEL = 0.035


def normalizar_texto(texto: str) -> str:
    if texto is None or pd.isna(texto):
        return ""

    texto = str(texto).strip().lower()
    texto = (
        texto.replace("ø", " diametro ")
        .replace("∅", " diametro ")
        .replace("º", " graus ")
        .replace("°", " graus ")
        .replace("”", '"')
        .replace("“", '"')
        .replace("’", "'")
        .replace("″", '"')
    )
    texto = unidecode(texto)
    substituicoes = {
        "fck": "resistencia caracteristica",
        "mpa": "megapascal",
        "astm a53/a53m": "astm a53",
        "astm a 53": "astm a53",
        "gr. b-s": "gr b",
        "gr. b": "gr b",
        "sch. 80": "sch 80",
        "sch. 40": "sch 40",
        "c/ rosca npt": "com rosca npt",
        "sc": "sem costura",
        "cc": "com costura",
        "pc": "ponta chanfrada",
        "concreto armado": "concreto estrutural armado",
        "concreto simples": "concreto sem armadura",
        "divisoria": "parede divisoria vedacao compartimentacao interna",
        "drywall": "parede leve em gesso acartonado",
        "alvenaria": "parede de alvenaria vedacao",
        "parede": "vedacao parede fechamento",
        "armacao": "armadura aco",
        "forma": "forma madeira compensado",
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
        "cuba inox para pia": "cuba de embutir em aco inox para pia",
        "cuba inox": "cuba em aco inox",
        "sifao metalico": "sifao metalico acessorio esgoto",
    }
    for de, para in sorted(substituicoes.items(), key=lambda item: len(item[0]), reverse=True):
        texto = re.sub(rf"(?<!\w){re.escape(de)}(?!\w)", para, texto)
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_nome_coluna(nome: str) -> str:
    nome = unidecode(str(nome or "")).strip().lower()
    nome = re.sub(r"[^\w\s]", "", nome)
    return re.sub(r"\s+", " ", nome).strip()


def resolver_colunas_disponiveis(
    colunas_disponiveis: List[str],
    colunas_solicitadas: List[str],
) -> Tuple[List[str], List[str]]:
    mapa_normalizado: Dict[str, str] = {}
    for coluna in colunas_disponiveis:
        coluna_str = str(coluna)
        chave = normalizar_nome_coluna(coluna_str)
        if chave and chave not in mapa_normalizado:
            mapa_normalizado[chave] = coluna_str

    resolvidas: List[str] = []
    faltando: List[str] = []
    for coluna in colunas_solicitadas:
        if coluna in colunas_disponiveis:
            resolvidas.append(coluna)
            continue
        coluna_resolvida = mapa_normalizado.get(normalizar_nome_coluna(coluna))
        if coluna_resolvida:
            resolvidas.append(coluna_resolvida)
            continue
        faltando.append(coluna)
    return resolvidas, faltando


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
        ("disjuntor", "disjuntor"),
        ("minidisjuntor", "minidisjuntor"),
        ("contator", "contator"),
        ("rele", "rele"),
        ("cuba", "cuba"),
        ("sifao", "sifao"),
        ("tomada", "tomada"),
        ("interruptor", "interruptor"),
        ("eletroduto", "eletroduto"),
        ("condulete", "condulete"),
    ]

    for termo_busca, termo_desc in pares:
        if termo_busca in busca_norm and termo_desc in descricao_norm:
            score += 0.08

    if "divisoria" in busca_norm and any(x in descricao_norm for x in ["drywall", "alvenaria", "parede", "vedacao"]):
        score += 0.20

    if "concreto" in busca_norm and "megapascal" in busca_norm:
        if "concreto" in descricao_norm and any(x in descricao_norm for x in ["megapascal", "resistencia caracteristica"]):
            score += 0.20

    if "disjuntor" in busca_norm:
        if "disjuntor" in descricao_norm or "minidisjuntor" in descricao_norm:
            score += 0.24
        if any(x in busca_norm and x in descricao_norm for x in ["curva b", "curva c", "curva d"]):
            score += 0.12
        if any(x in busca_norm and x in descricao_norm for x in ["1p", "2p", "3p", "4p", "monopolar", "bipolar", "tripolar", "tetrapolar"]):
            score += 0.12

    if "eletroduto" in busca_norm and "eletroduto" in descricao_norm:
        score += 0.12

    if "cuba" in busca_norm and "cuba" in descricao_norm:
        score += 0.22
    if "cuba" in busca_norm and "sifao" in descricao_norm:
        score -= 0.20
    if "curva 45" in busca_norm and "curva 45" in descricao_norm:
        score += 0.18
    if "curva 45" in busca_norm and "curva 90" in descricao_norm:
        score -= 0.18
    if "aco" in busca_norm and "aco" in descricao_norm:
        score += 0.12
    if "solda" in busca_norm and "solda" in descricao_norm:
        score += 0.10
    if "bisel" in busca_norm and "bisel" in descricao_norm:
        score += 0.10

    return min(score, 1.0)


def _normalizar_texto_tecnico(texto: str) -> str:
    texto = str(texto or "").lower()
    texto = (
        texto.replace("ø", " diametro ")
        .replace("∅", " diametro ")
        .replace("º", " graus ")
        .replace("°", " graus ")
        .replace("”", '"')
        .replace("“", '"')
        .replace("’", "'")
        .replace("″", '"')
    )
    texto = texto.replace("mm²", "mm2").replace("m²", "m2").replace("cm²", "cm2")
    texto = texto.replace("ø", " diametro ")
    texto = unidecode(texto)
    texto = re.sub(r"(\d),(\d)", r"\1.\2", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _normalizar_valor_tecnico(valor: str) -> str:
    try:
        numero = float(str(valor).replace(",", "."))
        if numero.is_integer():
            return str(int(numero))
        return f"{numero:.3f}".rstrip("0").rstrip(".")
    except ValueError:
        return str(valor).strip().lower()


def _extrair_valores(pattern: str, texto: str) -> Set[str]:
    valores: Set[str] = set()
    for match in re.finditer(pattern, texto, flags=re.IGNORECASE):
        valores.add(_normalizar_valor_tecnico(match.group(1)))
    return valores


def _extrair_tokens_lista(texto: str, termos: Dict[str, List[str]]) -> Set[str]:
    encontrados: Set[str] = set()
    for chave, aliases in termos.items():
        for alias in aliases:
            if re.search(rf"(^|[^\w]){re.escape(alias)}($|[^\w])", texto):
                encontrados.add(chave)
                break
    return encontrados


def _extrair_normas(texto: str) -> Set[str]:
    normas: Set[str] = set()
    patterns = {
        "astm_a53": r"astm\s*a\s*53(?:/a53m)?",
        "astm_a105": r"astm\s*a\s*105(?:/a105m)?",
        "asme_b16_5": r"asme\s*b\s*16\.?5",
        "asme_b16_11": r"asme\s*b\s*16\.?11",
        "asme_b16_21": r"asme\s*b\s*16\.?21",
        "asme_b36_10": r"asme\s*b\s*36\.?10m?",
        "abnt_nbr_5598": r"abnt\s*nbr\s*5598",
    }
    for key, pattern in patterns.items():
        if re.search(pattern, texto, flags=re.IGNORECASE):
            normas.add(key)
    return normas


def _extrair_polegadas(texto: str) -> Set[str]:
    polegadas: Set[str] = set()
    spans_consumidos: List[Tuple[int, int]] = []

    def ja_consumido(inicio: int, fim: int) -> bool:
        return any(inicio < fim_existente and fim > inicio_existente for inicio_existente, fim_existente in spans_consumidos)

    for match in re.finditer(
        r"(?:(?:diametro|dn)\s*)?(\d+)[\.\s]+(\d+/\d{1,2})\s*(?:\"|pol|polegada|polegadas)",
        texto,
        flags=re.IGNORECASE,
    ):
        spans_consumidos.append(match.span())
        polegadas.add(f"{match.group(1)}.{match.group(2)}")

    for match in re.finditer(
        r"diametro\s*(\d+)[\.\s]+(\d+/\d{1,2})(?!\s*/)",
        texto,
        flags=re.IGNORECASE,
    ):
        if ja_consumido(*match.span()):
            continue
        spans_consumidos.append(match.span())
        polegadas.add(f"{match.group(1)}.{match.group(2)}")

    for match in re.finditer(
        r"(?:(?:diametro|dn)\s*)?(\d+/\d{1,2})\s*(?:\"|pol|polegada|polegadas)",
        texto,
        flags=re.IGNORECASE,
    ):
        if ja_consumido(*match.span()):
            continue
        spans_consumidos.append(match.span())
        polegadas.add(match.group(1))

    for match in re.finditer(
        r"diametro\s*(\d+/\d{1,2})(?!\s*/)",
        texto,
        flags=re.IGNORECASE,
    ):
        if ja_consumido(*match.span()):
            continue
        spans_consumidos.append(match.span())
        polegadas.add(match.group(1))

    for match in re.finditer(
        r"(?<![\d./-])(\d+(?:\.\d+)?)\s*(?:\"|pol|polegada|polegadas)",
        texto,
        flags=re.IGNORECASE,
    ):
        if ja_consumido(*match.span()):
            continue
        spans_consumidos.append(match.span())
        polegadas.add(_normalizar_valor_tecnico(match.group(1)))
    return polegadas


def _extrair_diametro_nominal(texto: str) -> Set[str]:
    valores: Set[str] = set()
    for inteiro, fracao in re.findall(r"diametro\s*(\d+)[\.\s]+(\d+/\d+)", texto, flags=re.IGNORECASE):
        valores.add(f"{inteiro}.{fracao}")
    for valor in re.findall(r"diametro\s*(\d+/\d+)", texto, flags=re.IGNORECASE):
        valores.add(valor)
    for valor in re.findall(r"diametro\s*(\d+(?:\.\d+)?)(?!\s*[/.]\s*\d)", texto, flags=re.IGNORECASE):
        valores.add(_normalizar_valor_tecnico(valor))
    return valores


def inferir_familia_principal(
    texto: str,
    familias_detectadas: Set[str],
    aliases_familias: Dict[str, List[str]],
) -> Optional[str]:
    if not familias_detectadas:
        return None

    pontuacoes: List[Tuple[int, int, str]] = []
    for familia in familias_detectadas:
        aliases = aliases_familias.get(familia, [])
        negativos = FAMILY_NEGATIVE_CONTEXT.get(familia, [])
        requeridos = FAMILY_REQUIRED_ANY.get(familia, [])

        if any(negativo in texto for negativo in negativos):
            continue
        if requeridos and not any(token in texto for token in requeridos):
            continue

        score = sum(
            1 for alias in aliases if re.search(rf"(^|[^\w]){re.escape(alias)}($|[^\w])", texto)
        )
        if familia == "painel" and any(token in texto for token in ("qgbt", "qdc", "qdl", "ccm", "painel eletrico")):
            score += 2
        if familia == "terminal" and any(token in texto for token in ("circuitos terminais", "circuito terminal")):
            score -= 3

        prioridade_idx = PRIORITY_ORDER.get(familia, 999)
        pontuacoes.append((score, -prioridade_idx, familia))

    if not pontuacoes:
        return None
    pontuacoes.sort(reverse=True)
    return pontuacoes[0][2]

def inferir_subfamilias(texto: str, familia_principal: Optional[str]) -> Set[str]:
    if not familia_principal:
        return set()
    return _extrair_tokens_lista(texto, SUBFAMILY_ALIASES.get(familia_principal, {}))

def inferir_natureza_item(
    texto: str,
    familias_detectadas: Set[str],
    familia_principal: Optional[str],
) -> str:
    if any(marcador in texto for marcador in SERVICE_ADMIN_MARKERS):
        return "servico_administrativo"

    marcadores_servico = [
        "fornecimento e instalacao",
        "execucao",
        "aplicacao",
        "assentamento",
        "montagem",
        "demolicao",
        "escavacao",
        "aterro",
        "pintura",
    ]
    if any(marcador in texto for marcador in marcadores_servico) and familia_principal not in {
        "cabo",
        "tubo",
        "eletroduto",
        "disjuntor",
        "contator",
        "rele",
        "interruptor",
        "tomada",
        "quadro",
        "painel",
    }:
        return "servico"

    marcadores_composto = [
        "conjunto",
        "kit",
        "cabine",
        "coluna de hidrante",
        "hidrante tipo coluna",
        "guarda-corpo",
        "guarda corpo",
        "painel",
        "quadro",
        "com conexoes",
        "inclusive conexoes",
        "com valvulas",
        "montado",
        "completo",
    ]
    if any(marcador in texto for marcador in marcadores_composto):
        return "item_composto"

    if familia_principal in {"te", "curva", "reducao", "luva", "flange", "terminal", "valvula", "grampo", "bucha", "tomada", "interruptor", "rele", "contator", "sifao"}:
        return "acessorio"

    if familia_principal in {"hidrante", "vaso_sanitario", "cuba"}:
        return "conjunto"

    if len(familias_detectadas) >= 3:
        return "item_composto"
    return "item_simples"

def extrair_atributos_tecnicos(texto: str) -> Dict[str, Set[str]]:
    tecnico = _normalizar_texto_tecnico(texto)

    atributos: Dict[str, Set[str]] = {
        "bitola_mm2": _extrair_valores(r"(?<!\d)(\d+(?:\.\d+)?)\s*mm\s*2\b", tecnico),
        "dn": _extrair_valores(r"\bdn\s*(\d+(?:\.\d+)?)\b", tecnico),
        "diametro_mm": _extrair_valores(r"(?:diametro\s*)?(\d+(?:\.\d+)?)\s*mm\b(?!\s*2)", tecnico),
        "diametro_cm": _extrair_valores(r"(?:diametro\s*)?(\d+(?:\.\d+)?)\s*cm\b(?!\s*2)", tecnico),
        "polegada": _extrair_polegadas(tecnico),
        "diametro_nominal": _extrair_diametro_nominal(tecnico),
        "mpa": _extrair_valores(r"\b(?:fck\s*)?(\d+(?:\.\d+)?)\s*(?:mpa|megapascal)\b", tecnico),
        "tensao_v": _extrair_valores(r"\b(\d+(?:\.\d+)?)\s*(?:v|volt|volts)\b", tecnico),
        "tensao_kv": _extrair_valores(r"\b(\d+(?:\.\d+)?)\s*kv\b", tecnico),
        "corrente_a": _extrair_valores(r"\b(\d+(?:\.\d+)?)\s*a\b", tecnico),
        "interrupcao_ka": _extrair_valores(r"\b(\d+(?:\.\d+)?)\s*ka\b", tecnico),
        "angulo_graus": _extrair_valores(r"\b(?:curva|cotovelo)?\s*(45|90)\s*(?:graus|grau)?\b", tecnico),
        "normas": _extrair_normas(tecnico),
        "polos": _extrair_tokens_lista(
            tecnico,
            {
                "1p": ["1p", "1 p", "1 polo", "monopolar"],
                "2p": ["2p", "2 p", "2 polos", "bipolar"],
                "3p": ["3p", "3 p", "3 polos", "tripolar"],
                "4p": ["4p", "4 p", "4 polos", "tetrapolar"],
            },
        ),
        "curva_disparo": _extrair_tokens_lista(
            tecnico,
            {"b": ["curva b"], "c": ["curva c"], "d": ["curva d"]},
        ),
        "materiais": _extrair_tokens_lista(tecnico, MATERIAL_ALIASES),
        "classes": _extrair_tokens_lista(tecnico, CLASS_ALIASES),
    }

    familias_detectadas: Set[str] = set()
    for familia, aliases in FAMILY_ALIASES.items():
        encontrou_alias = any(
            re.search(rf"(^|[^\w]){re.escape(alias)}($|[^\w])", tecnico)
            for alias in aliases
        )
        if not encontrou_alias:
            continue
        negativos = FAMILY_NEGATIVE_CONTEXT.get(familia, [])
        if any(negativo in tecnico for negativo in negativos):
            continue
        requeridos = FAMILY_REQUIRED_ANY.get(familia, [])
        if requeridos and not any(token in tecnico for token in requeridos):
            continue
        familias_detectadas.add(familia)

    # Recover shorthand-heavy electrical protection descriptions even when the
    # word "disjuntor" is omitted from the text.
    if "disjuntor" not in familias_detectadas:
        if atributos["polos"] and atributos["corrente_a"] and (
            "termomagnetico" in atributos["classes"] or "caixa_moldada" in atributos["classes"] or "tmf" in tecnico
        ):
            familias_detectadas.add("disjuntor")

    atributos["familias"] = familias_detectadas
    familia_principal = inferir_familia_principal(tecnico, familias_detectadas, FAMILY_ALIASES) if familias_detectadas else None
    if familia_principal:
        atributos["familia_principal"] = {familia_principal}
        atributos["macrodominio"] = {DOMAIN_BY_FAMILY.get(familia_principal, "geral")}
        subfamilias = inferir_subfamilias(tecnico, familia_principal)
        if subfamilias:
            atributos["subfamilias"] = subfamilias
        atributos["natureza_item"] = {inferir_natureza_item(tecnico, familias_detectadas, familia_principal)}
    elif any(marcador in tecnico for marcador in SERVICE_ADMIN_MARKERS):
        atributos["natureza_item"] = {"servico_administrativo"}
        atributos["macrodominio"] = {"canteiro_e_administracao"}

    return {chave: valores for chave, valores in atributos.items() if valores}

def detectar_coincidencias_tecnicas(atributos_busca: Dict[str, Set[str]], atributos_base: Dict[str, Set[str]]) -> List[str]:
    coincidencias: List[str] = []
    for chave, valores_busca in atributos_busca.items():
        valores_base = atributos_base.get(chave)
        if valores_busca and valores_base and (valores_busca & valores_base):
            coincidencias.append(chave)
    return coincidencias


def detectar_conflitos_tecnicos(atributos_busca: Dict[str, Set[str]], atributos_base: Dict[str, Set[str]]) -> List[str]:
    conflitos: List[str] = []
    for chave in {
        "bitola_mm2",
        "dn",
        "polegada",
        "diametro_nominal",
        "diametro_mm",
        "diametro_cm",
        "materiais",
        "macrodominio",
        "familia_principal",
        "subfamilias",
        "classes",
        "normas",
        "tensao_v",
        "tensao_kv",
        "corrente_a",
        "interrupcao_ka",
        "angulo_graus",
        "polos",
        "curva_disparo",
    }:
        valores_busca = atributos_busca.get(chave)
        if not valores_busca:
            continue
        valores_base = atributos_base.get(chave)
        if chave == "materiais" and not valores_base:
            conflitos.append(chave)
            continue
        if valores_base and not (valores_busca & valores_base):
            conflitos.append(chave)
    return conflitos


def avaliar_confianca_match(
    score_final: float,
    score_base: float,
    score_segundo_colocado: float,
    atributos_busca: Dict[str, Set[str]],
    coincidencias: List[str],
    conflitos: List[str],
    score_minimo_usuario: float,
) -> dict:
    score_minimo_aplicado = max(score_minimo_usuario, SCORE_MINIMO_CONFIAVEL)
    gap = max(0.0, score_final - score_segundo_colocado)
    coincidencias_set = set(coincidencias)
    conflitos_set = set(conflitos)
    motivos: List[str] = []

    chaves_estruturais = {
        "bitola_mm2",
        "dn",
        "polegada",
        "diametro_mm",
        "diametro_cm",
        "familia_principal",
        "macrodominio",
        "subfamilias",
        "classes",
        "normas",
        "corrente_a",
        "polos",
        "curva_disparo",
        "tensao_v",
        "tensao_kv",
    }
    tem_atributo_estrutural = any(atributos_busca.get(chave) for chave in chaves_estruturais)
    tem_match_estrutural = any(chave in coincidencias_set for chave in chaves_estruturais)

    if score_final < score_minimo_aplicado:
        motivos.append("score_final_abaixo_do_limiar")
    if score_base < 0.22 and score_final < 0.60:
        motivos.append("base_textual_fraca")
    if gap < GAP_MINIMO_CONFIAVEL and score_final < 0.70:
        motivos.append("gap_pequeno_para_segundo_colocado")
    if tem_atributo_estrutural and not tem_match_estrutural:
        motivos.append("sem_match_estrutural")
    if "familia_principal" in conflitos_set and score_final < 0.80:
        motivos.append("conflito_de_familia_principal")
    if "macrodominio" in conflitos_set and score_final < 0.76:
        motivos.append("conflito_de_macrodominio")
    if "normas" in conflitos_set and score_final < 0.82:
        motivos.append("conflito_de_norma_tecnica")
    if "polos" in conflitos_set and score_final < 0.85:
        motivos.append("conflito_de_polos")
    if "corrente_a" in conflitos_set and score_final < 0.85:
        motivos.append("conflito_de_corrente")
    if "tensao_v" in conflitos_set or "tensao_kv" in conflitos_set:
        if score_final < 0.90:
            motivos.append("faixa_tensao_incompativel")
    if "bitola_mm2" in conflitos_set and score_final < 0.78:
        motivos.append("conflito_de_bitola")
    if "diametro_nominal" in conflitos_set and score_final < 0.90:
        motivos.append("conflito_de_diametro_nominal")
    if "polegada" in conflitos_set and score_final < 0.82:
        motivos.append("conflito_de_polegada")
    if "angulo_graus" in conflitos_set and score_final < 0.90:
        motivos.append("conflito_de_angulo")
    if "dn" in conflitos_set and score_final < 0.82:
        motivos.append("conflito_de_dn")
    if ("diametro_mm" in conflitos_set or "diametro_cm" in conflitos_set) and score_final < 0.82:
        motivos.append("conflito_de_diametro")

    return {
        "aceito": not motivos,
        "motivos": motivos,
        "gap_para_segundo": round(gap, 4),
        "score_minimo_aplicado": round(score_minimo_aplicado, 4),
    }


def _score_bonus_tecnico(atributos_busca: Dict[str, Set[str]], atributos_base: Dict[str, Set[str]]) -> tuple[float, List[str], List[str]]:
    coincidencias = detectar_coincidencias_tecnicas(atributos_busca, atributos_base)
    conflitos = detectar_conflitos_tecnicos(atributos_busca, atributos_base)
    bonus = 0.0
    pesos = {
        "familia_principal": 0.14,
        "macrodominio": 0.10,
        "subfamilias": 0.08,
        "bitola_mm2": 0.08,
        "dn": 0.08,
        "polegada": 0.08,
        "diametro_nominal": 0.12,
        "diametro_mm": 0.08,
        "diametro_cm": 0.08,
        "classes": 0.06,
        "normas": 0.12,
        "materiais": 0.05,
        "tensao_v": 0.07,
        "tensao_kv": 0.07,
        "corrente_a": 0.07,
        "interrupcao_ka": 0.07,
        "angulo_graus": 0.10,
        "polos": 0.07,
        "curva_disparo": 0.05,
        "natureza_item": 0.06,
    }
    for chave in coincidencias:
        bonus += pesos.get(chave, 0.03)
    pesos_conflito = {
        "familia_principal": 0.22,
        "macrodominio": 0.16,
        "subfamilias": 0.14,
        "bitola_mm2": 0.18,
        "dn": 0.18,
        "polegada": 0.45,
        "diametro_nominal": 0.34,
        "diametro_mm": 0.20,
        "diametro_cm": 0.20,
        "classes": 0.12,
        "normas": 0.24,
        "materiais": 0.10,
        "tensao_v": 0.16,
        "tensao_kv": 0.16,
        "corrente_a": 0.16,
        "interrupcao_ka": 0.16,
        "angulo_graus": 0.34,
        "polos": 0.18,
        "curva_disparo": 0.14,
        "natureza_item": 0.12,
    }
    for chave in conflitos:
        bonus -= pesos_conflito.get(chave, 0.06)
    return bonus, coincidencias, conflitos


def _ajuste_compatibilidade_de_familia(
    busca_norm: str,
    texto_base_norm: str,
    atributos_busca: Dict[str, Set[str]],
    atributos_base: Dict[str, Set[str]],
) -> float:
    familia_busca = next(iter(atributos_busca.get("familia_principal", set())), None)
    familia_base = next(iter(atributos_base.get("familia_principal", set())), None)
    if not familia_busca or not familia_base:
        return 0.0

    bonus = 0.0
    if familia_busca == familia_base:
        bonus += 0.10
    else:
        bonus -= 0.06

    penalidades_fortes = {
        ("te", "curva"): 0.34,
        ("curva", "te"): 0.22,
        ("flange", "junta"): 0.34,
        ("tubo", "luva"): 0.30,
        ("tubo", "curva"): 0.26,
        ("tubo", "te"): 0.24,
        ("tubo", "sensor"): 0.36,
        ("tubo", "eletroduto"): 0.40,
        ("filtro", "eletroduto"): 0.34,
        ("filtro", "tubo"): 0.10,
        ("disjuntor", "tomada"): 0.32,
        ("disjuntor", "cabo"): 0.28,
        ("disjuntor", "eletroduto"): 0.28,
    }
    bonus -= penalidades_fortes.get((familia_busca, familia_base), 0.0)

    if familia_busca == "te":
        if "te" in busca_norm and "te" in texto_base_norm:
            bonus += 0.20
        if any(token in texto_base_norm for token in ("cotovelo", "curva 90", "joelho")):
            bonus -= 0.25

    if familia_busca == "flange":
        if "cego" in busca_norm and "cego" in texto_base_norm:
            bonus += 0.18
        if "junta" in texto_base_norm:
            bonus -= 0.30

    if familia_busca == "tubo":
        if "tubo" in texto_base_norm:
            bonus += 0.14
        if any(token in texto_base_norm for token in ("luva", "curva", "te ", "te de", "sensor", "eletroduto")):
            bonus -= 0.16

    if familia_busca == "filtro":
        if any(token in texto_base_norm for token in ("filtro", "coalescente", "separador")):
            bonus += 0.18
        if any(token in texto_base_norm for token in ("condulete", "sarjeta", "eletroduto")):
            bonus -= 0.24

    if familia_busca == "disjuntor":
        if "disjuntor" in texto_base_norm:
            bonus += 0.20

    return bonus


def _selecionar_pool_por_familia(
    similaridades,
    attrs_base_lista: List[Dict[str, Set[str]]],
    atributos_busca: Dict[str, Set[str]],
    limite_final: int,
) -> List[int]:
    total_base = len(attrs_base_lista)
    if total_base == 0:
        return []

    limite_recall = min(total_base, max(TOP_K_RECALL_AMPLIADO, limite_final * 6, TOP_K_RERANK_TECNICO * 2))
    ordem_recall = similaridades.argsort()[::-1][:limite_recall]

    familia_alvo = next(iter(atributos_busca.get("familia_principal", set())), None)
    dominio_alvo = next(iter(atributos_busca.get("macrodominio", set())), None)

    if not familia_alvo and not dominio_alvo:
        return [int(idx) for idx in ordem_recall[:limite_final]]

    mesmo_item: List[int] = []
    mesmo_dominio: List[int] = []
    outros: List[int] = []

    for idx in ordem_recall:
        atributos_base = attrs_base_lista[int(idx)]
        familia_base = next(iter(atributos_base.get("familia_principal", set())), None)
        dominio_base = next(iter(atributos_base.get("macrodominio", set())), None)

        if familia_alvo and familia_base == familia_alvo:
            mesmo_item.append(int(idx))
        elif dominio_alvo and dominio_base == dominio_alvo:
            mesmo_dominio.append(int(idx))
        else:
            outros.append(int(idx))

    pool: List[int] = []
    vistos: Set[int] = set()

    def adicionar(indices: List[int], limite: int) -> None:
        for idx in indices[:limite]:
            if idx in vistos:
                continue
            vistos.add(idx)
            pool.append(idx)

    cotas_familia = {
        "disjuntor": 160,
        "tubo": 160,
        "te": 140,
        "curva": 140,
        "flange": 140,
        "filtro": 120,
        "valvula": 120,
    }
    cota_mesma_familia = min(limite_final, cotas_familia.get(familia_alvo or "", 120))
    cota_mesmo_dominio = min(max(40, limite_final // 3), limite_final)

    adicionar(mesmo_item, cota_mesma_familia)
    adicionar(mesmo_dominio, cota_mesmo_dominio)
    adicionar([int(idx) for idx in ordem_recall], limite_final)

    return pool[:limite_final]


def preparar_base_para_busca(df_base: pd.DataFrame, coluna_texto_base: str):
    df_base_proc = df_base.copy()
    df_base_proc[coluna_texto_base] = df_base_proc[coluna_texto_base].fillna("").astype(str)
    df_base_proc["__texto_base_original__"] = df_base_proc[coluna_texto_base]
    df_base_proc["__texto_base_norm__"] = df_base_proc[coluna_texto_base].map(normalizar_texto)
    df_base_proc["__atributos_tecnicos__"] = df_base_proc[coluna_texto_base].map(extrair_atributos_tecnicos)
    df_base_proc["__familia_principal__"] = df_base_proc["__atributos_tecnicos__"].map(
        lambda attrs: next(iter(attrs.get("familia_principal", set())), None)
    )
    df_base_proc["__macrodominio__"] = df_base_proc["__atributos_tecnicos__"].map(
        lambda attrs: next(iter(attrs.get("macrodominio", set())), None)
    )

    vetorizador = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=False,
    )
    matriz_base = vetorizador.fit_transform(df_base_proc["__texto_base_norm__"].tolist())
    return df_base_proc, vetorizador, matriz_base


def buscar_melhor_item_em_lote(
    buscas_norm_unicas: List[str],
    buscas_originais_por_norm: Dict[str, str],
    df_base_proc: pd.DataFrame,
    vetorizador,
    matriz_base,
    top_k_candidatos: int,
    score_minimo_usuario: float,
    *,
    llm_config: Optional[LLMDecisionConfig] = None,
) -> Dict[str, Optional[Tuple[int, dict]]]:
    if not buscas_norm_unicas:
        return {}

    llm_config = llm_config or LLMDecisionConfig()
    matriz_buscas = vetorizador.transform(buscas_norm_unicas)
    k = min(max(top_k_candidatos, TOP_K_PADRAO, TOP_K_RERANK_TECNICO), len(df_base_proc))
    resultados: Dict[str, Optional[Tuple[int, dict]]] = {}
    textos_base_originais = df_base_proc["__texto_base_original__"].tolist()
    textos_base_norm = df_base_proc["__texto_base_norm__"].tolist()
    attrs_base_lista = df_base_proc["__atributos_tecnicos__"].tolist()

    for pos_busca, busca_norm in enumerate(buscas_norm_unicas):
        similaridades = linear_kernel(matriz_buscas[pos_busca], matriz_base).ravel()
        if similaridades.size == 0:
            resultados[busca_norm] = None
            continue

        busca_original = buscas_originais_por_norm.get(busca_norm, busca_norm)
        atributos_busca = extrair_atributos_tecnicos(busca_original)
        melhores_indices = _selecionar_pool_por_familia(
            similaridades=similaridades,
            attrs_base_lista=attrs_base_lista,
            atributos_busca=atributos_busca,
            limite_final=k,
        )

        candidatos_ranqueados: List[dict] = []
        for idx_base in melhores_indices:
            texto_base_norm = textos_base_norm[int(idx_base)]
            atributos_base = attrs_base_lista[int(idx_base)]
            texto_base_original = textos_base_originais[int(idx_base)]
            score_sem = float(similaridades[int(idx_base)])
            score_fuzzy = fuzz.token_set_ratio(busca_norm, texto_base_norm) / 100.0
            score_reg = score_regras(busca_norm, texto_base_norm)
            bonus_tecnico, coincidencias, conflitos = _score_bonus_tecnico(atributos_busca, atributos_base)
            bonus_familia = _ajuste_compatibilidade_de_familia(
                busca_norm=busca_norm,
                texto_base_norm=texto_base_norm,
                atributos_busca=atributos_busca,
                atributos_base=atributos_base,
            )
            score_base = (PESO_SEMANTICO * score_sem) + (PESO_FUZZY * score_fuzzy) + (PESO_REGRAS * score_reg)
            score_final = max(
                0.0,
                min(
                    1.0,
                    score_base + bonus_tecnico + bonus_familia,
                ),
            )
            candidatos_ranqueados.append(
                {
                    "idx_base": int(idx_base),
                    "descricao": str(texto_base_original),
                    "score_final": round(score_final, 4),
                    "score_base": round(score_base, 4),
                    "score_semantico": round(score_sem, 4),
                    "score_fuzzy": round(score_fuzzy, 4),
                    "score_regras": round(score_reg, 4),
                    "score_familia": round(bonus_familia, 4),
                    "familia_principal": sorted(atributos_base.get("familia_principal", set())),
                    "subfamilias": sorted(atributos_base.get("subfamilias", set())),
                    "natureza_item": sorted(atributos_base.get("natureza_item", set())),
                    "coincidencias_tecnicas": coincidencias,
                    "conflitos_tecnicos": conflitos,
                }
            )

        candidatos_ranqueados.sort(key=lambda item: (-item["score_final"], item["idx_base"]))
        if not candidatos_ranqueados:
            resultados[busca_norm] = None
            continue

        top_candidatos = candidatos_ranqueados[: min(len(candidatos_ranqueados), max(5, llm_config.max_candidates, top_k_candidatos))]
        melhor = dict(top_candidatos[0])
        melhor_idx = int(melhor["idx_base"])
        segundo_score = top_candidatos[1]["score_final"] if len(top_candidatos) > 1 else 0.0
        gap = max(0.0, float(melhor["score_final"]) - float(segundo_score))

        score_llm = None
        llm_usada = False
        llm_motivo = ""

        if (
            llm_config.enabled
            and len(top_candidatos) >= 2
            and llm_config.decision_min_top_score <= float(melhor["score_final"]) <= llm_config.decision_max_top_score
            and gap <= llm_config.decision_max_gap
        ):
            decisao_semantica = decide_best_candidate_with_llm(
                busca_original,
                top_candidatos[: llm_config.max_candidates],
                metadata={
                    "score_atual": float(melhor["score_final"]),
                    "score_segundo_colocado": float(segundo_score),
                    "score_minimo_usuario": float(score_minimo_usuario),
                    "matching_engine": MOTOR_BUSCA,
                },
                config=llm_config,
            )
            llm_usada = bool(decisao_semantica.get("semantic_decision_used"))
            llm_motivo = decisao_semantica.get("short_reason") or decisao_semantica.get("fallback_reason") or ""
            score_llm = decisao_semantica.get("semantic_confidence") or decisao_semantica.get("confidence")
            if llm_usada:
                indice_semantico = int(decisao_semantica["chosen_index"]) - 1
                if 0 <= indice_semantico < len(top_candidatos):
                    melhor = dict(top_candidatos[indice_semantico])
                    melhor_idx = int(melhor["idx_base"])

        melhor["score_llm"] = score_llm
        melhor["llm_usada"] = llm_usada
        melhor["llm_motivo"] = llm_motivo
        confianca = avaliar_confianca_match(
            score_final=float(melhor["score_final"]),
            score_base=float(melhor.get("score_base", melhor["score_final"])),
            score_segundo_colocado=float(segundo_score),
            atributos_busca=atributos_busca,
            coincidencias=melhor.get("coincidencias_tecnicas", []),
            conflitos=melhor.get("conflitos_tecnicos", []),
            score_minimo_usuario=score_minimo_usuario,
        )
        melhor["gap_para_segundo"] = confianca["gap_para_segundo"]
        melhor["score_minimo_aplicado"] = confianca["score_minimo_aplicado"]
        melhor["aceito"] = confianca["aceito"]
        melhor["motivos_baixa_confianca"] = confianca["motivos"]
        melhor["matching_engine"] = MOTOR_BUSCA
        melhor["top_candidatos"] = top_candidatos[:5]

        resultados[busca_norm] = (melhor_idx, melhor)

    return resultados
