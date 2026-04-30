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

logger = logging.getLogger("site_orcamento_ia.motor")

MOTOR_BUSCA = "tfidf-char-ngrams"
PESO_SEMANTICO = 0.55
PESO_FUZZY = 0.20
PESO_REGRAS = 0.25
TOP_K_PADRAO = 50
TOP_K_RERANK_TECNICO = 200
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


def _extrair_polegadas(texto: str) -> Set[str]:
    polegadas: Set[str] = set()
    for inteiro, fracao in re.findall(r"(?<![\d./])(\d+)[\.\s]+(\d+/\d+)\s*(?:\"|pol|polegada|polegadas)?", texto, flags=re.IGNORECASE):
        polegadas.add(f"{inteiro}.{fracao}")
    for valor in re.findall(r"(?<![\d./])(\d+/\d+)\s*(?:\"|pol|polegada|polegadas)?", texto, flags=re.IGNORECASE):
        polegadas.add(valor)
    for valor in re.findall(r"(?<![\d./-])(\d+(?:\.\d+)?)\s*(?:\"|pol|polegada|polegadas)", texto, flags=re.IGNORECASE):
        polegadas.add(_normalizar_valor_tecnico(valor))
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
    prioridade = [
        "hidrante",
        "vaso_sanitario",
        "cuba",
        "sifao",
        "isolamento",
        "disjuntor",
        "contator",
        "rele",
        "interruptor",
        "tomada",
        "quadro",
        "painel",
        "valvula",
        "flange",
        "grampo",
        "bucha",
        "te",
        "curva",
        "reducao",
        "luva",
        "terminal",
        "tubo",
        "eletroduto",
        "cabo",
        "porta",
        "janela",
        "piso",
        "concreto",
        "argamassa",
        "alvenaria",
        "pintura",
        "escavacao",
    ]
    pontuacoes: List[Tuple[int, int, str]] = []
    for familia in familias_detectadas:
        aliases = aliases_familias.get(familia, [])
        score = sum(
            1 for alias in aliases if re.search(rf"(^|[^\w]){re.escape(alias)}($|[^\w])", texto)
        )
        prioridade_idx = prioridade.index(familia) if familia in prioridade else len(prioridade)
        pontuacoes.append((score, -prioridade_idx, familia))
    pontuacoes.sort(reverse=True)
    return pontuacoes[0][2]


def inferir_subfamilias(texto: str, familia_principal: Optional[str]) -> Set[str]:
    if not familia_principal:
        return set()
    subfamilias_por_familia: Dict[str, Dict[str, List[str]]] = {
        "cabo": {
            "afumex": ["afumex"],
            "monopolar": ["monopolar", "1x1c", "1c#", "singelo"],
            "tripolar": ["tripolar", "3x", "3c#"],
            "tetrapolar": ["tetrapolar", "4x", "4c#"],
            "flexivel": ["flexivel"],
            "epr": ["epr"],
            "xlpe": ["xlpe"],
            "pvc": ["pvc"],
        },
        "tubo": {
            "roscavel": ["rosca", "roscavel", "npt"],
            "galvanizado": ["galvanizado"],
            "solda": ["solda", "bisel", "biselada"],
            "sem_costura": ["sem costura", "sc"],
            "com_costura": ["com costura", "cc"],
        },
        "disjuntor": {
            "minidisjuntor": ["minidisjuntor", "mini disjuntor"],
            "monopolar": ["monopolar", "1p", "1 p", "1 polo"],
            "bipolar": ["bipolar", "2p", "2 p", "2 polos"],
            "tripolar": ["tripolar", "3p", "3 p", "3 polos"],
            "tetrapolar": ["tetrapolar", "4p", "4 p", "4 polos"],
            "curva_b": ["curva b"],
            "curva_c": ["curva c"],
            "curva_d": ["curva d"],
        },
        "valvula": {
            "esfera": ["esfera"],
            "globo": ["globo"],
            "gaveta": ["gaveta"],
            "borboleta": ["borboleta"],
        },
    }
    return _extrair_tokens_lista(texto, subfamilias_por_familia.get(familia_principal, {}))


def inferir_natureza_item(
    texto: str,
    familias_detectadas: Set[str],
    familia_principal: Optional[str],
) -> str:
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

    materiais = {
        "cobre": ["cobre", "cobre estanhado", "liga de cobre"],
        "aluminio": ["aluminio", "aluminio anodizado"],
        "aco": ["aco", "aco carbono", "aco galvanizado", "galvanizado", "inox", "aco inox"],
        "pvc": ["pvc", "pvc rigido"],
        "pead": ["pead"],
        "ppr": ["ppr"],
        "cpvc": ["cpvc"],
        "concreto": ["concreto"],
        "argamassa": ["argamassa"],
        "madeira": ["madeira", "compensado"],
        "gesso": ["gesso", "drywall", "gesso acartonado"],
        "ceramica": ["ceramica", "porcelanato"],
        "vidro": ["vidro"],
    }
    familias = {
        "hidrante": ["hidrante", "coluna de hidrante", "hidrante tipo coluna"],
        "vaso_sanitario": ["vaso sanitario", "bacia sanitaria", "caixa acoplada", "louca"],
        "cuba": ["cuba", "cubas", "cuba de embutir", "cuba inox", "cuba em inox"],
        "sifao": ["sifao", "sifao metalico"],
        "isolamento": ["isolamento", "isolante", "la de rocha", "lã de rocha"],
        "terminal": ["terminal", "terminais", "conector", "olhal", "sapata"],
        "cabo": ["cabo", "cabos", "condutor", "condutores"],
        "disjuntor": ["disjuntor", "disjuntores", "minidisjuntor", "mini disjuntor", "mini-disjuntor"],
        "contator": ["contator", "contatores"],
        "rele": ["rele", "reles", "relé", "relés"],
        "interruptor": ["interruptor", "interruptores"],
        "tomada": ["tomada", "tomadas", "plug", "plugs", "plugue", "plugues"],
        "quadro": ["quadro de distribuicao", "qdc", "qdl", "qgbt"],
        "painel": ["painel", "paineis", "ccm"],
        "tubo": ["tubo", "tubos", "tubulacao", "tubulacoes", "cano", "canos"],
        "eletroduto": ["eletroduto", "eletrodutos", "conduite", "conduites"],
        "grampo": ["grampo", "grampos", "suporte u", "abracadeira", "abraçadeira"],
        "bucha": ["bucha", "buchas"],
        "te": ["te de reducao", "te reto", "te", "tê", "tee"],
        "curva": ["curva", "joelho", "cotovelo"],
        "reducao": ["reducao", "redução"],
        "luva": ["luva", "meia luva", "uniao", "união"],
        "flange": ["flange", "flanges"],
        "valvula": ["valvula", "valvulas", "válvula", "válvulas", "registro", "registros"],
        "concreto": ["concreto"],
        "argamassa": ["argamassa", "rejunte", "chapisco", "reboco", "emboço", "emboco"],
        "alvenaria": ["alvenaria", "bloco", "tijolo"],
        "piso": ["piso", "revestimento", "porcelanato", "ceramica"],
        "porta": ["porta"],
        "janela": ["janela", "esquadria"],
        "pintura": ["pintura", "tinta", "selador", "verniz"],
        "escavacao": ["escavacao", "aterro", "compactacao"],
    }
    classes = {
        "aci": ["ac-i", "ac i", "aci"],
        "acii": ["ac-ii", "ac ii", "acii"],
        "aciii": ["ac-iii", "ac iii", "aciii"],
        "sch40": ["sch 40", "sch. 40", "schedule 40"],
        "sch80": ["sch 80", "sch. 80", "schedule 80"],
        "sn4": ["sn4", "sn 4"],
        "sn8": ["sn8", "sn 8"],
        "pn10": ["pn10", "pn 10"],
        "pn16": ["pn16", "pn 16"],
        "pba": ["pba"],
        "soldavel": ["soldavel"],
        "roscavel": ["roscavel"],
        "flexivel": ["flexivel"],
        "rigido": ["rigido"],
        "din": ["tipo din", "trilho din", "din"],
        "termomagnetico": ["termomagnetico", "termo magnetico"],
        "caixa_moldada": ["caixa moldada"],
    }

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
        "materiais": _extrair_tokens_lista(tecnico, materiais),
        "familias": _extrair_tokens_lista(tecnico, familias),
        "classes": _extrair_tokens_lista(tecnico, classes),
    }

    familia_principal = inferir_familia_principal(tecnico, atributos["familias"], familias) if atributos["familias"] else None
    if familia_principal:
        atributos["familia_principal"] = {familia_principal}
        subfamilias = inferir_subfamilias(tecnico, familia_principal)
        if subfamilias:
            atributos["subfamilias"] = subfamilias
        atributos["natureza_item"] = {inferir_natureza_item(tecnico, atributos["familias"], familia_principal)}
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
        "familia_principal",
        "subfamilias",
        "classes",
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
        "subfamilias",
        "classes",
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
        "subfamilias": 0.08,
        "bitola_mm2": 0.08,
        "dn": 0.08,
        "polegada": 0.08,
        "diametro_nominal": 0.12,
        "diametro_mm": 0.08,
        "diametro_cm": 0.08,
        "classes": 0.06,
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
        "subfamilias": 0.14,
        "bitola_mm2": 0.18,
        "dn": 0.18,
        "polegada": 0.45,
        "diametro_nominal": 0.34,
        "diametro_mm": 0.20,
        "diametro_cm": 0.20,
        "classes": 0.12,
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


def preparar_base_para_busca(df_base: pd.DataFrame, coluna_texto_base: str):
    df_base_proc = df_base.copy()
    df_base_proc[coluna_texto_base] = df_base_proc[coluna_texto_base].fillna("").astype(str)
    df_base_proc["__texto_base_original__"] = df_base_proc[coluna_texto_base]
    df_base_proc["__texto_base_norm__"] = df_base_proc[coluna_texto_base].map(normalizar_texto)
    df_base_proc["__atributos_tecnicos__"] = df_base_proc[coluna_texto_base].map(extrair_atributos_tecnicos)

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
        melhores_indices = similaridades.argsort()[::-1][:k]

        candidatos_ranqueados: List[dict] = []
        for idx_base in melhores_indices:
            texto_base_norm = textos_base_norm[int(idx_base)]
            atributos_base = attrs_base_lista[int(idx_base)]
            texto_base_original = textos_base_originais[int(idx_base)]
            score_sem = float(similaridades[int(idx_base)])
            score_fuzzy = fuzz.token_set_ratio(busca_norm, texto_base_norm) / 100.0
            score_reg = score_regras(busca_norm, texto_base_norm)
            bonus_tecnico, coincidencias, conflitos = _score_bonus_tecnico(atributos_busca, atributos_base)
            score_base = (PESO_SEMANTICO * score_sem) + (PESO_FUZZY * score_fuzzy) + (PESO_REGRAS * score_reg)
            score_final = max(
                0.0,
                min(
                    1.0,
                    score_base + bonus_tecnico,
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
