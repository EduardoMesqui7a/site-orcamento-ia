from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class FamilySpec:
    key: str
    domain: str
    aliases: tuple[str, ...]
    negative_context: tuple[str, ...] = ()
    required_tokens_any: tuple[str, ...] = ()
    priority: int = 100


DOMAIN_LABELS: Dict[str, str] = {
    "civil": "Civil",
    "hidrossanitario": "Hidrossanitário",
    "incendio": "Incêndio",
    "eletrica": "Elétrica",
    "tubulacao_industrial": "Tubulação industrial",
    "hvac": "HVAC",
    "canteiro_e_administracao": "Canteiro e administração",
}


MATERIAL_ALIASES: Dict[str, List[str]] = {
    "cobre": ["cobre", "cobre estanhado", "liga de cobre"],
    "aluminio": ["aluminio", "aluminio anodizado", "acm"],
    "aco": ["aco", "aco carbono", "aco galvanizado", "galvanizado", "inox", "aco inox", "ferro fundido"],
    "pvc": ["pvc", "pvc rigido"],
    "pead": ["pead"],
    "ppr": ["ppr"],
    "cpvc": ["cpvc"],
    "concreto": ["concreto"],
    "argamassa": ["argamassa", "rejunte"],
    "madeira": ["madeira", "compensado", "mdf", "mdp"],
    "gesso": ["gesso", "drywall", "gesso acartonado"],
    "ceramica": ["ceramica", "porcelanato", "louca"],
    "vidro": ["vidro", "temperado", "laminado"],
    "borracha": ["borracha", "buna", "neoprene", "epdm"],
    "pir": ["pir", "poliisocianurato"],
}


CLASS_ALIASES: Dict[str, List[str]] = {
    "aci": ["ac-i", "ac i", "aci"],
    "acii": ["ac-ii", "ac ii", "acii"],
    "aciii": ["ac-iii", "ac iii", "aciii"],
    "sch40": ["sch 40", "sch. 40", "schedule 40"],
    "sch80": ["sch 80", "sch. 80", "schedule 80"],
    "sn4": ["sn4", "sn 4"],
    "sn8": ["sn8", "sn 8"],
    "pn10": ["pn10", "pn 10"],
    "pn16": ["pn16", "pn 16"],
    "pn25": ["pn25", "pn 25"],
    "pba": ["pba"],
    "soldavel": ["soldavel"],
    "roscavel": ["roscavel", "rosqueado", "rosca npt", "npt"],
    "flexivel": ["flexivel"],
    "rigido": ["rigido"],
    "din": ["tipo din", "trilho din", "din"],
    "termomagnetico": ["termomagnetico", "termo magnetico", "tmf"],
    "caixa_moldada": ["caixa moldada"],
    "classe_150": ["150 lbs", "classe 150", "150#"],
    "classe_3000": ["3000 lbs", "3000#", "classe 3000"],
    "classe_6000": ["6000 lbs", "6000#", "classe 6000"],
}


SUBFAMILY_ALIASES: Dict[str, Dict[str, List[str]]] = {
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
        "solda": ["solda", "bisel", "biselada", "encaixe solda"],
        "sem_costura": ["sem costura", "sc"],
        "com_costura": ["com costura", "cc"],
        "ranhurado": ["ranhurada", "ranhurado"],
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
        "caixa_moldada": ["caixa moldada", "tmf"],
    },
    "valvula": {
        "esfera": ["esfera"],
        "globo": ["globo"],
        "gaveta": ["gaveta"],
        "borboleta": ["borboleta"],
        "angular": ["angular"],
    },
    "flange": {
        "cego": ["flange cego", "cego"],
        "roscado": ["flange roscado", "roscado"],
        "sobreposto": ["flange sobreposto", "sobreposto"],
        "solda_pescoco": ["welding neck", "pescoco", "weld neck"],
    },
    "curva": {
        "45_graus": ["45 graus", "45°", "45o"],
        "90_graus": ["90 graus", "90°", "90o"],
        "solda": ["solda", "bisel", "biselada", "encaixe solda"],
        "ranhurada": ["ranhurada", "ranhurado"],
    },
    "te": {
        "reto": ["te reto", "tee straight"],
        "reducao": ["te de reducao", "te reducao", "reducao"],
        "solda": ["solda", "bisel", "biselada", "encaixe solda"],
        "roscado": ["rosca", "roscavel", "npt"],
    },
    "cuba": {
        "embutir": ["embutir", "sobrepor"],
        "apoio": ["apoio"],
        "inox": ["inox", "aco inox"],
        "louca": ["louca", "ceramica"],
    },
    "sifao": {
        "metalico": ["metalico", "cromado"],
        "garrafa": ["garrafa"],
    },
    "painel": {
        "eletrico": ["qgbt", "qdc", "qdl", "ccm", "painel eletrico"],
        "fachada": ["painel sandwich", "acm"],
    },
}


SERVICE_ADMIN_MARKERS: List[str] = [
    "mobilizacao",
    "desmobilizacao",
    "licenca",
    "taxa da obra",
    "locacao de modulos",
    "locacao de tenda",
    "operacao do canteiro",
    "almoxarife",
    "apontador",
    "encargos complementares",
    "momento extraordinario de transporte",
    "carga manobra e descarga",
    "retirada de entulho",
    "administracao local",
]


FAMILY_SPECS: Dict[str, FamilySpec] = {
    "hidrante": FamilySpec(
        key="hidrante",
        domain="incendio",
        aliases=("hidrante", "coluna de hidrante", "hidrante tipo coluna"),
        priority=1,
    ),
    "vaso_sanitario": FamilySpec(
        key="vaso_sanitario",
        domain="hidrossanitario",
        aliases=("vaso sanitario", "bacia sanitaria", "caixa acoplada", "louca sanitaria"),
        priority=2,
    ),
    "cuba": FamilySpec(
        key="cuba",
        domain="hidrossanitario",
        aliases=("cuba", "cubas", "cuba de embutir", "cuba inox", "cuba em inox"),
        negative_context=("sifao",),
        priority=3,
    ),
    "sifao": FamilySpec(
        key="sifao",
        domain="hidrossanitario",
        aliases=("sifao", "sifao metalico"),
        priority=4,
    ),
    "isolamento": FamilySpec(
        key="isolamento",
        domain="hvac",
        aliases=("isolamento", "isolante", "la de rocha", "lã de rocha", "calha de la de rocha"),
        priority=5,
    ),
    "disjuntor": FamilySpec(
        key="disjuntor",
        domain="eletrica",
        aliases=("disjuntor", "disjuntores", "minidisjuntor", "mini disjuntor", "mini-disjuntor"),
        required_tokens_any=("a", "p", "tripolar", "monopolar", "bipolar", "tetrapolar", "tmf", "din"),
        priority=6,
    ),
    "contator": FamilySpec(
        key="contator",
        domain="eletrica",
        aliases=("contator", "contatores"),
        priority=7,
    ),
    "rele": FamilySpec(
        key="rele",
        domain="eletrica",
        aliases=("rele", "reles", "relé", "relés"),
        priority=8,
    ),
    "interruptor": FamilySpec(
        key="interruptor",
        domain="eletrica",
        aliases=("interruptor", "interruptores"),
        priority=9,
    ),
    "tomada": FamilySpec(
        key="tomada",
        domain="eletrica",
        aliases=("tomada", "tomadas", "plug", "plugs", "plugue", "plugues"),
        negative_context=("grelha de tomada de ar",),
        priority=10,
    ),
    "quadro": FamilySpec(
        key="quadro",
        domain="eletrica",
        aliases=("quadro de distribuicao", "qdc", "qdl", "qgbt"),
        priority=11,
    ),
    "painel": FamilySpec(
        key="painel",
        domain="eletrica",
        aliases=("painel", "paineis", "ccm"),
        negative_context=("painel sandwich", "painel em acm", "andaime metalico tubular", "torre"),
        required_tokens_any=("eletrico", "qgbt", "qdc", "qdl", "ccm"),
        priority=12,
    ),
    "valvula": FamilySpec(
        key="valvula",
        domain="hidrossanitario",
        aliases=("valvula", "valvulas", "válvula", "válvulas", "registro", "registros"),
        priority=13,
    ),
    "flange": FamilySpec(
        key="flange",
        domain="tubulacao_industrial",
        aliases=("flange", "flanges"),
        negative_context=("junta para flange",),
        priority=14,
    ),
    "grampo": FamilySpec(
        key="grampo",
        domain="tubulacao_industrial",
        aliases=("grampo", "grampos", "suporte u", "abracadeira", "abraçadeira"),
        priority=15,
    ),
    "bucha": FamilySpec(
        key="bucha",
        domain="hidrossanitario",
        aliases=("bucha", "buchas"),
        priority=16,
    ),
    "te": FamilySpec(
        key="te",
        domain="tubulacao_industrial",
        aliases=("te de reducao", "te reto", "te", "tê", "tee"),
        negative_context=("teste",),
        priority=17,
    ),
    "curva": FamilySpec(
        key="curva",
        domain="tubulacao_industrial",
        aliases=("curva", "joelho", "cotovelo"),
        priority=18,
    ),
    "reducao": FamilySpec(
        key="reducao",
        domain="tubulacao_industrial",
        aliases=("reducao", "redução", "redutor", "bucha de redução"),
        priority=19,
    ),
    "luva": FamilySpec(
        key="luva",
        domain="tubulacao_industrial",
        aliases=("luva", "meia luva", "uniao", "união"),
        priority=20,
    ),
    "terminal": FamilySpec(
        key="terminal",
        domain="eletrica",
        aliases=("terminal", "terminais", "conector", "olhal", "terminal tipo sapata", "sapata terminal"),
        negative_context=("circuitos terminais", "circuito terminal", "ramal terminal", "tomada terminal"),
        priority=21,
    ),
    "tubo": FamilySpec(
        key="tubo",
        domain="tubulacao_industrial",
        aliases=("tubo", "tubos", "tubulacao", "tubulacoes", "cano", "canos"),
        negative_context=("eletroduto", "conduite", "sensor de vazao", "isolamento"),
        priority=22,
    ),
    "eletroduto": FamilySpec(
        key="eletroduto",
        domain="eletrica",
        aliases=("eletroduto", "eletrodutos", "conduite", "conduites"),
        priority=23,
    ),
    "cabo": FamilySpec(
        key="cabo",
        domain="eletrica",
        aliases=("cabo", "cabos", "condutor", "condutores"),
        negative_context=("cabo de aco para içamento",),
        priority=24,
    ),
    "porta": FamilySpec(
        key="porta",
        domain="civil",
        aliases=("porta",),
        priority=25,
    ),
    "janela": FamilySpec(
        key="janela",
        domain="civil",
        aliases=("janela", "esquadria"),
        priority=26,
    ),
    "piso": FamilySpec(
        key="piso",
        domain="civil",
        aliases=("piso", "revestimento", "porcelanato", "ceramica"),
        negative_context=("remocao de telhas", "telha", "telhado"),
        priority=27,
    ),
    "concreto": FamilySpec(
        key="concreto",
        domain="civil",
        aliases=("concreto",),
        priority=28,
    ),
    "argamassa": FamilySpec(
        key="argamassa",
        domain="civil",
        aliases=("argamassa", "rejunte", "chapisco", "reboco", "emboço", "emboco"),
        priority=29,
    ),
    "alvenaria": FamilySpec(
        key="alvenaria",
        domain="civil",
        aliases=("alvenaria", "bloco", "tijolo"),
        priority=30,
    ),
    "pintura": FamilySpec(
        key="pintura",
        domain="civil",
        aliases=("pintura", "tinta", "selador", "verniz", "epoxi"),
        priority=31,
    ),
    "escavacao": FamilySpec(
        key="escavacao",
        domain="civil",
        aliases=("escavacao", "aterro", "compactacao", "subleito", "movimento de terra"),
        priority=32,
    ),
    "mobilizacao": FamilySpec(
        key="mobilizacao",
        domain="canteiro_e_administracao",
        aliases=("mobilizacao", "desmobilizacao", "canteiro de obras", "locacao de modulos", "locacao de tenda"),
        priority=33,
    ),
    "taxas_licencas": FamilySpec(
        key="taxas_licencas",
        domain="canteiro_e_administracao",
        aliases=("licencas", "taxas da obra", "taxas", "alvara", "anotacao de responsabilidade"),
        priority=34,
    ),
    "transporte_logistica": FamilySpec(
        key="transporte_logistica",
        domain="canteiro_e_administracao",
        aliases=("momento extraordinario de transporte", "carga manobra e descarga", "retirada de entulho", "transporte de material"),
        priority=35,
    ),
    "junta": FamilySpec(
        key="junta",
        domain="tubulacao_industrial",
        aliases=("junta", "junta plana", "gaxeta"),
        priority=36,
    ),
    "filtro": FamilySpec(
        key="filtro",
        domain="tubulacao_industrial",
        aliases=("filtro", "coalescente", "separador de condensado"),
        priority=37,
    ),
    "sensor": FamilySpec(
        key="sensor",
        domain="eletrica",
        aliases=("sensor", "transmissor", "chave de fluxo", "medidor de vazao"),
        priority=38,
    ),
    "adaptador": FamilySpec(
        key="adaptador",
        domain="hidrossanitario",
        aliases=("adaptador", "adaptador curto", "adaptador longo"),
        priority=39,
    ),
    "demolicao": FamilySpec(
        key="demolicao",
        domain="civil",
        aliases=("demolicao", "remocao", "retirada", "desmontagem"),
        priority=40,
    ),
    "mao_de_obra": FamilySpec(
        key="mao_de_obra",
        domain="canteiro_e_administracao",
        aliases=("encargos complementares", "almoxarife", "apontador", "servente", "pedreiro", "eletricista"),
        priority=41,
    ),
}


PRIORITY_ORDER: Dict[str, int] = {spec.key: spec.priority for spec in FAMILY_SPECS.values()}
FAMILY_ALIASES: Dict[str, List[str]] = {spec.key: list(spec.aliases) for spec in FAMILY_SPECS.values()}
DOMAIN_BY_FAMILY: Dict[str, str] = {spec.key: spec.domain for spec in FAMILY_SPECS.values()}
FAMILY_NEGATIVE_CONTEXT: Dict[str, List[str]] = {spec.key: list(spec.negative_context) for spec in FAMILY_SPECS.values()}
FAMILY_REQUIRED_ANY: Dict[str, List[str]] = {spec.key: list(spec.required_tokens_any) for spec in FAMILY_SPECS.values()}


def iter_family_specs() -> Iterable[FamilySpec]:
    return FAMILY_SPECS.values()
