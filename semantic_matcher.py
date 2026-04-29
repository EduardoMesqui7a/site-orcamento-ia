from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

SYSTEM_PROMPT_PTBR = """
Você é um especialista em orçamento de obras. Escolha apenas entre os candidatos enviados o item que melhor representa a descrição.
Priorize equivalência técnica e conceitual, não apenas semelhança textual. Considere família, material, dimensão, bitola, DN, polegada,
classe, norma, aplicação e natureza do item. Diferencie item simples, acessório, conjunto e acabamento. Se houver ambiguidade, seja conservador.
Responda somente com JSON válido.
Formato:
{"chosen_index":1,"chosen_item":"...","confidence":0.0,"short_reason":"...","rejected_items_summary":"..."}
""".strip()


@dataclass(frozen=True)
class SemanticCandidate:
    index: int
    item: str
    metadata: Dict[str, Any]


def build_system_prompt(model_name: str) -> str:
    return f"{SYSTEM_PROMPT_PTBR}\nModelo alvo: {model_name}".strip()


def _normalizar_texto_curto(texto: Any) -> str:
    if texto is None:
        return ""
    valor = str(texto).strip()
    return re.sub(r"\s+", " ", valor)


def _truncar_texto(texto: Any, limite: int = 180) -> str:
    valor = _normalizar_texto_curto(texto)
    if len(valor) <= limite:
        return valor
    return valor[: max(0, limite - 3)].rstrip() + "..."


def _resumir_metadados_compactos(metadados: Optional[Dict[str, Any]]) -> str:
    if not metadados:
        return ""
    partes: List[str] = []
    for chave in (
        "score_final",
        "familia_principal",
        "subfamilias",
        "natureza_item",
        "coincidencias_tecnicas",
        "conflitos_tecnicos",
        "unidade",
        "codigo",
    ):
        valor = metadados.get(chave)
        if valor in (None, "", [], {}, ()):
            continue
        if isinstance(valor, (list, tuple, set)):
            itens = [_normalizar_texto_curto(item) for item in valor if _normalizar_texto_curto(item)]
            if itens:
                partes.append(f"{chave}={','.join(itens[:4])}")
        else:
            partes.append(f"{chave}={_truncar_texto(valor, 60)}")
    return "; ".join(partes)


def normalizar_candidatos(candidates: Iterable[Any], max_candidates: Optional[int] = None) -> List[SemanticCandidate]:
    candidatos: List[SemanticCandidate] = []
    for idx, candidato in enumerate(candidates, start=1):
        if isinstance(candidato, dict):
            item = (
                candidato.get("descricao")
                or candidato.get("item")
                or candidato.get("texto")
                or candidato.get("label")
                or str(candidato)
            )
            metadata = {
                chave: valor
                for chave, valor in candidato.items()
                if chave
                not in {"descricao", "item", "texto", "label", "__compat_tecnica__"}
                and valor not in (None, "", [], {}, ())
            }
        else:
            item = str(candidato)
            metadata = {}
        candidatos.append(SemanticCandidate(index=idx, item=_normalizar_texto_curto(item), metadata=metadata))

    if max_candidates is not None and max_candidates > 0:
        candidatos = candidatos[:max_candidates]
    return candidatos


def build_user_prompt(
    input_description: str,
    candidates: Iterable[Any],
    metadata: Optional[Dict[str, Any]] = None,
    max_candidates: Optional[int] = None,
) -> str:
    candidatos = normalizar_candidatos(candidates, max_candidates=max_candidates)
    linhas = [f"DESCRICAO: {_truncar_texto(input_description, 220)}"]
    contexto = _resumir_metadados_compactos(metadata or {})
    if contexto:
        linhas.append(f"CONTEXTO: {contexto}")
    linhas.append("CANDIDATOS:")
    for cand in candidatos:
        meta = _resumir_metadados_compactos(cand.metadata)
        item = _truncar_texto(cand.item, 180)
        if meta:
            linhas.append(f"{cand.index}) {item} | {meta}")
        else:
            linhas.append(f"{cand.index}) {item}")
    linhas.append("INSTRUCAO: escolha um único candidato e responda apenas JSON válido.")
    linhas.append(
        'JSON: {"chosen_index":1,"chosen_item":"...","confidence":0.0,"short_reason":"...","rejected_items_summary":"..."}'
    )
    return "\n".join(linhas)


def _strip_code_fences(texto: str) -> str:
    texto = texto.strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*", "", texto, flags=re.IGNORECASE)
        texto = re.sub(r"\s*```$", "", texto)
    return texto.strip()


def _extract_balanced_json(texto: str) -> Optional[str]:
    texto = _strip_code_fences(texto)
    if not texto:
        return None

    start = None
    depth = 0
    in_string = False
    escape = False
    for pos, char in enumerate(texto):
        if start is None:
            if char == "{":
                start = pos
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return texto[start : pos + 1]

    if texto.startswith("{") and texto.endswith("}"):
        return texto
    return None


def extract_json_payload(texto: str) -> Optional[Dict[str, Any]]:
    if not texto:
        return None

    texto = _strip_code_fences(texto)
    for candidate in (texto, _extract_balanced_json(texto)):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(candidate)
            except Exception:
                continue
        if isinstance(payload, dict):
            return payload
    return None


def validate_llm_payload(payload: Dict[str, Any], candidate_count: int) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload da LLM inválido.")

    try:
        chosen_index = int(payload.get("chosen_index"))
    except (TypeError, ValueError) as exc:
        raise ValueError("chosen_index inválido.") from exc

    if chosen_index < 1 or chosen_index > candidate_count:
        raise ValueError("chosen_index fora da lista.")

    chosen_item = _normalizar_texto_curto(payload.get("chosen_item"))
    if not chosen_item:
        raise ValueError("chosen_item vazio.")

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence inválida.") from exc

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence fora da faixa.")

    short_reason = _normalizar_texto_curto(payload.get("short_reason"))
    rejected_items_summary = _normalizar_texto_curto(payload.get("rejected_items_summary"))
    if not short_reason:
        raise ValueError("short_reason ausente.")
    if not rejected_items_summary:
        raise ValueError("rejected_items_summary ausente.")

    return {
        "chosen_index": chosen_index,
        "chosen_item": chosen_item,
        "confidence": confidence,
        "short_reason": short_reason,
        "rejected_items_summary": rejected_items_summary,
    }
