from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from llm_service import BackendUnavailableError, LLMBackend, LLMDecisionConfig, create_default_backend, run_llm_decision
from semantic_matcher import extract_json_payload, validate_llm_payload

logger = logging.getLogger("site_orcamento_ia.semantic")


def _texto_candidato(candidato: Any) -> str:
    if isinstance(candidato, dict):
        for campo in ("descricao", "item", "texto", "label"):
            if candidato.get(campo):
                return str(candidato[campo]).strip()
        return str(candidato).strip()
    return str(candidato).strip()


def _score_candidato(candidato: Any) -> float:
    if isinstance(candidato, dict):
        for campo in ("score_final", "score", "score_base", "confidence"):
            valor = candidato.get(campo)
            if isinstance(valor, (int, float)):
                return float(valor)
    return 0.0


def _compactar_lista(valor: Any, limite: int = 4) -> list[str]:
    if not isinstance(valor, (list, tuple, set)):
        if valor in (None, "", [], {}, ()):
            return []
        return [_texto_candidato(valor)]
    itens: list[str] = []
    for item in valor:
        texto = _texto_candidato(item)
        if texto:
            itens.append(texto)
        if len(itens) >= limite:
            break
    return itens


def _candidato_tem_sinal_tecnico(candidato: Any, config: LLMDecisionConfig) -> bool:
    if not isinstance(candidato, dict):
        return False
    if float(candidato.get("score_base", 0.0) or 0.0) > 0.0:
        if float(candidato.get("score_base", 0.0) or 0.0) >= config.decision_low_score_min_base:
            return True
    for campo in ("familia_principal", "subfamilias", "coincidencias_tecnicas"):
        valor = candidato.get(campo)
        if isinstance(valor, (list, tuple, set)) and len(valor) > 0:
            return True
    return False


def _resumo_candidato_llm(candidato: Any) -> Dict[str, Any]:
    if not isinstance(candidato, dict):
        return {"descricao": _texto_candidato(candidato)}
    return {
        "descricao": _texto_candidato(candidato),
        "score_final": round(float(candidato.get("score_final", 0.0) or 0.0), 4),
        "familia_principal": _compactar_lista(candidato.get("familia_principal")),
        "subfamilias": _compactar_lista(candidato.get("subfamilias")),
        "natureza_item": _compactar_lista(candidato.get("natureza_item")),
        "coincidencias_tecnicas": _compactar_lista(candidato.get("coincidencias_tecnicas")),
        "conflitos_tecnicos": _compactar_lista(candidato.get("conflitos_tecnicos")),
        "unidade": _texto_candidato(candidato.get("unidade")),
        "codigo": _texto_candidato(candidato.get("codigo")),
    }


def _fallback_candidate(candidates: List[Any]) -> tuple[int, str, float, Dict[str, Any]]:
    if not candidates:
        return 1, "", 0.0, {"semantic_decision_used": False, "fallback_reason": "sem_candidatos"}

    melhor_pos = max(range(len(candidates)), key=lambda pos: _score_candidato(candidates[pos]))
    escolhido = candidates[melhor_pos]
    confidence = max(0.0, min(1.0, _score_candidato(escolhido)))
    return (
        melhor_pos + 1,
        _texto_candidato(escolhido),
        confidence,
        {"semantic_decision_used": False, "fallback_reason": "ranking_tradicional"},
    )


def _deve_usar_decisao_semantica(candidates: List[Any], config: LLMDecisionConfig) -> tuple[bool, str]:
    if len(candidates) < 2:
        return False, "menos_de_dois_candidatos"

    ordenados = sorted(candidates, key=_score_candidato, reverse=True)
    melhor_score = _score_candidato(ordenados[0])
    segundo_score = _score_candidato(ordenados[1])
    gap = melhor_score - segundo_score

    if config.decision_min_top_score <= melhor_score <= config.decision_max_top_score and gap <= config.decision_max_gap:
        return True, f"ambiguo(score_topo={melhor_score:.3f}|gap={gap:.3f})"

    if config.decision_low_score_min <= melhor_score < config.decision_low_score_max:
        if segundo_score < config.decision_low_score_second_min:
            return False, f"baixo_score_sem_segundo_plausivel({segundo_score:.3f}<{config.decision_low_score_second_min:.3f})"
        topo_tem_sinal = _candidato_tem_sinal_tecnico(ordenados[0], config)
        segundo_tem_sinal = _candidato_tem_sinal_tecnico(ordenados[1], config)
        if not (topo_tem_sinal or segundo_tem_sinal):
            return False, "baixo_score_sem_sinal_tecnico"
        return True, f"baixo_score_recuperavel(score_topo={melhor_score:.3f}|score_segundo={segundo_score:.3f})"

    if melhor_score < config.decision_min_top_score:
        return False, f"score_topo_baixo({melhor_score:.3f}<{config.decision_min_top_score:.3f})"
    if melhor_score > config.decision_max_top_score:
        return False, f"score_topo_muito_alto({melhor_score:.3f}>{config.decision_max_top_score:.3f})"
    if gap > config.decision_max_gap:
        return False, f"gap_confianca_alto({gap:.3f}>{config.decision_max_gap:.3f})"
    return True, f"ambiguo(score_topo={melhor_score:.3f}|gap={gap:.3f})"


def decide_best_candidate_with_llm(
    input_description: str,
    candidates: List[Any],
    metadata: Optional[Dict[str, Any]] = None,
    *,
    backend: Optional[LLMBackend] = None,
    config: Optional[LLMDecisionConfig] = None,
) -> Dict[str, Any]:
    config = config or LLMDecisionConfig()
    metadata = metadata or {}

    if not candidates:
        idx, item, confidence, extras = _fallback_candidate([])
        return {
            "chosen_index": idx,
            "chosen_item": item,
            "confidence": confidence,
            "short_reason": "Sem candidatos disponíveis.",
            "rejected_items_summary": "",
            **extras,
            "semantic_confidence": None,
            "backend": None,
            "model_name": config.model_name,
            "inference_ms": 0.0,
        }

    backend_atual = backend
    if backend_atual is None and config.enabled:
        backend_atual = create_default_backend(config)

    fallback_idx, fallback_item, fallback_confidence, fallback_extras = _fallback_candidate(candidates)
    if not config.enabled or backend_atual is None:
        return {
            "chosen_index": fallback_idx,
            "chosen_item": fallback_item,
            "confidence": fallback_confidence,
            "short_reason": "Fallback do ranking tradicional.",
            "rejected_items_summary": "",
            **fallback_extras,
            "semantic_confidence": None,
            "backend": None,
            "model_name": config.model_name,
            "inference_ms": 0.0,
        }

    candidatos_limitados = list(candidates[: max(2, min(config.max_candidates, len(candidates)))])
    deve_usar_llm, motivo_gatilho = _deve_usar_decisao_semantica(candidatos_limitados, config)
    if not deve_usar_llm:
        return {
            "chosen_index": fallback_idx,
            "chosen_item": fallback_item,
            "confidence": fallback_confidence,
            "short_reason": "Fallback do ranking tradicional.",
            "rejected_items_summary": "",
            **fallback_extras,
            "semantic_confidence": None,
            "backend": None,
            "model_name": config.model_name,
            "inference_ms": 0.0,
            "candidate_count": len(candidatos_limitados),
            "semantic_decision_skipped": motivo_gatilho,
        }

    start = time.perf_counter()
    try:
        response_text, inference_ms = run_llm_decision(
            backend_atual,
            input_description=input_description,
            candidates=[_resumo_candidato_llm(candidato) for candidato in candidatos_limitados],
            metadata=metadata,
            config=config,
        )
        payload = extract_json_payload(response_text)
        if payload is None:
            raise ValueError("Não foi possível extrair JSON da resposta da LLM.")

        payload_validado = validate_llm_payload(payload, len(candidatos_limitados))
        confidence_semantica = float(payload_validado["confidence"])
        if confidence_semantica < config.min_confidence:
            raise ValueError(f"Confidence abaixo do mínimo ({confidence_semantica:.3f} < {config.min_confidence:.3f}).")

        escolhido = candidatos_limitados[payload_validado["chosen_index"] - 1]
        item_escolhido = _texto_candidato(escolhido)
        return {
            "chosen_index": payload_validado["chosen_index"],
            "chosen_item": item_escolhido,
            "confidence": confidence_semantica,
            "short_reason": payload_validado["short_reason"],
            "rejected_items_summary": payload_validado["rejected_items_summary"],
            "semantic_decision_used": True,
            "fallback_reason": None,
            "semantic_confidence": confidence_semantica,
            "backend": backend_atual.__class__.__name__,
            "model_name": config.model_name,
            "inference_ms": inference_ms,
            "candidate_count": len(candidatos_limitados),
        }
    except (TimeoutError, BackendUnavailableError, ValueError, KeyError, IndexError, Exception) as exc:
        logger.warning("Fallback da decisão semântica acionado em %.2fms: %s", (time.perf_counter() - start) * 1000.0, exc)
        return {
            "chosen_index": fallback_idx,
            "chosen_item": fallback_item,
            "confidence": fallback_confidence,
            "short_reason": "Fallback do ranking tradicional.",
            "rejected_items_summary": "",
            **fallback_extras,
            "semantic_confidence": None,
            "backend": backend_atual.__class__.__name__ if backend_atual is not None else None,
            "model_name": config.model_name,
            "inference_ms": round((time.perf_counter() - start) * 1000.0, 2),
            "fallback_exception": str(exc),
            "candidate_count": len(candidatos_limitados),
        }
