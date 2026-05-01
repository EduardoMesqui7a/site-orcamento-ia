from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_service import LLMDecisionConfig
from motor_itemiza import buscar_melhor_item_em_lote, normalizar_texto, preparar_base_para_busca


BASE_PATH = ROOT / "taxonomy_inputs" / "base_teste3.xlsx"
CASES_PATH = ROOT / "benchmarks" / "teste3_cases.json"
OUTPUT_DIR = ROOT / "benchmarks" / "results"


def evaluate_case(result: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    description = result["descricao_encontrada"].lower()
    family = result["familia_principal"]
    preferred = [kw.lower() for kw in case.get("preferred_keywords", [])]
    required_any = [kw.lower() for kw in case.get("required_keywords_any", [])]
    required_all = [kw.lower() for kw in case.get("required_keywords_all", [])]
    forbidden = [kw.lower() for kw in case.get("forbidden_keywords", [])]

    family_ok = family == case["expected_family"]
    preferred_hits = [kw for kw in preferred if kw in description]
    required_any_hits = [kw for kw in required_any if kw in description]
    required_all_hits = [kw for kw in required_all if kw in description]
    forbidden_hits = [kw for kw in forbidden if kw in description]

    passed = (
        family_ok
        and not forbidden_hits
        and (not preferred or bool(preferred_hits))
        and (not required_any or bool(required_any_hits))
        and (not required_all or len(required_all_hits) == len(required_all))
    )
    return {
        "family_ok": family_ok,
        "preferred_hits": preferred_hits,
        "required_any_hits": required_any_hits,
        "required_all_hits": required_all_hits,
        "forbidden_hits": forbidden_hits,
        "passed": passed,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_base = pd.read_excel(BASE_PATH, sheet_name="orcamento", header=2)
    df_base.columns = [str(col).strip() for col in df_base.columns]
    df_base_proc, vetorizador, matriz_base = preparar_base_para_busca(df_base, "DESCRIÇÃO")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    buscas_originais_por_norm = {normalizar_texto(case["description"]): case["description"] for case in cases}
    buscas_norm_unicas = list(buscas_originais_por_norm.keys())

    results = buscar_melhor_item_em_lote(
        buscas_norm_unicas=buscas_norm_unicas,
        buscas_originais_por_norm=buscas_originais_por_norm,
        df_base_proc=df_base_proc,
        vetorizador=vetorizador,
        matriz_base=matriz_base,
        top_k_candidatos=50,
        score_minimo_usuario=0.42,
        llm_config=LLMDecisionConfig(enabled=False),
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        busca_norm = normalizar_texto(case["description"])
        match = results.get(busca_norm)
        if not match:
            rows.append(
                {
                    "line": case["line"],
                    "description": case["description"],
                    "expected_family": case["expected_family"],
                    "familia_principal": None,
                    "descricao_encontrada": None,
                    "score_final": None,
                    "passed": False,
                    "reason": "sem_resultado",
                }
            )
            continue

        idx_base, det = match
        familia_principal = (det.get("familia_principal") or [None])[0]
        row = {
            "line": case["line"],
            "description": case["description"],
            "expected_family": case["expected_family"],
            "familia_principal": familia_principal,
            "descricao_encontrada": det.get("descricao", ""),
            "score_final": det.get("score_final"),
            "idx_base": idx_base,
            "top_candidatos": det.get("top_candidatos", []),
        }
        evaluation = evaluate_case(row, case)
        row.update(evaluation)
        rows.append(row)

    json_path = OUTPUT_DIR / "teste3_benchmark_results.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Teste 3 Benchmark Results")
    lines.append("")
    passed_count = sum(1 for row in rows if row["passed"])
    lines.append(f"- Passed: `{passed_count}/{len(rows)}`")
    lines.append("")
    for row in rows:
        lines.append(f"## Linha {row['line']}")
        lines.append(f"- Esperado: `{row['expected_family']}`")
        lines.append(f"- Encontrado: `{row['familia_principal']}`")
        lines.append(f"- Score: `{row['score_final']}`")
        lines.append(f"- Passou: `{row['passed']}`")
        lines.append(f"- Descrição encontrada: {row['descricao_encontrada']}")
        if row.get("forbidden_hits"):
            lines.append(f"- Forbidden hits: `{row['forbidden_hits']}`")
        if row.get("preferred_hits"):
            lines.append(f"- Preferred hits: `{row['preferred_hits']}`")
        if row.get("required_any_hits"):
            lines.append(f"- Required-any hits: `{row['required_any_hits']}`")
        if row.get("required_all_hits"):
            lines.append(f"- Required-all hits: `{row['required_all_hits']}`")
        lines.append("")

    (OUTPUT_DIR / "teste3_benchmark_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
