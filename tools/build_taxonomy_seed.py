from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motor_itemiza import extrair_atributos_tecnicos, normalizar_texto


INPUT_DIR = ROOT / "taxonomy_inputs"
OUTPUT_DIR = ROOT / "taxonomy"
DOCS_DIR = ROOT / "docs"


@dataclass(frozen=True)
class SourceConfig:
    key: str
    filename: str
    description_column: str


SOURCES = [
    SourceConfig("base_teste3", "base_teste3.xlsx", "C"),
    SourceConfig("base_teste4", "base_teste4.xlsx", "C"),
    SourceConfig("base_dados", "base_dados.xlsx", "G"),
    SourceConfig("contraprova", "contraprova.xlsx", "G"),
]


def excel_column_index(column_letter: str) -> int:
    return ord(column_letter.upper()) - ord("A")


def descriptions_by_sheet(path: Path, column_letter: str) -> list[tuple[str, list[str]]]:
    idx = excel_column_index(column_letter)
    xls = pd.ExcelFile(path)
    result: list[tuple[str, list[str]]] = []
    blacklist = {"descricao", "descrição", "item", "codigo", "código"}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        if idx >= len(df.columns):
            continue
        values = [
            str(value).strip()
            for value in df.iloc[:, idx].tolist()
            if value is not None
            and not pd.isna(value)
            and str(value).strip()
            and str(value).strip().lower() not in blacklist
        ]
        if values:
            result.append((sheet_name, values))
    if not result:
        raise ValueError(f"Nenhuma descrição não vazia encontrada em {path.name} coluna {column_letter}")
    return result


def shortlist_examples(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        desc = row["description"]
        if desc in seen:
            continue
        seen.add(desc)
        examples.append(
            {
                "source": row["source"],
                "sheet": row["sheet"],
                "description": desc,
            }
        )
        if len(examples) >= limit:
            break
    return examples


def flatten_attr_counter(rows: list[dict[str, Any]], attribute_name: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        values = row["attributes"].get(attribute_name, set())
        for value in values:
            counter[str(value)] += 1
    return counter


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, Any]] = {}
    source_sheet_stats: dict[str, list[dict[str, Any]]] = {}

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subfamily_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    material_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nature_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_family_rows: list[dict[str, Any]] = []

    family_counter: Counter[str] = Counter()
    subfamily_counter: Counter[str] = Counter()
    material_counter: Counter[str] = Counter()
    nature_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()

    for source in SOURCES:
        path = INPUT_DIR / source.filename
        sheets = descriptions_by_sheet(path, source.description_column)
        source_stats[source.key] = {
            "path": str(path),
            "description_column": source.description_column,
            "sheet_count": len(sheets),
            "description_count": sum(len(descriptions) for _, descriptions in sheets),
        }
        source_sheet_stats[source.key] = [
            {"sheet": sheet_name, "description_count": len(descriptions)}
            for sheet_name, descriptions in sheets
        ]

        for sheet_name, descriptions in sheets:
            for description in descriptions:
                normalized = normalizar_texto(description)
                attrs = extrair_atributos_tecnicos(description)

                row = {
                    "source": source.key,
                    "sheet": sheet_name,
                    "description": description,
                    "normalized": normalized,
                    "attributes": attrs,
                }
                all_rows.append(row)

                for token in normalized.split():
                    if len(token) >= 4:
                        token_counter[token] += 1

                families = sorted(attrs.get("familia_principal", set()))
                subfamilies = sorted(attrs.get("subfamilias", set()))
                materials = sorted(attrs.get("materiais", set()))
                natures = sorted(attrs.get("natureza_item", set()))

                if families:
                    family_counter[families[0]] += 1
                    family_rows[families[0]].append(row)
                else:
                    no_family_rows.append(row)

                for subfamily in subfamilies:
                    subfamily_counter[subfamily] += 1
                    subfamily_rows[subfamily].append(row)

                for material in materials:
                    material_counter[material] += 1
                    material_rows[material].append(row)

                for nature in natures:
                    nature_counter[nature] += 1
                    nature_rows[nature].append(row)

    frequent_attributes = {
        "classes": flatten_attr_counter(all_rows, "classes").most_common(40),
        "polegada": flatten_attr_counter(all_rows, "polegada").most_common(40),
        "dn": flatten_attr_counter(all_rows, "dn").most_common(40),
        "corrente_a": flatten_attr_counter(all_rows, "corrente_a").most_common(40),
        "tensao_v": flatten_attr_counter(all_rows, "tensao_v").most_common(40),
        "tensao_kv": flatten_attr_counter(all_rows, "tensao_kv").most_common(40),
        "bitola_mm2": flatten_attr_counter(all_rows, "bitola_mm2").most_common(40),
    }

    taxonomy_seed = {
        "source_stats": source_stats,
        "source_sheet_stats": source_sheet_stats,
        "totals": {
            "descriptions": len(all_rows),
            "families_detected": len(family_counter),
            "subfamilies_detected": len(subfamily_counter),
            "materials_detected": len(material_counter),
            "nature_types_detected": len(nature_counter),
            "unclassified_descriptions": len(no_family_rows),
        },
        "top_families": family_counter.most_common(50),
        "top_subfamilies": subfamily_counter.most_common(50),
        "top_materials": material_counter.most_common(40),
        "top_nature_types": nature_counter.most_common(30),
        "frequent_attributes": frequent_attributes,
        "top_tokens_without_taxonomy_filter": token_counter.most_common(120),
        "family_examples": {
            family: shortlist_examples(rows, limit=8)
            for family, rows in sorted(family_rows.items(), key=lambda item: (-len(item[1]), item[0]))
        },
        "subfamily_examples": {
            subfamily: shortlist_examples(rows, limit=6)
            for subfamily, rows in sorted(subfamily_rows.items(), key=lambda item: (-len(item[1]), item[0]))
        },
        "material_examples": {
            material: shortlist_examples(rows, limit=6)
            for material, rows in sorted(material_rows.items(), key=lambda item: (-len(item[1]), item[0]))
        },
        "unclassified_examples": shortlist_examples(no_family_rows, limit=80),
    }

    (OUTPUT_DIR / "seed_from_bases.json").write_text(
        json.dumps(taxonomy_seed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Taxonomy Seed From Real Budget Bases")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for key, stats in source_stats.items():
        lines.append(f"- `{key}`: `{stats['path']}`")
        lines.append(f"  - coluna de descrição: `{stats['description_column']}`")
        lines.append(f"  - abas lidas: `{stats['sheet_count']}`")
        lines.append(f"  - descrições lidas: `{stats['description_count']}`")
        for sheet in source_sheet_stats[key][:10]:
            lines.append(f"  - aba `{sheet['sheet']}`: {sheet['description_count']} descrições")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for key, value in taxonomy_seed["totals"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Top Families")
    lines.append("")
    for family, count in family_counter.most_common(30):
        lines.append(f"- `{family}`: {count}")
        for example in shortlist_examples(family_rows[family], limit=3):
            lines.append(f"  - {example['description']}")
    lines.append("")
    lines.append("## Top Subfamilies")
    lines.append("")
    for subfamily, count in subfamily_counter.most_common(25):
        lines.append(f"- `{subfamily}`: {count}")
    lines.append("")
    lines.append("## Top Materials")
    lines.append("")
    for material, count in material_counter.most_common(20):
        lines.append(f"- `{material}`: {count}")
    lines.append("")
    lines.append("## Frequent Technical Attributes")
    lines.append("")
    for attribute_name, pairs in frequent_attributes.items():
        lines.append(f"### `{attribute_name}`")
        for value, count in pairs[:15]:
            lines.append(f"- `{value}`: {count}")
        lines.append("")
    lines.append("## Unclassified Examples")
    lines.append("")
    for example in shortlist_examples(no_family_rows, limit=50):
        lines.append(f"- [{example['source']} | {example['sheet']}] {example['description']}")

    (DOCS_DIR / "taxonomy_seed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
