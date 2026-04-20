#!/usr/bin/env python3
"""
Export-Skript fuer Oleksandras Dropout-Analyse (UZ v4+1).

Zieht alle usbekischen Kandidaten mit bekanntem Outcome aus der GLOBOGATE
External API (via den Netlify-Proxy), klassifiziert Region und schreibt
eine CSV, die Oleksandra direkt fuer ihre Analyse nutzen kann.

Verwendung:
    cd /Users/alexanderrhode/Documents/globogate-lineup-screener
    python3 scripts/export-uz-dropout-dataset.py

Output:
    exports/uz_dropout_dataset_<YYYY-MM-DD>.csv
"""

import csv
import json
import os
import sys
import urllib.request
from datetime import datetime, date
from pathlib import Path
from typing import Optional

PROXY_URL = (
    "https://globogate-lineup-screener.netlify.app/.netlify/functions/"
    "api-proxy?endpoint=/persons?state=all"
)

# UZ-Region-Klassifikation — identisch zu app.js (UZ_REGION_KEYWORDS)
UZ_REGION_KEYWORDS = {
    "Tashkent": ["tashkent", "toshkent"],
    "Fergana": ["fergana", "fargona", "ferghana"],
    "Andijan": ["andijan", "andijon"],
    "Namangan": ["namangan"],
    "Samarkand": ["samarkand", "samarqand"],
    "Bukhara": ["bukhara", "buxoro"],
    "Kashkadarya": ["kashkadarya", "qashqadaryo", "karshi"],
    "Surkhandarya": ["surkhandarya", "surxondaryo", "termez"],
    "Khorezm": ["khorezm", "xorazm", "urgench"],
    "Navoi": ["navoi", "navoiy"],
    "Jizzakh": ["jizzakh", "jizzax"],
    "Sirdarya": ["sirdarya", "syrdarya", "guliston"],
    "Karakalpakstan": ["karakalpakstan", "nukus", "qoraqalpog"],
}


def classify_region_uz(city: str) -> str:
    if not city:
        return "Other"
    c = city.lower().strip()
    for region, keywords in UZ_REGION_KEYWORDS.items():
        if any(kw in c for kw in keywords):
            return region
    return "Other"


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def years_between(d_from: Optional[date], d_to: Optional[date]) -> Optional[float]:
    if not d_from or not d_to:
        return None
    return round((d_to - d_from).days / 365.25, 1)


def main():
    print("==> Fetching API data via Netlify proxy...", file=sys.stderr)
    with urllib.request.urlopen(PROXY_URL, timeout=90) as resp:
        data = json.load(resp)

    # API returns flat list (or sometimes dict-of-dicts)
    if isinstance(data, dict):
        if "original" in data:
            data = data["original"]
        if isinstance(data, dict):
            data = list(data.values())

    total = len(data)
    print(f"    {total:,} Personen insgesamt geladen.", file=sys.stderr)

    # Filter: UZ mit bekanntem Outcome
    uz_with_outcome = [
        p for p in data
        if p.get("country") == "Uzbekistan"
        and (p.get("arrival_fin") or p.get("dropout_date_fin"))
    ]
    print(f"    {len(uz_with_outcome):,} UZ-Kandidaten mit Outcome.", file=sys.stderr)

    # Outcome-Buckets
    arrived = [p for p in uz_with_outcome if p.get("arrival_fin")]
    dropped_before = [
        p for p in uz_with_outcome
        if p.get("dropout_date_fin") and not p.get("arrival_fin")
    ]
    dropped_after = [
        p for p in uz_with_outcome
        if p.get("dropout_date_fin") and p.get("arrival_fin")
    ]
    print(
        f"    Outcome-Breakdown: "
        f"arrived={len(arrived)}, "
        f"dropped_before={len(dropped_before)}, "
        f"dropped_after={len(dropped_after)}",
        file=sys.stderr,
    )

    # Output-Verzeichnis
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "exports"
    out_dir.mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = out_dir / f"uz_dropout_dataset_{today}.csv"

    columns = [
        "person_id",
        "reference_id",
        "name",
        "gender",
        "birth_date",
        "age_at_arrival_or_dropout",
        "marital_status",
        "years_experience",
        "origin_city",
        "current_city",
        "region",
        "hospital_type",
        "category",
        "icu_category",
        "arrival_date",
        "dropout_date",
        "dropout_reason",
        "outcome",
    ]

    rows_written = 0
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(columns)
        for p in uz_with_outcome:
            birth = parse_date(p.get("person_birth_date"))
            arrival = parse_date(p.get("arrival_fin"))
            dropout = parse_date(p.get("dropout_date_fin"))
            # Age at outcome event (prefer arrival, else dropout)
            event_date = arrival or dropout
            age = years_between(birth, event_date) if birth and event_date else ""

            if arrival and dropout:
                outcome = "dropped_after_arrival"
            elif arrival:
                outcome = "arrived"
            else:
                outcome = "dropped_before_arrival"

            current_city = p.get("person_city") or ""
            birth_place = p.get("person_birth_place") or ""
            # Fuer Herkunftsregion: zuerst birth_place, dann person_city als Fallback.
            # birth_place ist stabil (aendert sich nicht bei Umzug nach DE).
            region_from_birth = classify_region_uz(birth_place)
            region_from_city = classify_region_uz(current_city)
            # Wenn birth_place eine UZ-Region ergibt, nimm die. Sonst die aus city.
            region = region_from_birth if region_from_birth != "Other" else region_from_city
            # Origin-City fuer die Spalte: bevorzugt birth_place, sonst current_city
            origin_city = birth_place or current_city

            writer.writerow([
                p.get("person_id", ""),
                p.get("reference_id", ""),
                p.get("person_name", ""),
                p.get("person_gender", ""),
                p.get("person_birth_date", "") or "",
                age if age != "" else "",
                p.get("marital_status", "") or "",
                p.get("total_years_experience_rn", "") or "",
                origin_city,
                current_city,
                region,
                p.get("person_hospital", "") or "",
                p.get("person_categories", "") or "",
                p.get("person_icu_category", "") or "",
                p.get("arrival_fin", "") or "",
                p.get("dropout_date_fin", "") or "",
                p.get("dropout_reason", "") or "",
                outcome,
            ])
            rows_written += 1

    print(f"\nFertig: {out_path}")
    print(f"    {rows_written:,} Zeilen geschrieben (+ Header).")
    print(f"    Semikolon-getrennt, UTF-8 BOM (Excel-DE-kompatibel).")


if __name__ == "__main__":
    main()
