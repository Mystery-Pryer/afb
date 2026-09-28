#!/usr/bin/env python3
"""Deterministic bounded UniMorph Gulf Arabic gloss lexicon dry run.

Processes seven pinned physical TSV rows chosen to exercise:
- physical row 1 as data rather than a header;
- all observed POS tags (N, ADJ, V);
- the two source blank-gloss rows;
- one exact lexical-content duplicate pair whose physical-line identities must
  remain distinct.

Writes a temporary Parquet and a no-source-text validation manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SNAPSHOT_ID = "SNP-000004"
RESOURCE_ID = "RES-000004"
ARTIFACT_ID = "ART-000501"
SOURCE_OBJECT = "unimorph_afb.gloss.tsv"
SOURCE_REPO_PATH = Path("source/legacy_relocated/gulf/mirrored/unimorph_afb/afb.gloss")
EXPECTED_SHA256 = "c7f6f1aa8070c9811be9c69c4f5ed33df70b86c3d29d6c78b177322abc298eb4"
EXPECTED_BYTES = 163928
EXPECTED_ROWS = 6707
SELECTED_LINES = [1, 3, 11, 12, 33, 4931, 6615]
EXPECTED_POS_COUNTS = {"ADJ": 1, "N": 4, "V": 2}
EXPECTED_BLANK_GLOSS_LINES = {4931, 6615}
DUPLICATE_CONTENT_LINES = (12, 33)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def foundation_id(physical_line_number: int) -> str:
    preimage = SNAPSHOT_ID + ARTIFACT_ID + str(physical_line_number)
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def read_source(source: Path) -> tuple[list[dict], dict]:
    if source.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"source byte-size mismatch: {source.stat().st_size} != {EXPECTED_BYTES}"
        )
    source_sha = sha256_file(source)
    if source_sha != EXPECTED_SHA256:
        raise RuntimeError(f"source sha256 mismatch: {source_sha}")

    text = source.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if len(lines) != EXPECTED_ROWS:
        raise RuntimeError(f"physical row count mismatch: {len(lines)} != {EXPECTED_ROWS}")

    selected: list[dict] = []
    for line_no in SELECTED_LINES:
        raw_line = lines[line_no - 1]
        fields = raw_line.split("\t")
        if len(fields) != 3:
            raise RuntimeError(f"line {line_no}: expected exactly 3 TSV fields")
        lemma, pos, english_gloss = fields
        selected.append(
            {
                "foundation_record_id": foundation_id(line_no),
                "snapshot_id": SNAPSHOT_ID,
                "resource_id": RESOURCE_ID,
                "artifact_id": ARTIFACT_ID,
                "source_object": SOURCE_OBJECT,
                "source_record_locator": (
                    "git:source/legacy_relocated/gulf/mirrored/unimorph_afb/"
                    f"afb.gloss#line={line_no}"
                ),
                "source_split": None,
                "source_record_id": f"{SOURCE_OBJECT}:{line_no}",
                "record_family": "lexicon_entry",
                "source_attributes": {"physical_line_number": str(line_no)},
                "lemma": lemma,
                "part_of_speech": pos,
                "english_gloss": english_gloss,
            }
        )

    return selected, {
        "source_sha256": source_sha,
        "source_bytes": source.stat().st_size,
        "physical_data_rows": len(lines),
    }


def write_parquet(rows: list[dict], output: Path) -> None:
    schema = pa.schema(
        [
            pa.field("foundation_record_id", pa.string(), nullable=False),
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("resource_id", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string(), nullable=False),
            pa.field("source_object", pa.string(), nullable=False),
            pa.field("source_record_locator", pa.string(), nullable=False),
            pa.field("source_split", pa.string(), nullable=True),
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("record_family", pa.string(), nullable=False),
            pa.field("source_attributes", pa.map_(pa.string(), pa.string()), nullable=False),
            pa.field("lemma", pa.string(), nullable=False),
            pa.field("part_of_speech", pa.string(), nullable=False),
            pa.field("english_gloss", pa.string(), nullable=False),
        ]
    )
    normalized = []
    for row in rows:
        item = dict(row)
        item["source_attributes"] = list(item["source_attributes"].items())
        normalized.append(item)
    table = pa.Table.from_pylist(normalized, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd", use_dictionary=True)


def validate(rows: list[dict], output: Path, source_meta: dict) -> dict:
    table = pq.read_table(output)
    out = table.to_pylist()
    if table.num_rows != len(SELECTED_LINES):
        raise RuntimeError(
            f"expected {len(SELECTED_LINES)} rows, found {table.num_rows}"
        )
    if len({row["foundation_record_id"] for row in out}) != len(SELECTED_LINES):
        raise RuntimeError("foundation IDs are not unique")

    output_line_numbers = [
        int(dict(row["source_attributes"])["physical_line_number"]) for row in out
    ]
    if output_line_numbers != SELECTED_LINES:
        raise RuntimeError(f"physical-line selection changed: {output_line_numbers}")

    pos_counts = dict(sorted(Counter(row["part_of_speech"] for row in out).items()))
    if pos_counts != EXPECTED_POS_COUNTS:
        raise RuntimeError(f"POS accounting mismatch: {pos_counts}")

    blank_gloss_lines = {
        int(dict(row["source_attributes"])["physical_line_number"])
        for row in out
        if row["english_gloss"] == ""
    }
    if blank_gloss_lines != EXPECTED_BLANK_GLOSS_LINES:
        raise RuntimeError(f"blank-gloss preservation mismatch: {blank_gloss_lines}")

    for original, emitted in zip(rows, out, strict=True):
        for field in (
            "foundation_record_id",
            "source_record_locator",
            "source_record_id",
            "lemma",
            "part_of_speech",
            "english_gloss",
        ):
            if original[field] != emitted[field]:
                raise RuntimeError(f"{field} changed after Parquet round trip")

    by_line = {
        int(dict(row["source_attributes"])["physical_line_number"]): row for row in out
    }
    left = by_line[DUPLICATE_CONTENT_LINES[0]]
    right = by_line[DUPLICATE_CONTENT_LINES[1]]
    lexical_fields = ("lemma", "part_of_speech", "english_gloss")
    if tuple(left[f] for f in lexical_fields) != tuple(right[f] for f in lexical_fields):
        raise RuntimeError("pinned duplicate lexical rows are no longer identical")
    if left["foundation_record_id"] == right["foundation_record_id"]:
        raise RuntimeError("legitimate duplicate rows collapsed to one Foundation ID")

    lexical_collection = "\n".join(
        f"{row['lemma']}\t{row['part_of_speech']}\t{row['english_gloss']}"
        for row in out
    )
    locator_collection = "\n".join(row["source_record_locator"] for row in out)
    schema_text = str(table.schema)

    return {
        "checkpoint": "8C-1F",
        "source_snapshot": SNAPSHOT_ID,
        "record_family": "lexicon_entry",
        "subset_rule": (
            "physical lines 1, 3, 11, 12, 33, 4931, 6615; tests row-1-as-data, "
            "all POS tags, two blank glosses, and a legitimate exact-content duplicate pair"
        ),
        "canonical_row_count": len(out),
        "selected_physical_lines": SELECTED_LINES,
        "source_physical_data_rows_verified": source_meta["physical_data_rows"],
        "pos_row_counts": pos_counts,
        "blank_gloss_row_count": len(blank_gloss_lines),
        "duplicate_content_pair_preserved_as_distinct_records": True,
        "unique_foundation_record_ids": len(out),
        "identity_preimage_rule": (
            "sha256(snapshot_id + artifact_id + physical_line_number)"
        ),
        "source_sha256": source_meta["source_sha256"],
        "source_bytes": source_meta["source_bytes"],
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "output_parquet_sha256": sha256_file(output),
        "lexical_collection_sha256": hashlib.sha256(
            lexical_collection.encode("utf-8")
        ).hexdigest(),
        "source_locator_collection_sha256": hashlib.sha256(
            locator_collection.encode("utf-8")
        ).hexdigest(),
        "compression": "zstd",
        "checks": {
            "schema_readable": True,
            "bounded_subset_selection": True,
            "first_physical_row_preserved_as_data": True,
            "stable_unique_record_ids": True,
            "source_provenance": True,
            "lemma_preservation": True,
            "pos_preservation": True,
            "english_gloss_preservation": True,
            "blank_gloss_preserved_as_empty_string": True,
            "legitimate_duplicate_rows_distinguished_by_identity": True,
            "source_sha256": True,
            "source_row_accounting": True,
        },
        "raw_source_text_emitted_in_manifest": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=SOURCE_REPO_PATH)
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    rows, source_meta = read_source(args.source)
    write_parquet(rows, args.output_parquet)
    manifest = validate(rows, args.output_parquet, source_meta)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
