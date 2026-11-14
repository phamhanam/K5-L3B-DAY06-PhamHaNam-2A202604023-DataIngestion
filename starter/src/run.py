import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from .ingestion import load_pdf
from .chunking import fixed_size, by_paragraph
from .embeddings import MyEmbedder
from .store import PgVectorStore
from .evaluation import score_question, evaluate_strategy

load_dotenv()

MANIFEST_PATH = Path("data/manifest.json")
BENCHMARK_PATH = Path("data/benchmark.json")
OUT_DIR = Path("outputs/checkpoint_3_30")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    # 1. Load corpus
    pages, issue_log = load_pdf(
        path=manifest["local_path"],
        document_id=manifest["source_id"],
        source=manifest["url_or_path"],
        allowed_pages=set(manifest["page_ids"]),
    )

    # 2. Build chunks for A, B, C
    chunks_a = [c for p in pages for c in fixed_size(p, size=500, overlap=50)]
    chunks_b = [c for p in pages for c in by_paragraph(p, max_size=800)]
    chunks_c = chunks_b  # Strategy C uses same chunks as B, plus SQL filter when supplied

    # 3. Embed + store
    embedder = MyEmbedder()
    store = PgVectorStore(os.environ["DATABASE_URL"], embedder.dimensions)
    store.create_schema()

    strategy_chunks = {"A": chunks_a, "B": chunks_b, "C": chunks_c}
    ingest_stats = {}

    for strat, chunks in strategy_chunks.items():
        t0 = time.perf_counter()
        vecs = embedder.embed_documents([c.text for c in chunks])
        store.upsert(strat, chunks, vecs)
        ingest_stats[strat] = {
            "chunk_count": len(chunks),
            "ingest_sec": round(time.perf_counter() - t0, 2),
        }

    # 4. Run dev + test queries across A, B, C
    query_vecs = {item["id"]: embedder.embed_query(item["question"]) for item in benchmark}
    metrics_report = {
        "selection_criterion_locked_before_test": (
            "1. Evidence Hit@3 (dev, verified on test) -> "
            "2. Grounding quality -> "
            "3. Chunk coherence & ingest/query cost"
        ),
        "embedder": {"model_name": embedder.model_name, "dimensions": embedder.dimensions},
        "ingest_stats": ingest_stats,
        "strategies": {},
    }

    all_traces = {}
    for strat in ("A", "B", "C"):
        traces = []
        for item in benchmark:
            flt = item.get("filter") if strat == "C" else None
            retrieved = store.search(strat, query_vecs[item["id"]], top_k=3, filters=flt)
            traces.append(score_question(item, retrieved))

        all_traces[strat] = traces
        (OUT_DIR / f"traces_{strat}.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
        metrics_report["strategies"][strat] = evaluate_strategy(traces)

    (OUT_DIR / "metrics_summary.json").write_text(
        json.dumps(metrics_report, indent=2), encoding="utf-8"
    )

    # 5. Generate pre-filled manual audit worksheet (10 fixed page positions + 5 test questions)
    sampled_pages = sorted(manifest["page_ids"])[:: max(1, len(manifest["page_ids"]) // 10)][:10]
    audit_sheet = {
        "bias_note": "Scored manually by benchmark author (self-evaluation bias risk). No LLM judge used.",
        "1_chunk_coherence_0_to_2": {
            "sampled_pages_picked_before_scores": sampled_pages,
            "rubric": "0 = cut mid-idea, 1 = partial, 2 = self-contained",
            "scores": {
                strat: {
                    str(p): {
                        "first_chunk_preview": next(
                            (c.text[:220] for c in strategy_chunks[strat] if c.page == p), ""
                        ),
                        "score": None,
                    }
                    for p in sampled_pages
                }
                for strat in ("A", "B", "C")
            },
        },
        "2_metadata_usefulness_5_test": {
            "rubric": "Verify source, page, and filter fields vs corpus; report correct_and_complete/15 and filter_compliance",
            "items": [
                {
                    "id": t["id"],
                    "strategy": strat,
                    "filter_expected": t["filter"],
                    "retrieved_meta": [
                        {"chunk_id": r["chunk_id"], "page": r["page"], "metadata": r["metadata"]}
                        for r in t["retrieved"]
                    ],
                    "correct_and_complete_out_of_3": None,
                    "filter_compliant_out_of_3_or_NA": None if strat != "C" else None,
                }
                for strat in ("A", "B", "C")
                for t in all_traces[strat]
                if t["split"] == "test"
            ],
        },
        "3_grounding_quality_0_to_2": {
            "rubric": "0 = no evidence, 1 = incomplete, 2 = sufficient + traceable (retrieval only, not LLM answer)",
            "items": [
                {
                    "id": t["id"],
                    "strategy": strat,
                    "question": t["question"],
                    "gold_page": t["gold_page"],
                    "evidence_quote": t["evidence_quote"],
                    "top3_pages": [r["page"] for r in t["retrieved"]],
                    "score_0_to_2": None,
                }
                for strat in ("A", "B", "C")
                for t in all_traces[strat]
                if t["split"] == "test"
            ],
        },
    }
    (OUT_DIR / "manual_audit_sheet.json").write_text(
        json.dumps(audit_sheet, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics_report, indent=2))


if __name__ == "__main__":
    main()
    