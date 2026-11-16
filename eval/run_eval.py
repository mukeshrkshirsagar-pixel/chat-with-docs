"""Evaluate retrieval (always) and end-to-end answers (with --answers and an API key).

    python -m eval.run_eval                  # retrieval only: hit@k, MRR, refusal accuracy
    python -m eval.run_eval --answers        # + keyword check, citation validity (uses the LLM)
    python -m eval.run_eval --sweep          # print the dense-score distribution to calibrate MIN_SIMILARITY

Answerable questions carry an `evidence` string: retrieval "hits" if any top-k chunk contains it.
Unanswerable ones (doc = null) should be refused by the similarity gate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from app.config import Settings
from app.embeddings import build_embedder
from app.ingestion import ingest
from app.llm import build_llm
from app.rag import RagPipeline
from app.retrieval import Retriever
from app.store import Store

ROOT = Path(__file__).parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also run the LLM and check answers")
    ap.add_argument("--sweep", action="store_true", help="show the score distribution per class")
    args = ap.parse_args()

    cfg = Settings()
    embedder = build_embedder(cfg.embedder, cfg.embedding_model)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = Store(str(Path(tmp) / "eval.db"), embedder.dim)
        for f in sorted((ROOT / "sample_docs").iterdir()):
            ingest(store, embedder, cfg, f.name, f.read_bytes())
        retriever = Retriever(store, embedder, cfg.candidates, cfg.top_k)
        cases = [
            json.loads(line)
            for line in (ROOT / "eval" / "golden.jsonl").read_text("utf-8").splitlines()
            if line
        ]

        hits_at = {1: 0, 3: 0, cfg.top_k: 0}
        mrr, n_ans, correct_refusals, n_unans = 0.0, 0, 0, 0
        ans_scores, unans_scores = [], []
        for c in cases:
            r = retriever.retrieve(c["q"])
            if c["doc"] is None:
                n_unans += 1
                unans_scores.append(r.top_dense)
                if args.sweep:
                    print(f"  unanswerable {r.top_dense:.3f}  {c['q']}")
                refused = r.top_dense < cfg.min_similarity
                correct_refusals += refused
                if not refused:
                    print(f"  [gate miss] answerable-looking: {c['q']!r} (top_dense={r.top_dense:.3f})")
                continue
            n_ans += 1
            ans_scores.append(r.top_dense)
            rank = next((i for i, h in enumerate(r.hits, 1) if c["evidence"] in h.chunk.text), None)
            if rank:
                mrr += 1 / rank
                for k in hits_at:
                    hits_at[k] += rank <= k
            else:
                print(f"  [retrieval miss] {c['q']!r}")
            if r.top_dense < cfg.min_similarity:
                print(f"  [false refusal] {c['q']!r} (top_dense={r.top_dense:.3f})")

        print(
            f"\nembedder={type(embedder).__name__}:{cfg.embedding_model}  min_similarity={cfg.min_similarity}"
        )
        for k, v in hits_at.items():
            print(f"hit@{k}: {v}/{n_ans} = {v / n_ans:.2f}")
        print(f"MRR: {mrr / n_ans:.3f}")
        print(f"refusal accuracy (unanswerable): {correct_refusals}/{n_unans}")
        if args.sweep:
            print(
                f"top dense cosine, answerable:   min={min(ans_scores):.3f} mean={sum(ans_scores) / len(ans_scores):.3f}"
            )
            print(
                f"top dense cosine, unanswerable: max={max(unans_scores):.3f} mean={sum(unans_scores) / len(unans_scores):.3f}"
            )

        if args.answers:
            if cfg.resolved_provider is None:
                raise SystemExit("--answers needs ANTHROPIC_API_KEY or XAI_API_KEY")
            pipe = RagPipeline(cfg, store, retriever, build_llm(cfg))
            kw_ok = cite_ok = answered = 0
            for c in cases:
                if c["doc"] is None:
                    continue
                answered += 1
                text, done = "", {}
                for ev in pipe.answer_stream(c["q"]):
                    if ev["event"] == "token":
                        text += ev["data"]
                    elif ev["event"] == "done":
                        done = ev["data"]
                # models emit typographic spaces/hyphens; normalise before substring matching
                norm = text.lower().replace(" ", " ").replace(" ", " ").replace("‑", "-").replace("‐", "-")
                ok = all(k.lower() in norm for k in c["keywords"])
                cited = bool(done.get("cited")) and not done.get("invalid_citations")
                kw_ok += ok
                cite_ok += cited
                if not (ok and cited):
                    print(
                        f"  [answer issue] {c['q']!r} keywords_ok={ok} cited_ok={cited}\n    {text[:200]!r}"
                    )
            print(f"answer keyword accuracy: {kw_ok}/{answered}   valid citations: {cite_ok}/{answered}")

        store.close()


if __name__ == "__main__":
    main()
