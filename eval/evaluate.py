"""
Evaluation harness for IITB Insti-Assist.

Measures, on a labelled question set (eval/eval_dataset.json):

  RETRIEVAL     Hit@k, MRR                -- does the correct source doc appear in top-k?
  THRESHOLD     MIN_SCORE ablation        -- refusal tradeoff (score-gate only, no LLM)
  REFUSAL       end-to-end (both guards)  -- via pipeline.answer() on out-of-scope Qs
  FAITHFULNESS  LLM-as-judge              -- is each answer supported by retrieved context?
  CORRECTNESS   LLM-as-judge vs gold      -- only for items where answer_key is filled

Usage
-----
Place this file + eval_dataset.json in an `eval/` folder inside the repo, then:

    # retrieval + threshold ablation only  (NO API key needed, ~10s)
    python eval/evaluate.py --mode retrieval

    # full run: also end-to-end refusal + faithfulness (needs GEMINI_API_KEY)
    # free tier is 5 req/min, so this paces itself (~12-15 min). Speed up with --rpm.
    python eval/evaluate.py --mode full --rpm 5

Outputs: prints a summary and writes eval/eval_results.json + eval/eval_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

# --- make the repo root importable whether this lives in root or eval/ ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "eval" else HERE
sys.path.insert(0, str(ROOT))

import config                                    # noqa: E402
from rag.vectorstore import VectorStore          # noqa: E402
from rag.pipeline import RAGPipeline             # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_dataset() -> list[dict]:
    data = json.loads((HERE / "eval_dataset.json").read_text(encoding="utf-8"))
    return data["items"]


def source_match(retrieved: str, expected: str) -> bool:
    """Robust filename match: case-insensitive, basename, substring either way."""
    r = Path(str(retrieved)).name.lower().strip()
    e = Path(str(expected)).name.lower().strip()
    if not e:
        return False
    return e in r or r in e


def first_hit_rank(sources: list[str], expected: str) -> int | None:
    for i, s in enumerate(sources, start=1):
        if source_match(s, expected):
            return i
    return None


def is_refusal(answer: str) -> bool:
    a = (answer or "").strip().lower()
    ref = config.REFUSAL.strip().lower()
    return a.startswith(ref[:15]) or ref[:15] in a


# --------------------------------------------------------------------------- #
# RETRIEVAL metrics  (no LLM)
# --------------------------------------------------------------------------- #
def eval_retrieval(pipe: RAGPipeline, items: list[dict], k: int) -> dict:
    in_scope = [it for it in items if it["type"] == "in_scope"]
    hits, rrs, per_item = 0, [], []
    for it in in_scope:
        chunks = pipe.retrieve(it["question"], k=k)
        srcs = [c["source"] for c in chunks]
        rank = first_hit_rank(srcs, it["expected_source"])
        hits += 1 if rank else 0
        rrs.append(1.0 / rank if rank else 0.0)
        per_item.append({"id": it["id"], "expected": it["expected_source"],
                         "retrieved": srcs, "hit_rank": rank})
    n = len(in_scope)
    return {"k": k, "n": n,
            "hit_at_k": round(hits / n, 3) if n else 0.0,
            "mrr": round(mean(rrs), 3) if rrs else 0.0,
            "per_item": per_item}


# --------------------------------------------------------------------------- #
# THRESHOLD ablation over MIN_SCORE  (score-gate only -> no LLM)
# --------------------------------------------------------------------------- #
def eval_threshold_ablation(pipe: RAGPipeline, items: list[dict],
                            k: int, thresholds: list[float]) -> list[dict]:
    top_scores = {}
    for it in items:
        chunks = pipe.retrieve(it["question"], k=k)
        top_scores[it["id"]] = (it["type"], chunks[0]["score"] if chunks else 0.0)

    rows = []
    for thr in thresholds:
        oos_correct = oos_total = false_refuse = ans_total = 0
        for typ, top in top_scores.values():
            refused = top < thr
            if typ == "out_of_scope":
                oos_total += 1
                oos_correct += 1 if refused else 0
            else:
                ans_total += 1
                false_refuse += 1 if refused else 0
        rows.append({"min_score": thr,
                     "oos_refusal_acc": round(oos_correct / oos_total, 3) if oos_total else 0.0,
                     "false_refusal_rate": round(false_refuse / ans_total, 3) if ans_total else 0.0})
    return rows


# --------------------------------------------------------------------------- #
# rate limiting for free-tier LLM APIs (Gemini free tier = 5 req/min)
# --------------------------------------------------------------------------- #
_RPM = 5
_last_call = [0.0]


def _set_rpm(rpm: int) -> None:
    global _RPM
    _RPM = max(1, rpm)


def _interval() -> float:
    return (60.0 / _RPM) * 1.10   # 10% safety margin


def _paced_call(fn, *args, **kwargs):
    """Space out + retry LLM calls so we respect the per-minute quota."""
    for attempt in range(8):
        wait = _interval() - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            _last_call[0] = time.time()
            return fn(*args, **kwargs)
        except Exception as e:                       # noqa: BLE001
            msg = str(e).lower()
            transient = ("resource_exhausted" in msg or "429" in msg
                         or "quota" in msg or "rate" in msg or "exhausted" in msg)
            if transient and attempt < 7:
                back = 20 * (attempt + 1)
                print(f"   [rate-limited; backing off {back}s, retry {attempt + 1}/7]")
                time.sleep(back)
                continue
            raise
    raise RuntimeError("gave up after repeated rate-limit errors")


# --------------------------------------------------------------------------- #
# LLM-as-judge + end-to-end answer eval  (needs API key)
# --------------------------------------------------------------------------- #
def judge(prompt: str) -> str:
    from rag import llm
    return _paced_call(llm.generate, prompt,
                       system="You are a strict, terse evaluation judge.").strip()


def eval_answers(pipe: RAGPipeline, items: list[dict], k: int) -> dict:
    in_scope = [it for it in items if it["type"] == "in_scope"]
    oos = [it for it in items if it["type"] == "out_of_scope"]
    faith_scores, correct_scores, per_item = [], [], []
    false_refusals = 0

    # ---- in-scope: faithfulness, correctness, false-refusal (end-to-end) ----
    for it in in_scope:
        res = _paced_call(pipe.answer, it["question"], k=k)
        ans = res["answer"]
        refused = is_refusal(ans) or not res["grounded"]
        if refused:
            false_refusals += 1

        ctx = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(res["sources"]))
        f_prompt = (
            "Given the CONTEXT and an ANSWER, reply with exactly one word: "
            "SUPPORTED if every claim in the answer is backed by the context, "
            "or UNSUPPORTED otherwise. If the answer is a refusal, reply SUPPORTED.\n\n"
            f"CONTEXT:\n{ctx}\n\nANSWER:\n{ans}\n\nVerdict:"
        )
        f_verdict = judge(f_prompt).upper()
        faith = 1.0 if "SUPPORTED" in f_verdict and "UNSUPPORTED" not in f_verdict else 0.0
        faith_scores.append(faith)

        corr = None
        if it.get("answer_key"):
            c_prompt = (
                "Compare a MODEL ANSWER against the GOLD ANSWER for the QUESTION. "
                "Reply with exactly one word: CORRECT (matches key facts), "
                "PARTIAL (some key facts right, some missing/wrong), or WRONG.\n\n"
                f"QUESTION: {it['question']}\nGOLD: {it['answer_key']}\n"
                f"MODEL: {ans}\n\nVerdict:"
            )
            v = judge(c_prompt).upper()
            if "WRONG" in v or "INCORRECT" in v:
                corr = 0.0
            elif "PARTIAL" in v:
                corr = 0.5
            elif "CORRECT" in v:
                corr = 1.0
            else:
                corr = 0.0
            correct_scores.append(corr)

        per_item.append({"id": it["id"], "faithful": faith, "correct": corr,
                         "refused": refused, "answer": ans[:300]})

    # ---- out-of-scope: end-to-end refusal accuracy (both guards) ----
    oos_refused = 0
    for it in oos:
        res = _paced_call(pipe.answer, it["question"], k=k)
        refused = is_refusal(res["answer"]) or not res["grounded"]
        oos_refused += 1 if refused else 0
        per_item.append({"id": it["id"], "refused": refused, "answer": res["answer"][:300]})

    return {
        "faithfulness": round(mean(faith_scores), 3) if faith_scores else 0.0,
        "correctness": round(mean(correct_scores), 3) if correct_scores else None,
        "n_correctness_scored": len(correct_scores),
        "e2e_oos_refusal_acc": round(oos_refused / len(oos), 3) if oos else 0.0,
        "e2e_false_refusal_rate": round(false_refusals / len(in_scope), 3) if in_scope else 0.0,
        "per_item": per_item,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval",
                    help="retrieval = metrics+ablation (no API); full = also LLM judge")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--rpm", type=int, default=5,
                    help="LLM requests/min budget (Gemini free tier = 5). Only affects --mode full.")
    args = ap.parse_args()
    _set_rpm(args.rpm)

    store = VectorStore.load()
    if store is None:
        print("No index found. Run `python build_index.py` first.")
        sys.exit(1)
    pipe = RAGPipeline(store)
    items = load_dataset()

    report = {"index_size": store.size, "k": args.k}
    report["retrieval"] = {kk: eval_retrieval(pipe, items, k=args.k)[kk]
                           for kk in ("k", "n", "hit_at_k", "mrr")}
    report["k_sweep"] = [{kk: r[kk] for kk in ("k", "hit_at_k", "mrr")}
                         for r in (eval_retrieval(pipe, items, k=kk) for kk in (1, 2, 4, 6, 8))]
    report["threshold_ablation"] = eval_threshold_ablation(
        pipe, items, k=args.k,
        thresholds=[0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])

    if args.mode == "full":
        report["answers"] = eval_answers(pipe, items, k=args.k)

    (HERE / "eval_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report)
    _write_markdown(report)
    print(f"\nSaved: {HERE / 'eval_results.json'} and {HERE / 'eval_report.md'}")


def _print_report(r: dict) -> None:
    print("\n===== INSTI-ASSIST EVALUATION =====")
    print(f"Index vectors: {r['index_size']}  |  top-k: {r['k']}")
    ret = r["retrieval"]
    print(f"\nRETRIEVAL (n={ret['n']} in-scope Qs):  Hit@{ret['k']} = {ret['hit_at_k']:.1%}   MRR = {ret['mrr']:.3f}")
    print("\nHit@k sweep:")
    for row in r["k_sweep"]:
        print(f"   k={row['k']:>2}   Hit@k={row['hit_at_k']:.1%}   MRR={row['mrr']:.3f}")
    print("\nMIN_SCORE ablation (score-gate only):")
    print("   thr    OOS-refusal-acc    false-refusal-rate")
    for row in r["threshold_ablation"]:
        print(f"   {row['min_score']:.2f}       {row['oos_refusal_acc']:.1%}              {row['false_refusal_rate']:.1%}")
    if "answers" in r:
        a = r["answers"]
        print("\nEND-TO-END (via pipeline.answer, both guards):")
        print(f"   Out-of-scope refusal accuracy: {a['e2e_oos_refusal_acc']:.1%}")
        print(f"   False-refusal rate (answerable Qs wrongly refused): {a['e2e_false_refusal_rate']:.1%}")
        print(f"   Faithfulness (answers supported by context): {a['faithfulness']:.1%}")
        if a["correctness"] is not None:
            print(f"   Correctness: {a['correctness']:.1%}  (scored on {a['n_correctness_scored']} Qs with gold keys)")
        else:
            print("   Correctness: not scored (no gold answer_keys filled in)")


def _write_markdown(r: dict) -> None:
    ret = r["retrieval"]
    lines = ["# Insti-Assist — Evaluation Report", "",
             f"- Index vectors: **{r['index_size']}**",
             f"- Retrieval (n={ret['n']}): **Hit@{ret['k']} = {ret['hit_at_k']:.1%}**, MRR = **{ret['mrr']:.3f}**", "",
             "## Hit@k sweep", "", "| k | Hit@k | MRR |", "|---|-------|-----|"]
    for row in r["k_sweep"]:
        lines.append(f"| {row['k']} | {row['hit_at_k']:.1%} | {row['mrr']:.3f} |")
    lines += ["", "## MIN_SCORE ablation (score-gate only)", "",
              "| MIN_SCORE | OOS refusal acc | False-refusal rate |",
              "|-----------|-----------------|--------------------|"]
    for row in r["threshold_ablation"]:
        lines.append(f"| {row['min_score']:.2f} | {row['oos_refusal_acc']:.1%} | {row['false_refusal_rate']:.1%} |")
    if "answers" in r:
        a = r["answers"]
        lines += ["", "## Answer quality (end-to-end)", "",
                  f"- Out-of-scope refusal accuracy: **{a['e2e_oos_refusal_acc']:.1%}**",
                  f"- False-refusal rate: **{a['e2e_false_refusal_rate']:.1%}**",
                  f"- Faithfulness: **{a['faithfulness']:.1%}**"]
        if a["correctness"] is not None:
            lines.append(f"- Correctness: **{a['correctness']:.1%}** (n={a['n_correctness_scored']})")
    (HERE / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
