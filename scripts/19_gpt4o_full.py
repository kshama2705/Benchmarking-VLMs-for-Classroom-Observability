"""
Full 1,784-clip GPT-4o zero-shot inference with explicit refusal tracking.

Fixes the CVPR-version bug where unparseable outputs silently fell back to L2,
inflating P3 accuracy. Now we record:
  - had_rating: whether a rating digit was extracted
  - is_refusal: whether the response looks like a safety/identity refusal
  - predicted: integer 0-3 if had_rating else empty

Metrics are computed two ways for transparency:
  - excluding refusals (proper)
  - with refusals counted as L2 fallback (replicates the CVPR bug, for direct comparison)

Output:
  results/full/gpt4o_full_p{N}.csv          (clip_id, predicted, ground_truth, is_refusal)
  results/full/gpt4o_full_p{N}_raw.csv      (raw answers for inspection)
  results/full/gpt4o_full_metrics.json
"""

import os, csv, json, base64, time, re, argparse
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE, "full_test.csv")
OUT = os.path.join(BASE, "results", "full")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

PROMPTS = {
    1: ("minimal",
        "Rate this student's engagement level:\n"
        "0 = not engaged\n1 = barely engaged\n2 = engaged\n3 = highly engaged\n\n"
        "Answer with just the number (0, 1, 2, or 3)."),
    2: ("rubric",
        "You are an educational observer. Assess this student's engagement "
        "based on their facial expression and body language:\n\n"
        "0 = Distracted, looking away, disinterested, sleepy\n"
        "1 = Passively present but not focused, neutral expression\n"
        "2 = Attentive and following along, maintaining eye contact with screen\n"
        "3 = Actively focused, leaning in, alert, showing curiosity\n\n"
        "Respond with only the number (0, 1, 2, or 3)."),
    3: ("chain_of_thought",
        "Analyze this student watching an educational video.\n\n"
        "Step 1: Describe the student's facial expression in one sentence.\n"
        "Step 2: Describe their body posture in one sentence.\n"
        "Step 3: Based on your observations, rate their engagement level:\n"
        "  0 = not engaged at all\n  1 = barely engaged\n  2 = engaged\n  3 = highly engaged\n\n"
        "Format your response as:\n"
        "Expression: ...\nPosture: ...\nRating: [0-3]"),
}

REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "cannot identify", "can't identify", "cannot determine",
    "cannot assess", "cannot rate", "cannot make assumptions",
    "i don't have", "i do not have",
    "personally identifiable", "real person", "specific individual",
    "actual student", "minor",
    "i apologize",
    "as an ai",
]


def parse(response_text):
    txt = response_text.strip()
    txt_low = txt.lower()
    refusal_hit = any(p in txt_low for p in REFUSAL_PHRASES)
    rating_match = re.search(r"Rating:\s*(\d)", txt)
    if rating_match:
        v = int(rating_match.group(1))
        if 0 <= v <= 3:
            return v, True, refusal_hit  # had_rating=True regardless of refusal text
    digits = re.findall(r"\b([0-3])\b", txt)
    if digits:
        return int(digits[-1]), True, refusal_hit
    # No digit found
    return None, False, True  # treat any unparseable as refusal


def metrics(yt, yp):
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa_quadratic": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_linear": float(cohen_kappa_score(yt, yp, weights="linear")),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mse": float(mean_squared_error(yt, yp)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def boot_kappa(yt, yp, n=1000):
    if len(yt) == 0:
        return [0.0, 0.0]
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def run_prompt(client, prompt_id, prompt_text, samples):
    print(f"\n=== P{prompt_id} ({len(samples)} clips) ===", flush=True)
    rows = []
    refusal_count = 0
    for i, s in enumerate(samples):
        with open(s["frame_path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                        ],
                    }],
                    max_tokens=150,
                    temperature=0.0,
                )
                answer = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"  retry {attempt+1} for {s['clip_id']}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            answer = ""
        pred, had_rating, is_refusal = parse(answer)
        if is_refusal or not had_rating:
            refusal_count += 1
        rows.append({
            "clip_id": s["clip_id"],
            "ground_truth": int(s["engagement"]),
            "predicted": pred if pred is not None else "",
            "is_refusal": int(is_refusal or not had_rating),
            "raw_answer": answer,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(samples)}  refusals_so_far={refusal_count}  last='{answer[:50]}' -> {pred}", flush=True)
        time.sleep(0.4)

    out_csv = os.path.join(OUT, f"gpt4o_full_p{prompt_id}.csv")
    raw_csv = os.path.join(OUT, f"gpt4o_full_p{prompt_id}_raw.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "predicted", "ground_truth", "is_refusal"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["clip_id", "predicted", "ground_truth", "is_refusal"]})
    with open(raw_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "raw_answer", "predicted", "ground_truth", "is_refusal"])
        w.writeheader()
        w.writerows(rows)

    # Compute metrics two ways
    yt_all = np.array([r["ground_truth"] for r in rows])
    yp_full = np.array([r["predicted"] if r["predicted"] != "" else 2 for r in rows])  # CVPR-bug-style L2 fallback
    keep = np.array([r["predicted"] != "" for r in rows])
    yt_kept = yt_all[keep]
    yp_kept = np.array([int(r["predicted"]) for r in rows if r["predicted"] != ""])

    m_kept = metrics(yt_kept, yp_kept) if len(yt_kept) > 0 else {"n": 0}
    if len(yt_kept) > 0:
        m_kept["kappa_q_ci95"] = boot_kappa(yt_kept, yp_kept)
    m_full = metrics(yt_all, yp_full)
    m_full["kappa_q_ci95"] = boot_kappa(yt_all, yp_full)

    summary = {
        "n_total": len(rows),
        "n_refusals": refusal_count,
        "refusal_rate": refusal_count / len(rows),
        "metrics_excluding_refusals": m_kept,
        "metrics_with_L2_fallback": m_full,
    }
    print(f"P{prompt_id} done: refusals={refusal_count}/{len(rows)} ({100*refusal_count/len(rows):.1f}%)", flush=True)
    if len(yt_kept) > 0:
        print(f"  excluding refusals: κ_q={m_kept['kappa_quadratic']:.3f} acc={m_kept['accuracy']:.3f} n={m_kept['n']}", flush=True)
    print(f"  L2 fallback (CVPR bug): κ_q={m_full['kappa_quadratic']:.3f} acc={m_full['accuracy']:.3f}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="1,2,3")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    from openai import OpenAI
    client = OpenAI()

    samples = []
    with open(INPUT) as f:
        for r in csv.DictReader(f):
            samples.append(r)
    print(f"Loaded {len(samples)} test clips", flush=True)

    all_summaries = {}
    for pid in [int(p) for p in args.prompts.split(",")]:
        name, prompt = PROMPTS[pid]
        all_summaries[f"P{pid}"] = run_prompt(client, pid, prompt, samples)

    with open(os.path.join(OUT, "gpt4o_full_metrics.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT, 'gpt4o_full_metrics.json')}", flush=True)


if __name__ == "__main__":
    main()
