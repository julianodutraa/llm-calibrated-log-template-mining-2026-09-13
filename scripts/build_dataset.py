"""Build the candidate merge-pair dataset used to evaluate the LLM judge.

Pipeline:
  1. Generate the synthetic, time-windowed log corpus.
  2. Run Drain online at a conservative similarity threshold (st=0.90),
     chosen because it is high enough that the dangerous status/state-enum
     merges do NOT happen automatically (verified empirically by the
     threshold sweep in tests/test_drain.py), at the cost of over-splitting
     several templates whose legitimate categorical variable (a table
     name, a tablespace name, ...) happens to fall below the threshold.
  3. For every pair of clusters that share a first-token bucket (the same
     comparison scope the online algorithm itself uses) plus one
     deliberately relaxed pair (the cosmetic-suffix case, which differs in
     token count and so is invisible to the online algorithm), compute the
     structural similarity and attach ground truth from event_groups.py.
  4. Write data/candidate_pairs.json: this is the blinded input that the
     LLM judge will see (only the two template strings), plus the hidden
     ground truth used later, purely for evaluation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logsentinel.drain import DrainParser, _seq_similarity  # noqa: E402
from logsentinel.event_groups import EVENT_GROUP, same_event_group  # noqa: E402
from logsentinel.synthetic import generate_corpus  # noqa: E402

OPERATING_THRESHOLD = 0.90


def majority_canonical(cluster, events) -> str:
    counts = Counter(events[i].canonical_id for i in cluster.log_ids)
    return counts.most_common(1)[0][0]


def main() -> None:
    events = generate_corpus()
    parser = DrainParser(similarity_threshold=OPERATING_THRESHOLD)
    for e in events:
        parser.add_log_message(e.message)

    clusters = parser.clusters
    canon_of = {id(c): majority_canonical(c, events) for c in clusters}

    # Group clusters by (length, first token) the same way the online
    # parser's index does, so pairwise comparisons mirror what a batch
    # reviewer scanning "clusters that share a bucket" would actually see.
    buckets: dict[tuple[int, str], list] = defaultdict(list)
    for c in clusters:
        key = (len(c.template_tokens), c.template_tokens[0] if c.template_tokens else "")
        buckets[key].append(c)

    pairs = []
    pair_id = 0
    for key, bucket_clusters in buckets.items():
        if len(bucket_clusters) < 2:
            continue
        for c_a, c_b in combinations(bucket_clusters, 2):
            sim = _seq_similarity(c_a.template_tokens, c_b.template_tokens)
            canon_a, canon_b = canon_of[id(c_a)], canon_of[id(c_b)]
            pairs.append(
                {
                    "pair_id": f"P{pair_id:03d}",
                    "template_a": c_a.template(),
                    "template_b": c_b.template(),
                    "structural_similarity": round(sim, 4),
                    "ground_truth_same_event": same_event_group(canon_a, canon_b),
                    "_canonical_a": canon_a,
                    "_canonical_b": canon_b,
                }
            )
            pair_id += 1

    # Deliberately relaxed pair: the cosmetic-suffix case is invisible to the
    # online bucket-based comparison (different token count), but a batch
    # reviewer comparing "same first token, similar length" clusters would
    # naturally still consider it. We add it explicitly rather than claiming
    # the online algorithm found it on its own.
    plain = next(c for c in clusters if canon_of[id(c)] == "2a_task_failed_plain")
    retryable = next(c for c in clusters if canon_of[id(c)] == "2b_task_failed_retryable")
    short, long_ = (
        (plain, retryable)
        if len(plain.template_tokens) <= len(retryable.template_tokens)
        else (retryable, plain)
    )
    padded_short = short.template_tokens + ["<PAD>"] * (
        len(long_.template_tokens) - len(short.template_tokens)
    )
    sim = _seq_similarity(padded_short, long_.template_tokens)
    pairs.append(
        {
            "pair_id": f"P{pair_id:03d}",
            "template_a": short.template(),
            "template_b": long_.template(),
            "structural_similarity": round(sim, 4),
            "ground_truth_same_event": same_event_group(
                canon_of[id(short)], canon_of[id(long_)]
            ),
            "_canonical_a": canon_of[id(short)],
            "_canonical_b": canon_of[id(long_)],
            "_note": "cross-length-bucket pair, added manually: invisible to the online index",
        }
    )

    # The online algorithm produces many small clusters per canonical event
    # type purely because of how many distinct categorical variable values
    # happened to appear (four DAG names, five table names, ...). Reviewing
    # every resulting pair would mean asking the same qualitative question
    # ("does a table name difference matter here") dozens of near-identical
    # times. We keep at most 3 example pairs per (canonical_a, canonical_b)
    # type combination, chosen deterministically, which preserves every
    # distinct kind of decision in the dataset without the redundancy.
    by_combo: dict[frozenset, list] = defaultdict(list)
    for p in pairs:
        combo = frozenset((p["_canonical_a"], p["_canonical_b"]))
        by_combo[combo].append(p)

    MAX_PER_COMBO = 3
    sampled = []
    for combo, combo_pairs in by_combo.items():
        combo_pairs.sort(key=lambda p: p["pair_id"])
        sampled.extend(combo_pairs[:MAX_PER_COMBO])

    sampled.sort(key=lambda p: p["pair_id"])
    for i, p in enumerate(sampled):
        p["pair_id"] = f"P{i:03d}"
    deduped = sampled

    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "candidate_pairs.json", "w") as f:
        json.dump(deduped, f, indent=2)
    pairs = deduped

    n_true = sum(1 for p in pairs if p["ground_truth_same_event"])
    print(f"wrote {len(pairs)} candidate pairs ({n_true} should-merge, {len(pairs) - n_true} should-not-merge)")
    print(f"n_clusters at st={OPERATING_THRESHOLD}: {len(clusters)} (canonical templates: {len(EVENT_GROUP)})")


if __name__ == "__main__":
    main()
