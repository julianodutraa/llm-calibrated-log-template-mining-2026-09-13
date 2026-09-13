"""Drain: a fixed-depth tree online log parser.

Reference method: Pinjia He, Jieming Zhu, Zibin Zheng, Michael R. Lyu,
"Drain: An Online Log Parsing Approach with Fixed Depth Tree", IEEE ICWS 2017.

Drain converts a stream of raw, free-text log lines into a small set of
"templates": a template is a token sequence in which the parts that vary
across occurrences (identifiers, durations, counters, error codes) have been
replaced by a wildcard token. For example, the two raw lines

    task_id=7421 finished in 512ms with status SUCCESS
    task_id=8830 finished in 190ms with status SUCCESS

are both instances of the template

    task_id=<*> finished in <*> with status SUCCESS

Templates are the unit that downstream observability tooling (log volume
reduction, per-template rate metrics, anomaly and drift detection) actually
operates on, instead of raw high-cardinality strings.

This module is a from-scratch implementation of the core Drain data
structure and matching rule. It intentionally omits some production
engineering concerns of the original paper (e.g. persistence, multi-worker
sharding) that are out of scope for a demonstration project, but the parsing
logic (length-indexed first layer, token-indexed second layer, similarity
based leaf search, template update rule) follows the paper directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .preprocess import mask_message

WILDCARD = "<*>"


def tokenize(message: str) -> list[str]:
    """Mask obvious variable tokens, then split into whitespace-delimited tokens.

    Masking runs first (see ``preprocess.mask_message``) because Drain's
    first-token indexing layer assumes the first token of a message is
    usually invariant across occurrences of the same event type; without
    masking, an identifier embedded in the first token would defeat that
    index entirely.
    """
    return mask_message(message).split()


@dataclass
class LogCluster:
    """A single discovered template and the log line ids that matched it."""

    template_tokens: list[str]
    log_ids: list[int] = field(default_factory=list)
    size: int = 0

    def template(self) -> str:
        return " ".join(self.template_tokens)


def _seq_similarity(cluster_tokens: list[str], msg_tokens: list[str]) -> float:
    """Token-position similarity between a stored template and a new message.

    Both sequences are assumed to already have equal length (Drain buckets
    by length before this is ever called). A wildcard position in the
    template always counts as a match, mirroring the original paper: a
    wildcard has already absorbed variability at that position, so it must
    not penalize the similarity of a new, possibly different, value.

    simSeq = (number of matching positions) / (sequence length)
    """
    if not cluster_tokens:
        return 0.0
    matches = 0
    for c_tok, m_tok in zip(cluster_tokens, msg_tokens):
        if c_tok == WILDCARD or c_tok == m_tok:
            matches += 1
    return matches / len(cluster_tokens)


class DrainParser:
    """Fixed-depth-tree online log parser.

    Parameters
    ----------
    similarity_threshold:
        Minimum simSeq required to accept a new message into an existing
        cluster (the paper's ``st``). Lower values merge more aggressively
        (fewer, coarser templates); higher values split more aggressively
        (more, finer templates). This is the single most consequential
        hyperparameter of Drain and the reason a downstream LLM-based merge
        review is useful: no single fixed threshold is simultaneously safe
        for short, low-variability messages and long, high-variability ones.
    max_children_per_node:
        Bound on the branching factor of the first-token index layer
        (the paper's parameter to keep the tree shallow and lookups O(1)-ish).
    """

    def __init__(self, similarity_threshold: float = 0.5, max_children_per_node: int = 100):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self.st = similarity_threshold
        self.max_children_per_node = max_children_per_node
        # length -> first_token -> list[LogCluster]
        self._tree: dict[int, dict[str, list[LogCluster]]] = {}
        self.clusters: list[LogCluster] = []
        self._next_log_id = 0

    def add_log_message(self, message: str) -> LogCluster:
        tokens = tokenize(message)
        log_id = self._next_log_id
        self._next_log_id += 1

        length = len(tokens)
        first_tok = tokens[0] if tokens else ""

        by_first_tok = self._tree.setdefault(length, {})
        bucket = by_first_tok.setdefault(first_tok, [])

        best_cluster = None
        best_sim = 0.0
        for cluster in bucket:
            sim = _seq_similarity(cluster.template_tokens, tokens)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster is not None and best_sim >= self.st:
            self._update_template(best_cluster, tokens)
            best_cluster.log_ids.append(log_id)
            best_cluster.size += 1
            return best_cluster

        new_cluster = LogCluster(template_tokens=list(tokens), log_ids=[log_id], size=1)
        if len(bucket) < self.max_children_per_node:
            bucket.append(new_cluster)
        else:
            # Bounded branching: fall back to the least-populated existing
            # cluster rather than growing the tree unboundedly. This trades
            # a small amount of parsing precision for bounded memory, which
            # is the same trade-off the original paper makes.
            bucket.sort(key=lambda c: c.size)
            fallback = bucket[0]
            self._update_template(fallback, tokens)
            fallback.log_ids.append(log_id)
            fallback.size += 1
            self.clusters_dirty = True
            return fallback

        self.clusters.append(new_cluster)
        return new_cluster

    @staticmethod
    def _update_template(cluster: LogCluster, tokens: list[str]) -> None:
        """Widen the template in place: any differing position becomes a wildcard."""
        for i, (c_tok, m_tok) in enumerate(zip(cluster.template_tokens, tokens)):
            if c_tok != WILDCARD and c_tok != m_tok:
                cluster.template_tokens[i] = WILDCARD

    def parse_all(self, messages: list[str]) -> list[LogCluster]:
        for msg in messages:
            self.add_log_message(msg)
        return self.clusters

    def template_for(self, message: str) -> str:
        """Look up (without mutating state) the template a message would match, if any."""
        tokens = tokenize(message)
        length = len(tokens)
        first_tok = tokens[0] if tokens else ""
        bucket = self._tree.get(length, {}).get(first_tok, [])
        best_cluster = None
        best_sim = 0.0
        for cluster in bucket:
            sim = _seq_similarity(cluster.template_tokens, tokens)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster
        if best_cluster is not None and best_sim >= self.st:
            return best_cluster.template()
        return WILDCARD.join(tokens)  # unmatched: no meaningful template
