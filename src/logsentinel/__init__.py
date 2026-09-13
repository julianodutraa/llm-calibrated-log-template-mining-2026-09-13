"""logsentinel: log template mining with LLM-calibrated merge decisions and drift detection.

This package implements a small, self-contained pipeline for turning raw,
high-cardinality log lines from data pipeline and infrastructure systems
(DAG orchestrators, Kubernetes, relational databases) into structured event
templates, refining borderline clustering decisions with a calibrated LLM
judge, and monitoring the resulting template distribution for drift.
"""

__version__ = "0.1.0"
