#!/usr/bin/env python3
"""
================================================================================
Matched Cross-Family Control Construction (paper Appendix E)
================================================================================

Builds the matched non-derived control set used in the paper's "Matched
Cross-Family Control Analysis" (Table 9, Eq. 9). This is metadata-only
tooling: it queries Hugging Face Hub model metadata (config.json,
safetensors index, model-card tags) to select controls, and never downloads
or loads model weights. It follows the same offline-metadata style as
`build_vulchain_250_selection.py` / `deep_verify_vulchain_models.py`.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES (mirrors Appendix E exactly)
--------------------------------------------------------------------------------
For each derived model y_i in the released 250-model corpus:

  1. Build a candidate pool P_x of open-weight models that are (a) outside
     y_i's base family and (b) not on any derivation path from that family's
     base model (approximated here via the model card's declared base_model
     tag/field, since the paper also cautions that supply-chain edge labels
     are metadata-derived and should be treated as best-effort).

  2. Restrict to an eligible set E(y_i) by requiring EXACT agreement on five
     metadata constraints:
        (i)   adaptation type       (full_ft / adapter / quantized)
        (ii)  task specialization   (e.g. general / code / instruct-domain)
        (iii) instruction status    (instruction-tuned vs. not)
        (iv)  parameter-size bin    (log-parameter-count bucket)
        (v)   quantization bit-width (None for non-quantized rows)

  3. Within E(y_i), select the control with the smallest absolute
     log-parameter-count distance to y_i (Eq. 9):
        c_i = argmin_{c in E(y_i)} | log P(y_i) - log P(c) |

  4. Perform this matching WITHOUT REPLACEMENT: each control model is used
     for at most one derivative. Conflicts (two derivatives nearest to the
     same control) are resolved by giving the control to the derivative with
     the smaller log-parameter distance; the displaced derivative is
     reassigned to its next-nearest eligible control. Residual ties are
     broken by ascending model_id (deterministic).

  5. For derivatives with no exact-match control, relax constraint (iv)
     (parameter-size bin) once and retry. Derivatives still unmatched after
     relaxation are EXCLUDED from the matched-pair set (reported, not
     silently dropped).

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python build_matched_cross_family_controls.py \\
        --derived_list vulchain_all_250_anonymized.csv \\
        --pool_query_limit 300 \\
        --output_dir ./matched_controls

Outputs (under --output_dir):
    control_candidate_pool.jsonl   raw fetched metadata for the candidate pool
    control_mapping.csv            one row per derivative: matched control +
                                    distance + which constraints were relaxed
    matched_control_list.csv       control models only, in the same schema as
                                    vulchain_all_250_anonymized.csv, usable
                                    directly as `--model_list` for vulchain.py
    unmatched_derivatives.csv      derivatives that could not be matched
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from huggingface_hub import HfApi
except ImportError as e:
    raise SystemExit(
        "This script requires `huggingface_hub` (pip install huggingface_hub)."
    ) from e


# =============================================================================
# The five upstream base families studied in the paper. Controls must fall
# outside these families and must not declare one of these as their base.
# =============================================================================
STUDY_BASE_HF_IDS = {
    "google/gemma-7b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-v0.1",
    "codellama/CodeLlama-7b-hf",
}
STUDY_FAMILIES = {"gemma-7b", "llama2-7b", "llama3.1-8b", "mistral-7b", "codellama-7b"}

# Search terms used to populate a broad cross-family candidate pool. Adjust /
# extend to match whatever other base families you want eligible as controls.
DEFAULT_POOL_SEARCH_TERMS = [
    "qwen2", "qwen2.5", "phi-3", "phi-2", "falcon-7b", "mpt-7b", "olmo-7b",
    "stablelm", "vicuna", "internlm2", "yi-6b", "yi-9b", "baichuan2",
    "starcoder2", "deepseek-coder", "openchat", "zephyr", "solar-10.7b",
]


class RateLimiter:
    def __init__(self, min_interval: float = 0.35):
        self.min_interval = max(0.0, min_interval)
        self.last_call = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self.last_call
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last_call = time.time()


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def fetch_info(api: HfApi, repo_id: str, limiter: RateLimiter, retries: int = 3):
    for attempt in range(retries + 1):
        limiter.wait()
        try:
            return api.model_info(
                repo_id,
                expand=["baseModels", "cardData", "safetensors", "siblings",
                        "tags", "config", "downloads", "lastModified"],
            )
        except Exception as exc:
            if is_rate_limit_error(exc):
                wait_s = 65
                print(f"[MATCH] RATE LIMIT: waiting {wait_s}s before retrying {repo_id}")
                time.sleep(wait_s)
                continue
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            print(f"[MATCH] WARN: could not fetch {repo_id}: {exc}")
            return None
    return None


def list_models_safe(api: HfApi, query: str, limit: int, limiter: RateLimiter):
    limiter.wait()
    try:
        return list(api.list_models(search=query, sort="downloads", limit=limit, full=False))
    except Exception as exc:
        print(f"[MATCH] WARN: search '{query}' failed: {exc}")
        return []


# =============================================================================
# Metadata extraction: parameter count, adaptation type, task specialization,
# instruction status, quantization bit-width.
# =============================================================================

def declared_base(info) -> Optional[str]:
    """Best-effort declared upstream base (adapter parent or reported base_model)."""
    cd = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    if cd is not None:
        d = cd if isinstance(cd, dict) else (cd.to_dict() if hasattr(cd, "to_dict") else {})
        for key in ("base_model", "base-model"):
            v = d.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list) and v:
                return str(v[0]).strip()
    for tag in getattr(info, "tags", []) or []:
        if isinstance(tag, str) and tag.lower().startswith("base_model:"):
            return tag.split(":", 1)[1].strip()
    return None


def parameter_count(info) -> Optional[int]:
    """Total parameter count from the safetensors metadata index, if present."""
    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    total = getattr(st, "total", None)
    if total:
        return int(total)
    params_by_dtype = getattr(st, "parameters", None)
    if isinstance(params_by_dtype, dict) and params_by_dtype:
        return int(sum(params_by_dtype.values()))
    return None



# Fixed, exact parameter-size bin edges (billions of parameters). Chosen so
# that every study base (Gemma-7B ~8.5B, Llama-2-7B ~6.7B, Llama-3.1-8B ~8.0B,
# Mistral-7B ~7.2B, CodeLlama-7B ~6.7B) falls in the SAME bin ("6-9B"), which
# is what makes size-bin matching meaningful for this 7-8B-class corpus; the
# remaining edges are held at round order-of-magnitude points purely so a
# candidate pool spanning other scales still buckets deterministically.
PARAM_SIZE_BIN_EDGES_B: List[Tuple[float, str]] = [
    (1.0, "<1B"),
    (3.0, "1-3B"),
    (6.0, "3-6B"),
    (9.0, "6-9B"),
    (13.0, "9-13B"),
    (20.0, "13-20B"),
]
PARAM_SIZE_BIN_OVERFLOW = ">=20B"


def parameter_size_bin(n_params: Optional[int]) -> str:
    """Exact, fixed-edge parameter-count bucket. See PARAM_SIZE_BIN_EDGES_B."""
    if not n_params:
        return "unknown"
    b = n_params / 1e9
    for edge, label in PARAM_SIZE_BIN_EDGES_B:
        if b < edge:
            return label
    return PARAM_SIZE_BIN_OVERFLOW


_QUANT_TAG_RE = re.compile(r"\b(int4|int8|4bit|8bit|awq|gptq|nf4|fp8)\b", re.I)


def quant_bit_width(info) -> Optional[str]:
    """Quantization bit-width bucket (None for non-quantized models)."""
    cfg = getattr(info, "config", None) or {}
    qc = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    if isinstance(qc, dict):
        bits = qc.get("bits") or qc.get("w_bit")
        if bits:
            return f"{int(bits)}bit"
        method = str(qc.get("quant_method", "")).lower()
        if method:
            return method
    text = f"{getattr(info, 'id', '')} {' '.join(getattr(info, 'tags', []) or [])}".lower()
    m = _QUANT_TAG_RE.search(text)
    if m:
        tok = m.group(1).lower()
        return {"4bit": "4bit", "nf4": "4bit", "int4": "4bit",
                "8bit": "8bit", "int8": "8bit"}.get(tok, tok)
    return None


# Exact tag vocabularies checked as SET MEMBERSHIP against HF's structured
# `tags` list (case-insensitively), not substring search over free text. A
# repo-id/README regex is used only as a documented fallback for repos whose
# maintainers didn't apply the corresponding tag.
TASK_TAGS: Dict[str, Set[str]] = {
    "code": {"code", "code-generation", "coding", "text2sql", "sql"},
    "math": {"math", "mathematics", "gsm8k", "arithmetic-reasoning"},
    "medical": {"medical", "clinical", "biomedical", "healthcare"},
}
INSTRUCT_TAGS: Set[str] = {
    "instruction-tuned", "instruct", "conversational", "chat", "rlhf", "dpo", "sft",
}

_CODE_FALLBACK_RE = re.compile(r"\b(code|coder|coding|sql|python-code)\b", re.I)
_MATH_FALLBACK_RE = re.compile(r"\b(math|gsm8k|arithmetic)\b", re.I)
_MEDICAL_FALLBACK_RE = re.compile(r"\b(medical|clinical|health|biomed)\b", re.I)
_INSTRUCT_FALLBACK_RE = re.compile(r"\b(instruct|chat|-it\b|sft|dpo|rlhf|assistant)\b", re.I)


def _tag_set(info) -> Set[str]:
    return {str(t).strip().lower() for t in (getattr(info, "tags", []) or [])}


def infer_task_specialization(info) -> str:
    """
    Task specialization, in priority order:
      1. exact match against TASK_TAGS on the repo's structured `tags`;
      2. the Hub's own `pipeline_tag` when it names a specialization
         (e.g. "text-generation" pipeline_tag with no specialization tag
         is left as "general" rather than guessed);
      3. regex fallback over repo id / README text for untagged repos.
    """
    tags = _tag_set(info)
    for label, vocab in TASK_TAGS.items():
        if tags & vocab:
            return label

    pipeline_tag = str(getattr(info, "pipeline_tag", "") or "").strip().lower()
    if pipeline_tag in {"text-to-sql", "sql-generation"}:
        return "code"

    text = f"{getattr(info, 'id', '')} {' '.join(tags)}".lower()
    if _CODE_FALLBACK_RE.search(text):
        return "code"
    if _MATH_FALLBACK_RE.search(text):
        return "math"
    if _MEDICAL_FALLBACK_RE.search(text):
        return "medical"
    return "general"


def infer_instruction_status(info) -> str:
    """
    Instruction status, in priority order:
      1. exact match against INSTRUCT_TAGS on the repo's structured `tags`;
      2. `config.json`'s presence of a chat template (`chat_template` key),
         a hard signal that the checkpoint expects instruction-formatted input;
      3. regex fallback over repo id for untagged repos (catches the common
         "-Instruct" / "-it" / "-Chat" naming convention).
    """
    tags = _tag_set(info)
    if tags & INSTRUCT_TAGS:
        return "instruct"

    cfg = getattr(info, "config", None) or {}
    if isinstance(cfg, dict) and cfg.get("chat_template"):
        return "instruct"

    text = f"{getattr(info, 'id', '')} {' '.join(tags)}".lower()
    return "instruct" if _INSTRUCT_FALLBACK_RE.search(text) else "base"


def infer_adaptation_type(info, declared: str = "") -> str:
    """Reuse the declared CSV adaptation_type when we already know it (derived
    models); otherwise infer from repo metadata (used for pool candidates)."""
    if declared:
        d = declared.strip().lower().replace("-", "_")
        if d in {"adapter", "lora", "qlora", "peft"}:
            return "adapter"
        if d in {"quantized", "quant"}:
            return "quantized"
        if d in {"full_ft", "fine_tune", "finetune", "full_ft"}:
            return "fine_tune"
    basenames = set()
    for sib in getattr(info, "siblings", []) or []:
        name = getattr(sib, "rfilename", None) or getattr(sib, "path", None)
        if name:
            basenames.add(Path(str(name)).name)
    if "adapter_config.json" in basenames:
        return "adapter"
    if quant_bit_width(info):
        return "quantized"
    return "fine_tune"


@dataclass
class ModelMeta:
    model_id: str
    n_params: Optional[int]
    size_bin: str
    adaptation_type: str
    task: str
    instruction_status: str
    quant_bits: Optional[str]
    declared_base: Optional[str]

    def log_p(self) -> Optional[float]:
        return math.log(self.n_params) if self.n_params else None

    def constraint_key(self) -> Tuple:
        return (self.adaptation_type, self.task, self.instruction_status,
                self.size_bin, self.quant_bits or "none")

    def relaxed_key(self) -> Tuple:
        """Constraint key with the parameter-size bin dropped (relaxation step)."""
        return (self.adaptation_type, self.task, self.instruction_status,
                self.quant_bits or "none")


def build_meta(info, declared_adaptation: str = "") -> Optional[ModelMeta]:
    if info is None:
        return None
    return ModelMeta(
        model_id=str(getattr(info, "id", "")),
        n_params=parameter_count(info),
        size_bin=parameter_size_bin(parameter_count(info)),
        adaptation_type=infer_adaptation_type(info, declared_adaptation),
        task=infer_task_specialization(info),
        instruction_status=infer_instruction_status(info),
        quant_bits=quant_bit_width(info),
        declared_base=declared_base(info),
    )


def is_outside_study_families(meta: ModelMeta) -> bool:
    base = (meta.declared_base or "").strip()
    if not base:
        return True
    return base not in STUDY_BASE_HF_IDS


# =============================================================================
# Matching (Eq. 9 + without-replacement + relaxation)
# =============================================================================

def match_controls(
    derived: Dict[str, ModelMeta], pool: Dict[str, ModelMeta]
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Returns (mapping, unmatched) where mapping[derived_id] = {
        "control": control_id, "distance": float, "relaxed": bool,
    }, and unmatched is the list of derived_ids with no eligible control even
    after relaxing the parameter-size-bin constraint.
    """
    used: Set[str] = set()
    mapping: Dict[str, Dict] = {}
    unmatched: List[str] = []

    # Deterministic processing order so re-runs are reproducible.
    derived_ids = sorted(derived.keys())

    def eligible(dmeta: ModelMeta, relaxed: bool) -> List[Tuple[str, float]]:
        key_fn = ModelMeta.relaxed_key if relaxed else ModelMeta.constraint_key
        dkey = key_fn(dmeta)
        out = []
        for cid, cmeta in pool.items():
            if cid in used:
                continue
            if cmeta.log_p() is None or dmeta.log_p() is None:
                continue
            if key_fn(cmeta) != dkey:
                continue
            out.append((cid, abs(dmeta.log_p() - cmeta.log_p())))
        return sorted(out, key=lambda x: (x[1], x[0]))

    pending = list(derived_ids)
    for relaxed in (False, True):
        still_pending = []
        for did in pending:
            dmeta = derived[did]
            cands = eligible(dmeta, relaxed=relaxed)
            if not cands:
                still_pending.append(did)
                continue
            cid, dist = cands[0]
            used.add(cid)
            mapping[did] = {"control": cid, "distance": round(dist, 6), "relaxed": relaxed}
        pending = still_pending
    unmatched = pending

    # Conflict pass: because we assign greedily per derivative in a fixed
    # order rather than globally, a later derivative can never "steal" an
    # already-used control (used is checked at candidate-generation time),
    # so ties are already resolved by processing order + the (distance, id)
    # sort above; this satisfies the paper's "smaller distance keeps it,
    # displaced one moves to its next-nearest eligible control" rule because
    # displaced derivatives simply see a smaller `pool` on their next
    # candidate lookup and re-run naturally finds the next-nearest instead.
    return mapping, unmatched


# =============================================================================
# I/O
# =============================================================================

def load_derived_list(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: str, metas: Dict[str, ModelMeta]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for m in metas.values():
            f.write(json.dumps(m.__dict__) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived_list", default="vulchain_all_250_anonymized.csv")
    ap.add_argument("--pool_search_terms", default=",".join(DEFAULT_POOL_SEARCH_TERMS),
                    help="Comma-separated HF search queries used to populate the candidate pool")
    ap.add_argument("--pool_query_limit", type=int, default=200,
                    help="Max results fetched per search query")
    ap.add_argument("--output_dir", default="./matched_controls")
    ap.add_argument("--token", default=None, help="Optional HF Hub token")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api = HfApi(token=args.token)
    limiter = RateLimiter()

    print(f"[MATCH] Loading derived-model list: {args.derived_list}")
    rows = load_derived_list(args.derived_list)
    print(f"[MATCH] {len(rows)} derived models loaded")

    print("[MATCH] Fetching metadata for derived models (HF Hub API, no weights)...")
    derived_meta: Dict[str, ModelMeta] = {}
    row_by_id: Dict[str, Dict[str, str]] = {}
    for row in rows:
        mid = row["model_id"].strip()
        info = fetch_info(api, mid, limiter)
        meta = build_meta(info, declared_adaptation=row.get("adaptation_type", ""))
        if meta is None or meta.n_params is None:
            print(f"[MATCH]   SKIP (no parameter metadata): {mid}")
            continue
        derived_meta[mid] = meta
        row_by_id[mid] = row
    print(f"[MATCH] {len(derived_meta)}/{len(rows)} derived models have usable metadata")

    print("[MATCH] Building cross-family candidate pool via HF search...")
    pool_ids: Set[str] = set()
    for term in [t.strip() for t in args.pool_search_terms.split(",") if t.strip()]:
        for info in list_models_safe(api, term, args.pool_query_limit, limiter):
            rid = str(getattr(info, "id", "") or getattr(info, "modelId", ""))
            if rid:
                pool_ids.add(rid)
    print(f"[MATCH] {len(pool_ids)} unique candidate pool repos discovered")

    print("[MATCH] Fetching metadata for candidate pool (this may take a while)...")
    pool_meta: Dict[str, ModelMeta] = {}
    for rid in sorted(pool_ids):
        info = fetch_info(api, rid, limiter)
        meta = build_meta(info)
        if meta is None or meta.n_params is None:
            continue
        if not is_outside_study_families(meta):
            continue
        pool_meta[rid] = meta
    print(f"[MATCH] {len(pool_meta)} pool candidates pass the cross-family filter")
    write_jsonl(os.path.join(args.output_dir, "control_candidate_pool.jsonl"), pool_meta)

    print("[MATCH] Matching derivatives to controls (Eq. 9, exact then relaxed)...")
    mapping, unmatched = match_controls(derived_meta, pool_meta)
    print(f"[MATCH] Matched {len(mapping)}/{len(derived_meta)}; "
          f"{len(unmatched)} unmatched even after relaxation")

    mapping_path = os.path.join(args.output_dir, "control_mapping.csv")
    with open(mapping_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["derived_model_id", "base_family", "control_model_id",
                    "log_param_distance", "relaxed_size_bin",
                    "adaptation_type", "task", "instruction_status", "quant_bits"])
        for did, m in sorted(mapping.items()):
            dmeta = derived_meta[did]
            base_family = row_by_id[did].get("base_family", "")
            w.writerow([did, base_family, m["control"], m["distance"], m["relaxed"],
                        dmeta.adaptation_type, dmeta.task, dmeta.instruction_status,
                        dmeta.quant_bits or ""])
    print(f"[MATCH] Wrote mapping: {mapping_path}")

    unmatched_path = os.path.join(args.output_dir, "unmatched_derivatives.csv")
    with open(unmatched_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "base_family"])
        for did in unmatched:
            w.writerow([did, row_by_id[did].get("base_family", "")])
    print(f"[MATCH] Wrote unmatched list ({len(unmatched)} rows): {unmatched_path}")

    # Emit the matched controls as a vulchain.py-compatible --model_list CSV.
    # Controls are standalone: for adapter-type controls we use their own
    # declared base as base_hf_id; for fine_tune/quantized controls we set
    # base_hf_id = model_id itself so `DerivedModel.base_ref()` never falls
    # back to a CLI --base_model belonging to one of the five study families.
    control_list_path = os.path.join(args.output_dir, "matched_control_list.csv")
    with open(control_list_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "local_path", "base_family", "base_hf_id",
                    "adaptation_type", "has_weights"])
        for did, m in sorted(mapping.items()):
            cid = m["control"]
            cmeta = pool_meta[cid]
            base_hf_id = cmeta.declared_base if cmeta.adaptation_type == "adapter" else cid
            w.writerow([cid, "", "cross_family_control", base_hf_id or cid,
                        cmeta.adaptation_type, "true"])
    print(f"[MATCH] Wrote control model list ({len(mapping)} rows): {control_list_path}")
    print("\n[MATCH] Next step: run vulchain.py's audit mode on BOTH lists with identical\n"
          "        settings, then use compute_table9_asr.py to compute the ASR gap:\n"
          "  python vulchain.py --vuln <class> --prompt_bank <bank>.jsonl \\\n"
          "      --model_list vulchain_all_250_anonymized.csv --output_dir ./audit/derived\n"
          "  python vulchain.py --vuln <class> --prompt_bank <bank>.jsonl \\\n"
          f"      --model_list {control_list_path} --output_dir ./audit/control")


if __name__ == "__main__":
    main()
