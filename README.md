# VulChain

Embedding-space prompt-mutation engine for auditing whether a **known upstream
vulnerability** remains reachable on a **derived** (fine-tuned / LoRA / quantized)
open-weight model. The engine performs bandit-guided mutation in the
derivative's own embedding space, scores responses with a dense composite score
`S_v` for search guidance, and confirms detections only with a strict,
externally verifiable detector `D_v`.

This is a **defensive, post-disclosure auditing** tool. It requires a known
vulnerable model–prompt pair as input and does not discover new vulnerability
classes. Generated artifacts (candidate package names, URLs) are analyzed
programmatically and are **never installed, registered, visited, or executed**.

---

## Layout

```
vulchain.py                    # engine + built-in classes + extension templates
requirements.txt
README.md
seed_prompts_pkg_1000.jsonl    # package-hallucination seed bank
seed_prompts_url_1000.jsonl    # insecure-URL seed bank
derived_models.template.csv    # model-list schema (fill with HF identifiers)
```

All engine code is in the single file `vulchain.py`; the rest are data/artifact
inputs. These seed banks are the *inputs* to the search (not the `D_v`-confirmed
trigger prompts `p'`), so they are safe to distribute under the paper's
open-science policy while the confirmed triggers remain withheld.

## Install

```bash
pip install -r requirements.txt
```

Core deps are `torch`, `transformers`, `accelerate`. Everything else degrades
gracefully (see `requirements.txt`).

## Quick start

List the registered vulnerability classes:

```bash
python vulchain.py --list
```

Package hallucination (offline allowlist; add `--live_verify` to check
allowlist misses against live PyPI/npm):

```bash
python vulchain.py \
    --vuln       package_hallucination \
    --model      /path/to/derivative_or_adapter \
    --base_model /path/to/base_model \
    --output_dir ./results/pkg_run_001 \
    --live_verify
```

Insecure URL generation (offline threat-DB snapshots + DNS):

```bash
python vulchain.py \
    --vuln           insecure_url \
    --model          /path/to/derivative_or_adapter \
    --base_model     /path/to/base_model \
    --output_dir     ./results/url_run_001 \
    --phishtank      ./cache/phishtank_verified.csv \
    --urlhaus        ./cache/urlhaus_online.csv \
    --benign_domains ./cache/known_benign_domains.txt \
    --dns_cache      ./cache/dns_cache.json
```

If `--model` points to a directory containing `adapter_config.json`, it is
loaded as a PEFT adapter over `--base_model`; otherwise it is loaded as a full
model. AWQ / GPTQ / compressed-tensors / 8-bit quantized checkpoints are
detected from `config.json` and loaded through the matching backend.

## Architecture

The engine is vulnerability-agnostic. Only three components are class-specific,
and all three sit behind the `VulnerabilityClass` interface:

| Component            | Interface method(s)                          |
|----------------------|----------------------------------------------|
| Search signals       | `reference_texts`, `indicative_tokens`       |
| Composite text score | `target_types` (default) or `score_text`     |
| Strict detector `D_v`| `detect`                                     |

Shared engine (never edited per class): embedding-space mutation, the five
perturbation operators, the non-stationary multi-armed bandit, quant/PEFT-aware
model loading, generation, logging, and the verify-then-iterate search loop.

## Adding a new vulnerability class

Subclass `VulnerabilityClass`, decorate with `@register_vuln`, and it becomes
selectable via `--vuln <name>`. No engine changes.

```python
@register_vuln
class MyVuln(VulnerabilityClass):
    name = "my_vuln"

    def reference_texts(self):   # exemplars for the S_emb tie-breaker
        return {"bad": ["...example vulnerable output..."]}

    def indicative_tokens(self): # tokens for the S_logit signal
        return ["...", "..."]

    def target_types(self):      # drives the default S_text (paper Eq. 3–4)
        return [TargetType(
            name="my_type", severity=2.5,
            cue_keywords=["..."],
            partial_matcher=lambda t: 0.0,          # -> [0, 1]
            strict_extractor=lambda t: [],          # -> candidate names
        )]

    def detect(self, text):      # strict, externally verifiable D_v
        return DetectionResult(is_vulnerable=False, tiers=[], artifacts=[])
```

Pass class-specific constructor arguments either through dedicated CLI flags or
via the generic `--vuln_config '{"key": "value"}'` escape hatch.

Two **template stubs** ship as worked examples and are inert until you wire in a
backend:

* `insecure_code` — replace `detect` with a static-analysis backend
  (e.g. Semgrep / CodeQL) over extracted code blocks.
* `pii_leakage` — provide a planted-canary file (`--canary_path`) so `detect`
  flags only verifiable memorized secrets.

## Ecosystem audit mode (Algorithm 2)

Single-prompt mode (above) runs one seed against one model. Audit mode runs a
**prompt bank** against a **model list**: for each derived model it iterates the
ordered bank with a fresh bandit per seed, consuming a per-model query budget,
and records model-level and prompt-level ASR.

```bash
python vulchain.py \
    --vuln             package_hallucination \
    --prompt_bank      seed_prompts_pkg_1000.jsonl \
    --model_list       derived_models.csv \
    --base_model       /path/to/default_base \      # fallback if a row omits base_model
    --per_model_budget 1200 \
    --output_dir       ./audit/pkg \
    --live_verify
```

Providing `--prompt_bank` and `--model_list` together switches to audit mode.

### Prompt-bank format (`.jsonl`)

One JSON object per line. Only two fields are required; everything else is kept
as opaque provenance metadata and logged:

| Field     | Required | Purpose                                             |
|-----------|----------|-----------------------------------------------------|
| `seed_id` | yes      | Stable unique id (used for deterministic ordering)  |
| `prompt`  | yes      | The seed prompt text                                |
| `s_up`    | no       | Upstream vulnerability score (Stage-3 sort key)     |

### Upstream score `s_up` and reproducibility

The Stage-3 ordering sorts the bank by upstream score. If the bank does not
carry `s_up`, you have three options:

* **`--upstream_model <path>`** — score every seed on the upstream model, sort
  by the measured score, and write the resulting `sup_map.json`. This
  reproduces Stage 3 rather than relying on a stored score.
* **`--sup_map scores.json`** — supply a precomputed `{seed_id: s_up}` map.
* **neither** — deterministic `seed_id` order. Model-level and prompt-level ASR
  reproduce; the QTD (queries-to-detection) sort does not.

The engine reports which ordering it used in `audit_summary.json` (`order`
field), so the release is self-documenting on this point.

### Model-list format (`.csv` or `.jsonl`)

Columns are case-insensitive; only `model_id` is strictly required. If
`base_model` is omitted on a row, the CLI `--base_model` value is used.

```csv
base_family,model_id,adaptation_type,base_model
Llama-3.1,org/llama31-lora-audit,LoRA,meta-llama/Llama-3.1-8B
Gemma,org/gemma-ft-audit,fine-tune,google/gemma-7b
```

Unknown columns are preserved as metadata. A model counts as vulnerable if any
seed yields a `D_v`-confirmed trigger within its budget. Use
`--no_stop_on_first` to keep going after the first trigger (full prompt-level
ASR per model); by default, the model stops at its first confirmed trigger.

### Audit outputs

```
audit_ledger.csv       # one row per (model, seed) attempt: success/steps/queries
audit_summary.csv      # one row per model: vulnerable / trigger_seed / queries
audit_summary.json     # model-level ASR overall and by base family
sup_map.json           # written when --upstream_model was used
<model>/<seed_id>/...   # full per-seed run logs, only with --detailed
```

## Outputs

Each run writes to `--output_dir`:

```
config.json                 # full run configuration
summary.csv                 # per-step loss / score / detector verdict
prompt_response_map.jsonl   # every prompt–response pair with all sub-scores
all_candidates.jsonl        # per-candidate averaged scores
best_per_step.jsonl         # best candidate per step
policy_evolution.csv        # bandit statistics over time
final_report.json           # outcome, artifacts, queries consumed
```

## Reproducibility

The search is seeded (`--seed`, default 42) and decoding is fixed
(temperature 0.9, nucleus 0.95, 96 new tokens). External verification sources
(registries, threat feeds, DNS, Tranco, Public Suffix List) change over time;
record the verification snapshot date and list versions alongside results.

## Responsible use

The framework targets already-documented vulnerability classes and is intended
for coordinated post-disclosure auditing. Do not release verified trigger
prompts or raw flagged artifacts; handle those through a coordinated-disclosure
process.
