# FBuilding a financial analyst with LLM, Fine tuning, LoRA and retrieval

Project 24 uses a tuned **Qwen3.5-2B** model to interpret financial evidence retrieved and calculated by Quantfinlab. It connects company disclosures, official macro releases, market conditions and selected news. Python handles financial calculations, source selection and availability checks; the model writes the interpretation.

The demonstration is [Notebook 24](../notebooks/24_financial_analysis_with_llm.ipynb). The Python entry point is `quantfinlab.analyst.FinancialAnalyst`.

## Published artifacts

| Artifact | Repository |
| --- | --- |
| Text LoRA adapter | [Financial-Analysis-LoRA](https://huggingface.co/ramtinasadi/Quantfinlab-Qwen3.5-2B-Financial-Analysis-LoRA) |
| Local Q4_K_M GGUF | [Financial-Analysis-GGUF](https://huggingface.co/ramtinasadi/Quantfinlab-Qwen3.5-2B-Financial-Analysis-GGUF) |

The GGUF is **1,312,164,224 bytes**. Its SHA256 is `d3d4a145333f96ee60f2f8b62d2776965ac49f89c827917440eab683411e777c`. `model_config.json` pins the exact revision and filename. The base is `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. The derivative retains its Apache-2.0 license; English is the supported language.

## Local use

Install the repository with the `analyst` extra and prepare the structured files described in [data/README.md](../data/README.md). The runtime requires an NVIDIA GPU, a working driver and sufficient system RAM for the context. Windows setup downloads a checksum-pinned CUDA llama.cpp distribution. On other platforms, install a CUDA-enabled `llama-server` and set `QUANTFINLAB_LLAMA_SERVER` to its executable.

```python
from quantfinlab.analyst import FinancialAnalyst

with FinancialAnalyst.from_repo() as analyst:
    report = analyst.ask("Which signals conflict with a simple risk-on interpretation?")
    print(report)
```

Jupyter displays cited observations in a paragraph, separates model interpretations, and retains uncertainty, sources and the full generated summary in the output. `report.analysis` exposes the structured result, `report.packet` exposes the evidence, and `report.save("analysis.html")` exports the rendered answer. Other methods include `company()`, `macro_release()`, `market()`, `events()` and `daily_brief()`.

The model is cached under ignored `models/local/`. Ordinary reruns use the same verified file without contacting Hugging Face. Downloads normally use `huggingface_hub.hf_hub_download`; `QUANTFINLAB_DOWNLOAD_TRANSPORT=range` enables resumable HTTP ranges for connections that stall on full-file transfers.

Inference uses **24,576 context tokens**, one sequence, GPU layer offload and KV memory in system RAM. The first run tests a large prompt twice and saves the highest successful offload setting. The profile is reused until the model, GPU, driver or runtime changes. Evidence is packed under exact tokenizer budgets. The default prompt cap is 7,000 tokens within the 24,576-token context; smaller relevant packets reduce latency and distraction. Stale snapshots remain visible in the status table and are omitted from the answer prompt.

Document indexes, event records, snapshots and answers live under ignored `workspace/financial_analyst/`. Answer caches include the question, cutoff, model checksum, prompt version, evidence, context and generation settings. A saved cutoff is reused until an update advances it. Pass an explicit timezone-aware `as_of` for historical analysis. Reports show stale or unavailable inputs.

## Three-layer design

Notebook 24 starts with visible return calculations, curve moves, accounting reconstruction, document selection, retrieval scores, token budgets and one complete generation/validation sequence. The second layer composes the corresponding public library functions. The final cells use the thin `FinancialAnalyst` facade over those same operations. Saved response identities prevent the layers from repeating identical LLM work.

| Module | Public operations |
| --- | --- |
| `market` | `market_moves`, `risk_measures`, `curve_moves`, `curve_shape`, `relative_moves` |
| `context` | Individual domain context builders and the snapshot cache registry |
| `documents`, `sec`, `macro`, `events` | Parsing, chunking, comparable filings, disclosure changes, release evidence and event records |
| `retrieval`, `evidence` | FTS search, reciprocal-rank fusion, context records, source attachment and exact token packing |
| `routing`, `prompts` | Query plans, analysis messages and one bounded repair request |
| `validation`, `caching`, `reports` | Claim checks, response identities, saved answers and readable rendering |
| `workflows`, `api` | Explicit packet/analysis orchestration and the final convenience facade |

## Evidence and calculations

SQLite FTS5/BM25 retrieves section-aware document chunks. The system uses one model, without embeddings or an agent framework. Context builders reuse selected components of Projects 3, 5, 9, 12, 15, 19, 21, 22 and 23 for risk, volatility, rates, financial conditions, factor exposures, company ratios, credit and macro conditions.

Retrieval limits the number of passages per document. Event recency uses the release or report date separately from the information-availability cutoff, so downloading an archive does not make an old release a new event. Runtime grammar bounds the response and restricts citation IDs to the supplied evidence.

SEC collection is restricted to requested companies and relevant filings and exhibits. Official text adapters cover the Federal Reserve, BLS, BEA, EIA and CFTC. GDELT supplies discovery metadata rather than definitive facts. Its Article List transport can be used when the DOC API is unavailable. Historical queries enforce `available_at <= as_of`.

```python
analyst = FinancialAnalyst.from_repo(identity="Research Contact contact@example.org")
receipts = analyst.update(tickers=["NVDA"], structured=["market"], limit=4)
company_report = analyst.company("NVDA")
```

Supply an actual identifying name and contact email. Each source folder documents its prerequisites and commands. Updates call existing source scripts, reuse raw caches, add missing chunks and report failures. Heavy forecasts, peer-universe estimates and default models are refreshed separately. Timestamped `ContextSnapshot` inputs can supplement the registry under `workspace/financial_analyst/inputs/`, named `<builder>-<ticker>.json` or `<builder>-market.json`.

## Training and limitations

The frozen corpus contains **2,631 training examples and 295 later validation examples** across event interpretation, SEC changes, macro releases, evidence reconciliation and market synthesis. Training ends December 30, 2024; validation starts January 6, 2025. Source IDs and hashes do not cross the split after deduplication.

Targets were drafted from source evidence using rules and coding-agent review: 578 frozen targets received individual agent review, while 2,348 passed automated checks after drafting-rule review. This is not an independently expert-labeled benchmark. Adjusted price histories are current-vendor reconstructions rather than preserved vendor vintages.

The published run used text-only 16-bit LoRA, completion-only supervision and the non-thinking chat template. It completed 658 scheduled optimizer steps with reported training loss 0.015860. The longest training example is 2,988 tokens; training context was 8,192. Hugging Face manifests retain the recipe and artifact identities.

All **30 adapter checks and five GGUF checks** passed without repair. These checks cover schema, evidence IDs, numerical traceability, termination and repetition. They do not establish financial judgment, causal correctness or investment performance. The model can repeat evidence, underweight a material change, attach an irrelevant caveat or omit a useful connection. Retrieval improves available information; it does not remove the reasoning limits of a 2B model.

Runtime answers receive at most one repair after validation failure. Fact claims must match a cited passage or a checked asset/return pairing, alongside date, unit, hash and citation checks. If repair fails, unsupported claims are removed and the report remains marked incomplete; source excerpts can be shown when no generated claim survives. Generated event interpretations are excluded from later daily-brief evidence. These checks do not prove the correctness of model interpretations or causal explanations. In the local demonstration, the broad market answer retains a partial result; it is not presented as a fully validated synthesis.

## Training and local export

[qwen_train_lora.ipynb](qwen_train_lora.ipynb) is a complete training workflow for **Colab with Google Drive or a local Linux/WSL2 machine**. It trains a new adapter, evaluates it, merges the weights and exports Q4_K_M GGUF locally. It contains no Hugging Face publishing or account-specific recovery steps. Using the published model in Notebook 24 does not require running this training notebook.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ramtin-asadi/Quantitative-Finance-Lab/blob/main/models/qwen_train_lora.ipynb)

The notebook uses Python 3.12 or 3.13, an NVIDIA CUDA GPU with at least 15 GB VRAM, and at least 25 GiB of free disk space. Linux includes WSL2 on Windows. A local environment can be prepared from the repository root:

```bash
python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r models/requirements-training.in jupyterlab
python -m jupyter lab models/qwen_train_lora.ipynb
```

The first cells check the environment. If they install or replace packages, restart the kernel once before continuing. Installation and all corpus checks happen before model loading. The notebook uses the published rank-16 text LoRA recipe, with optional early stopping added for new runs; it is not a replay of the published run. Defaults are two epochs, a learning rate of `5e-5`, batches of one and eight accumulated batches. Set `early_stopping_patience = 0` for a fixed two-epoch schedule.

Training data must contain `train.jsonl`, `validation.jsonl` and a matching frozen `manifest.json`. Each JSONL record follows `quantfinlab.analyst.schemas.TrainingExample`: an evidence packet and answer in `messages`, task, cutoff, source identifiers and hashes, grouping, and review status. Data preparation and review precede training; the notebook does not generate or silently approve its own targets. [prepare_data.py](prepare_data.py) exposes the corpus preparation stages.

For a prepared repository workspace, create a portable bundle:

```bash
python models/package_training.py
```

The archive is written to ignored `workspace/financial_analyst/training_bundle.zip`. It contains the frozen data, notebook and dependency files; it excludes weights, credentials, private paths and preparation scratch files.

For Colab, extract the bundle into `MyDrive/quantfinlab_training`:

```text
quantfinlab_training/
  data/
    train.jsonl
    validation.jsonl
    manifest.json
  qwen_train_lora.ipynb
  requirements-training.in
  requirements-training.lock
  requirements-export.in
  requirements-export.lock
```

Open the notebook using the badge or upload the `.ipynb` through Colab's File menu, select a GPU runtime, and run the cells in order. The badge targets the file on GitHub's `main` branch; uploading the notebook also works for changes that have not yet been published. Drive is mounted when `use_drive = True`. The `drive_project` setting selects the data and output directory. With Drive disabled, Colab files are temporary.

Locally, the notebook reads the repository's frozen corpus or a `data/` folder beside the notebook. The first cell exposes `data_dir`, `run_dir` and `cache_dir`. Checkpoints include the optimizer, scheduler, RNG and AMP scaler; rerunning with the same data and recipe resumes the latest checkpoint. A completed adapter is checksum-verified and reused. Training runs once, with no preliminary smoke-training phase. Finite-loss checks stop real numerical failures, while normal AMP overflow skips are logged without prematurely aborting training.

Logs and loss plots show progress and validation behavior. Thirty representative validation cases check the finished adapter by default. There is at most one repair per case, and completed checks are saved individually. Export uses an isolated CPU converter environment and builds CPU llama.cpp tools, avoiding a CUDA server compilation. The tokenizer is checked before conversion, subprocess output is visible, and verified merged weights, F16 GGUF and Q4_K_M files are reused. Five quantized generation checks run on CPU by default; `validate_gguf = False` skips them and records that the quantized model was not generation-tested.

The final output directory contains:

| Output | Contents |
| --- | --- |
| `adapter/` | LoRA weights and tokenizer |
| `merged/` | Merged 16-bit weights and tokenizer |
| `export/` | Q4_K_M GGUF, checksums, schema, template and manifests |
| `checkpoints/` | Saved training state for resumption |
| `training_log.jsonl`, `trainer_state.json` | Training and evaluation history |
| `adapter_checks.json`, `gguf_checks.json` | Generated answers and check results |
| `export.log`, `gguf_server.log` | Converter, build and quantized-runtime diagnostics |

No publication is required to use the resulting model locally. The base-model cache, run folders and downloaded artifacts are excluded from Git.

## Files and validation

| File | Purpose |
| --- | --- |
| `model_config.json` | Published model identity and runtime settings |
| `published_checks.json` | Acceptance summaries from the pinned public adapter and GGUF revisions |
| `qwen_train_lora.ipynb` | New training, checkpoint resumption and local export |
| `requirements-training.in` / `.lock` | Direct training pins and the resolved dependency snapshot |
| `requirements-export.in` / `.lock` | Separate Python 3.12 CPU converter requirements |
| `prepare_data.py` | Anchor, candidate and frozen-corpus preparation stages |
| `package_training.py` | Portable notebook and frozen-data bundle |

The repository tests use fixtures and fake inference. The final model has also been exercised in Notebook 24, including cached reruns. The reusable training notebook is checked statically and its masking/AMP controls are tested without another training run. Dependency, driver and hardware changes still require environment validation.

Implementation references: [Unsloth Qwen3.5 training](https://unsloth.ai/docs/models/qwen3.5/fine-tune), [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), and [the pinned llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/737e0980fef1c2d573afedc8b00f7caf30617652/tools/server/README.md).
