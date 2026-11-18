# Agent Contagion Lab: Measuring Prompt-Injection Contagion Across Multi-Agent Topologies

[![LessWrong Post](https://img.shields.io/badge/LessWrong-Read%20Article-2e8555?style=flat-square)](https://www.lesswrong.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

An empirical research laboratory testing whether downstream summarization sanitizes indirect prompt injections across four multi-agent communication topologies (**Chain**, **Star**, **Ring**, and **Tree**).

---

## 📌 Executive Summary

As language model architectures transition from isolated chat assistants to multi-agent pipelines (e.g., LangGraph, AutoGen, CrewAI), the output of an upstream agent becomes the direct operational context of downstream agents.

This repository contains the complete experimental test harness, carrier corpora, and evaluation telemetry for **Indirect Prompt-Injection Contagion** across **240 experimental trials (900 multi-agent interactions)** evaluated on:
* `openai/gpt-oss-20b`
* `openai/gpt-oss-120b`
* `qwen/qwen3.8-27b`

### Key Findings
1. **Summarization Fails as a Security Barrier:** Across all models, passing injected data through successive downstream summarizers does not sanitize the payload. Downstream agents treat the injection as an authoritative system fact and preserve it.
2. **Topological Influence (Model-Dependent):** On `gpt-oss-20b`, **Star** architectures exhibited the highest downstream vulnerability (35.0%) due to unmediated fan-out, whereas **Ring** architectures exhibited the lowest (23.3%) through multi-stage summarization dilution.
3. **The Qwen Effect:** On `qwen3.8-27b`, contagion reached **100% across all topologies and hops**, demonstrating that model-level susceptibility can completely dominate architectural defenses.
4. **Exact Verbatim Propagation:** In 100% of infected outputs across all 900 agent evaluations, the high-entropy canary token was reproduced verbatim (zero near-miss mutations or leetspeak).
5. **Zero False-Positive Baseline:** Control trials (clean carrier text) yielded strictly 0.0% false positives across 450 agent evaluations.

---

## 🏗 Evaluated Communication Topologies

```
1. SEQUENTIAL CHAIN:
   [ Patient Zero: Agent A ] ───> [ Hop 1: Agent B ] ───> [ Hop 2: Agent C ]

2. STAR FAN-OUT:
                                ┌───> [ Satellite 1: Agent B ]
   [ Patient Zero: Agent A ] ───┤
                                └───> [ Satellite 2: Agent C ]

3. CYCLIC RING:
   [ Patient Zero: Agent A ] ───> [ Agent B ] ───> [ Agent C ] ───> [ Agent A (Loop Feedback) ]

4. HIERARCHICAL TREE:
                                ┌───> [ Branch 1: Agent B ] ───> [ Leaf 1: Agent D ]
   [ Patient Zero: Agent A ] ───┤
                                └───> [ Branch 2: Agent C ] ───> [ Leaf 2: Agent E ]
```

---

## 📊 Empirical Results

### Downstream Infection Rate by Topology & Model

| Topology | `gpt-oss-20b` | `gpt-oss-120b` | `qwen3.8-27b` | Aggregate |
|:---|:---:|:---:|:---:|:---:|
| **Star** | 35.0% | 35.0% | 100.0% | **56.7%** |
| **Ring** | 23.3% | 36.7% | 100.0% | **53.3%** |
| **Chain** | 30.0% | 20.0% | 100.0% | **50.0%** |
| **Tree** | 35.0% | 5.0% | 100.0% | **46.7%** |

### Sequential Chain Hop Decay (`gpt-oss-20b`)
* **Hop 0 (Patient Zero):** 50.0%
* **Hop 1 (Agent B):** 30.0% (-20pp attenuation)
* **Hop 2 (Agent C):** 30.0% (plateau)

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/Muhammad-Rebaal/agent-contagion-lab.git
cd agent-contagion-lab

# Set up virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template and populate your API credentials:
```bash
cp .env.example .env
```
Supported backends include Groq, OpenAI, Gemini, OpenRouter, ModelScope, and SiliconFlow.

### 3. Run Experiments
```bash
# Run a quick 3-trial smoke test across all topologies
python runner.py --provider groq --model openai/gpt-oss-20b --smoke-test

# Run full evaluation suite (10 runs per cell)
python runner.py --provider groq --model openai/gpt-oss-20b --runs 10
```

### 4. Statistical Analysis & Plotting
```bash
# Calculate Wilson 95% confidence intervals and Fisher's exact tests
python analysis.py

# Generate 300-DPI publication figures saved to output/plots/
python plots.py
```

---

## 📁 Repository Structure

```
├── agents.py             # Agent definitions & system prompts
├── analysis.py           # Statistical analysis (Wilson CIs, Fisher tests)
├── anticipator_hook.py   # Deterministic boundary defense regex/AST hook
├── config.py             # Global experimental config & provider profiles
├── detectors.py          # 9-stage canary detection engine
├── llm.py                # Multi-provider unified inference client
├── payloads.py           # Summary-resistant prompt-injection templates
├── plots.py              # Publication figure generator
├── runner.py             # CLI test harness and execution engine
├── tasks.py              # Benign enterprise carrier documents
├── tasks_external.py     # Extended carrier corpus (zero-reuse enforcement)
├── topologies.py         # Plain-Python & LangGraph topology executors
├── output/               # Pre-recorded experimental CSV telemetry and plots
├── tests/                # Test suite for detectors and topologies
├── .env.example          # Environment variable template
└── requirements.txt      # Dependency specification
```

---

