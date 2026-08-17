Here is a comprehensive, production-grade **`README.md`** designed to highlight the architecture, engineering depth, and resume value of your project.

---

### Step 1: Update `README.md`

Open `README.md` in the root of your project directory and replace its contents with:

```markdown
# ⚡ Autonomous Agentic MLOps Engine

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq LPU](https://img.shields.io/badge/LLM%20Inference-Groq%20Cloud-black.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous, multi-agent MLOps framework that profiles tabular data, formulates machine learning strategies, synthesizes modular pipeline code, and iteratively debugs execution failures in a sandboxed runtime environment using a self-healing feedback loop.

---

## 🏗️ System Architecture


```

```
                              +-------------------+
                              |   Raw CSV Data    |
                              +---------+---------+
                                        |
                                        v
                             +---------------------+
                             | 1. Profiler Node    |
                             | (Stat / Missingness)|
                             +----------+----------+
                                        |
                                        v
                             +---------------------+
                             | 2. Strategy Agent   |
                             | (Pydantic JSON Spec)|
                             +----------+----------+
                                        |
                                        v
                             +---------------------+
                             | 3. Coder Agent      |
                             | (Python Synthesizer)|
                             +----------+----------+
                                        |
                                        v
                             +---------------------+
                 +---------->| 4. Sandbox Runner   |<---------+
                 |           | (Isolated Subprocess|          |
                 |           +----------+----------+          |
                 |                      |                     |
                 |               [Exit Code == 0?]            |
                 |                      |                     |
    (Failure / Traceback)         +-----+-----+         (Fix & Retry)
                 |                |           |               |
                 v               YES          NO              |
      +---------------------+     |           |               |
      | 5. Debugger Agent   |     |           +---------------+
      | (Self-Healing Loop) |     |       (Max Retries <= 3)
      +---------------------+     v
                           +---------------------+
                           | 6. Exporter Node    |
                           | (Pipeline & PDF Gen)|
                           +----------+----------+
                                      |
                                      v
                           +---------------------+
                           |  Production Assets  |
                           | (PDF, Code, Model)  |
                           +---------------------+

```

```

---

## ✨ Core Features & Agent Roles

* **Dataset Profiler (`profiler/`)**: Extracts summary statistics, type inferences, cardinality counts, skewness, class distributions, and missingness metrics deterministically without LLM hallucinations.
* **Strategy Agent (`agents/strategy_agent.py`)**: Consumes the JSON profile and generates a structured, strict-schema ML plan using Pydantic (data cleaning, feature encoding, scaling, cross-validation, model selection).
* **Code Generator Agent (`agents/code_generator.py`)**: Synthesizes clean, production-grade Scikit-Learn code based on the JSON strategy.
* **Execution Sandbox Runner (`runner/sandbox.py`)**: Safely runs generated scripts in an isolated subprocess with timeout guards, capturing `stdout` and `stderr` without risking host process corruption.
* **Self-Healing Debugger (`agents/debugger_agent.py`)**: Intercepts runtime stack traces (e.g., `ValueError`, `KeyError`, `IndexError`, fold mismatches), analyzes the failure context, patches the script, and re-submits it to the sandbox automatically.
* **Artifact Exporter & PDF Generator (`exporter/`)**: Exports validated Python pipeline scripts and generates an executive PDF summary using ReportLab containing run metrics, profiling summaries, and architectural decisions.
* **Interactive UI (`app.py`)**: Streamlit web dashboard providing drag-and-drop ingestion, real-time agent execution status, code viewer, and artifact downloads.

---

## 🛠️ Tech Stack

| Domain | Technologies Used |
| :--- | :--- |
| **Agentic Framework** | LangGraph, LangChain Core |
| **LLM Inference** | Groq Cloud (`qwen/qwen3.6-27b` / `llama-3.3-70b-versatile`) |
| **Data & ML** | Pandas, NumPy, Scikit-Learn, Joblib |
| **Execution & Sandbox** | Python Subprocess, Safe Timeout Wrappers |
| **Reporting & UI** | ReportLab (PDF Generation), Streamlit |

---

## 📁 Repository Structure

```text
agentic-mlops-engine/
├── agents/
│   ├── __init__.py
│   ├── code_generator.py      # Synthesizes Scikit-Learn pipelines
│   ├── debugger_agent.py      # Self-healing runtime error patcher
│   └── strategy_agent.py      # Structured Pydantic strategy planner
├── artifacts/                 # Generated outputs (ignored from git tracking)
│   ├── pipeline.py
│   └── summary_report.pdf
├── data/
│   └── sample_dataset.csv     # Sample tabular test data
├── exporter/
│   ├── __init__.py
│   └── report_generator.py    # ReportLab automated PDF builder
├── profiler/
│   ├── __init__.py
│   └── dataset_profiler.py    # Deterministic statistical profiler
├── runner/
│   ├── __init__.py
│   └── sandbox.py             # Subprocess execution harness
├── workflow/
│   ├── __init__.py
│   ├── graph.py               # LangGraph cyclic state machine
│   └── state.py               # TypedDict state definition
├── .env.example               # Template for API keys
├── .gitignore
├── app.py                     # Streamlit web interface
├── requirements.txt
└── README.md

```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone [https://github.com/raghav154m/agentic-mlops-engine.git](https://github.com/raghav154m/agentic-mlops-engine.git)
cd agentic-mlops-engine

```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key_here

```

### 4. Run via Terminal (LangGraph State Machine)

```bash
python3 -m workflow.graph

```

### 5. Run via Interactive Streamlit UI

```bash
streamlit run app.py

```

---

## 🔄 Self-Healing Loop in Action

When an execution edge-case occurs (e.g., small dataset class counts breaking a 5-fold cross-validation split):

```text
⚙️ [Node 4: Sandbox Runner] Executing script (Attempt 1)...
❌ Runtime error intercepted: ValueError: n_splits=5 cannot be greater than the number of members in each class.

🛠️ [Node 5: Debugger] Diagnosing error & patching code (Retry #1)...
⚙️ [Node 4: Sandbox Runner] Executing script (Attempt 2)...
✅ Pipeline executed successfully in sandbox!

📦 [Node 6: Exporter] Exporting pipeline code and PDF report...
✅ Exported script: artifacts/pipeline.py
✅ Exported report: artifacts/summary_report.pdf

```

---
