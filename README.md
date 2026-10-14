# Memento-Agri

Memento-Agri is a research prototype for climate-resilient agricultural advisory. It combines historical crop-climate experiences with a memory-augmented retrieval pipeline and a simple planner/executor pattern, so the system can answer a farmer's question using context from past comparable situations instead of relying only on the LLM's general knowledge.

The project is designed around a core idea:

- The model does not invent historical intervention outcomes.
- It retrieves similar historical situations from a structured case bank.
- It scores those cases using climate similarity and observed yield outcomes.
- It generates candidate actions from a transparent rule layer.
- It asks an LLM to turn all of that into a practical advisory message.

This project is intentionally transparent and research-oriented, not a production agricultural decision system.

---

## Why this project exists

Agricultural advice is highly local and context-dependent. A farmer's crop performance depends on location, seasonal rainfall, irrigation capability, climate anomalies, soil conditions, and crop-specific stress patterns. In many real datasets, the historical record contains observations and outcomes but not a trustworthy log of what management actions were actually taken.

Because of that, the project does not claim to know the exact historical intervention that caused a crop outcome. Instead, it uses the available data to do three things:

1. Build a memory of past cases.
2. Retrieve similar historical situations.
3. Generate a cautious advisory based on risk, precedent, and rule-based candidate actions.

---

## High-level system overview

The system is made up of five connected layers:

1. Data preprocessing
2. Climate feature engineering and risk scoring
3. Case-bank construction
4. Memory retrieval with outcome-aware reranking
5. Planner, executor, and advisory generation

The main execution flow is:

- Load and clean the raw agricultural dataset
- Compute rainfall and climate anomalies from real state/crop historical rainfall where available
- Calculate climate-risk and yield-anomaly features
- Convert each row into a case object with context, risk, and outcome
- Embed all case text into a vector index
- When a farmer asks a question, retrieve similar past cases
- Rank them by text similarity, climate similarity, and yield outcome signal
- Generate candidate interventions with transparent rules
- Use an LLM (or the deterministic local generator) to produce the final advisory

---

## Data pipeline

The project starts from a tabular agriculture dataset and normalizes it into a cleaned, processed dataset. The relevant processing logic is in:

- config.py
- data/preprocess.py
- data/real_climate.py

### Important preprocessing steps

- column alias normalization
- numeric cleaning
- removal of impossible values
- real historical rainfall integration from IMD-style climate records
- climate anomaly calculations
- yield-baseline estimation and yield-anomaly calculation
- climate stress score and risk category assignment
- case ID generation

### Climate and risk features

The project computes several derived features, including:

- rainfall_anomaly
- rainfall_anomaly_z
- temperature_anomaly
- humidity_anomaly
- wind_anomaly
- solar_anomaly
- yield_baseline
- yield_anomaly
- rainfall_stress
- climate_risk
- risk_category

The project uses fractional anomalies in the same convention throughout the codebase:

- -0.25 means 25% below normal
- +0.25 means 25% above normal

`rainfall_anomaly_z` is used for the standardized rainfall stress component, while the other anomaly columns are used as fractional signals in the retrieval and advisory logic.

---

## Memory construction

The memory system turns processed rows into reusable historical cases.

Relevant files:

- memory/case_bank.py
- memory/retriever.py
- memory/cases.json
- memory/cases.faiss
- memory/case_metadata.json

### What a case contains

Each case includes:

- case_id
- context (state, district, crop, season, year, rainfall anomaly, temperature anomaly, etc.)
- risk score and category
- observed outcome (yield, yield baseline, yield anomaly)
- optional experience metadata
- embedded text for retrieval

The case text is built to be semantically meaningful and searchable, for example:

- crop
- state
- district
- rainfall condition
- temperature condition
- climate risk category

This makes the memory function like a case-based memory for past agricultural situations.

### Retrieval logic

The memory retrieval process does not simply use a raw vector similarity search. It combines multiple signals:

- text similarity between the incoming query and historical cases
- climate match using rainfall anomaly similarity
- outcome score from yield anomaly
- optional verified experience boosts

The final ranking is a weighted scoring formula:

- text similarity contributes most
- climate match contributes strongly
- outcome signal contributes modestly
- verified experience adds a small bonus

This is implemented in `memory/retriever.py` and is one of the key ideas behind the project.

---

## Agent architecture

Relevant files:

- agent/agent.py
- agent/planner.py
- agent/executor.py
- agent/interventions.py
- agent/llm.py

### Planner

The planner determines the high-level strategy for answering the farmer's question. It is intentionally simple and explicit:

- interpret the farmer context
- assess climate risk
- compare with retrieved cases
- prefer evidence with outcome signals
- combine memory with transparent rules
- produce a practical recommendation

### Executor

The executor runs the plan and creates candidate interventions using the rule layer. It takes the current context and risk category and emits candidate actions such as:

- irrigation scheduling
- moisture conservation measures
- drainage management
- heat-stress monitoring
- crop- and growth-stage-specific guidance

These are generated from deterministic rules, not by guessing historical actions.

### LLM layer

The LLM layer is intentionally flexible:

- If `OPENAI_API_KEY` is set, the system will call OpenAI
- If no key is set, it falls back to a deterministic local advisory generator

This makes the project runnable in a local/offline environment while still allowing real LLM support when available.

The local generator summarises:

- the climate-risk interpretation
- dominant stress factors
- practical recommendations
- crop-specific recommendations
- monitoring priorities
- uncertainty and caution

---

## Why the project is careful about causality

The historical dataset contains observations and outcomes, but not a fully validated record of which interventions farmers actually used. This means the system does not claim that a historical pattern proves that a specific action caused a yield outcome.

This code therefore uses a strict separation:

- retrieved historical experiences are used as memory and evidence
- candidate interventions are generated by a transparent rule layer
- the final advice is phrased as practical guidance, not a proven causal claim

This is a core design principle of the repository.

---

## Models and evaluation

The repository also includes model experiments and baseline comparisons.

Relevant files:

- models/risk_model.py
- models/yield_model.py
- evaluation/baselines.py
- evaluation/metrics.py
- experiments/run_ablation.py
- experiments/run_memory_growth.py
- experiments/run_temporal_split.py

These scripts evaluate:

- risk prediction quality
- yield prediction quality
- baseline comparisons
- ablation experiments
- memory growth behavior
- temporal split robustness

Outputs are saved under the outputs directory.

---

## Project structure

```text
Memento-Agri-main/
├── app/
│   └── streamlit_app.py
├── agent/
│   ├── agent.py
│   ├── executor.py
│   ├── llm.py
│   ├── interventions.py
│   ├── planner.py
│   └── __init__.py
├── data/
│   ├── preprocess.py
│   ├── real_climate.py
│   ├── raw/
│   └── processed/
├── evaluation/
├── experiments/
├── memory/
│   ├── case_bank.py
│   ├── retriever.py
│   ├── cases.json
│   ├── cases.faiss
│   └── case_metadata.json
├── models/
├── outputs/
├── config.py
├── main.py
├── README.md
├── requirements.txt
├── RESEARCH_PROTOCOL.md
└── app.py or streamlit app entry points
```

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project is intended to work with Python 3.10+ and has been tested with modern Python versions in the environment used for this repository.

---

## Build the project

The main pipeline commands are:

```bash
python main.py build
python main.py train-risk
python main.py train-yield
python main.py build-memory
python main.py demo
```

### What each command does

- `build` preprocesses the raw dataset into the processed dataset
- `train-risk` trains and evaluates risk models
- `train-yield` trains and evaluates yield models
- `build-memory` creates the vector memory and case metadata
- `demo` runs a sample advisory using the last row of the processed data

---

## Run the Streamlit application

Start the app:

```bash
streamlit run app/streamlit_app.py
```

The app loads:

- the processed climate dataset
- the FAISS or NumPy memory index
- the agent and retriever

Then it lets a user enter crop and area information and generate a climate advisory.

---

## Optional LLM mode

To enable OpenAI-backed responses:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4.1-mini"
python main.py demo
```

If no key is set, the project automatically uses the deterministic local generator so the system remains runnable without external APIs.

---

## Free deployment options

This repository can be deployed in a few free ways:

### 1. Local machine exposure

Run the app locally and expose it via a tunnel, for example with Cloudflare Tunnel.

### 2. Streamlit Community Cloud

This is the cleanest free hosting path if the repo is public on GitHub:

- push the repo to GitHub
- deploy from Streamlit Community Cloud
- select the app entry file
- set any required env vars

### 3. Temporary public demo tunnel

For short-lived demos, a temporary Cloudflare tunnel can proxy the local app to the public internet without paid hosting.

---

## Research and limitations

This repository is a research prototype and should not be treated as a production advisory system. Some important limitations are:

- no verified historical intervention labels are available
- recommendations are generated from transparent rules and memory, not inferred causal policy
- climate data and crop data are historical and context specific
- local advisory should still be validated by agricultural experts, extension officers, and government guidance

---

## Disclaimer

This project is for research and experimentation. It is not a substitute for agronomic expert advice, local agronomy guidance, or a production agricultural decision-support platform.

Use the recommendations as one input among many, especially in real-world farm management decisions.

---

## Recommended next steps

- run the preprocessing pipeline
- generate the memory index
- launch the Streamlit app
- test with different crop and district combinations
- add your own local agronomic rules if you want domain-specific recommendations
- connect the app to a public deployment platform if you want broader access

If you want to extend the system, the most valuable next steps are:

- add more explicit local agronomic rules
- incorporate irrigation and soil metadata more deeply
- evaluate against more rigorous temporal backtests
- compare memory retrieval strategies with and without outcome-aware reranking
