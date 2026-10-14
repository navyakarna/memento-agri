# Memento-Agri Research Protocol

## Research question
Can an LLM agent improve localized climate-resilient agricultural advisory quality through outcome-aware memory while keeping LLM parameters frozen?

## Baselines
1. LLM-only
2. LLM + document RAG
3. LLM + transparent rules
4. LLM + similarity-only case retrieval
5. Memento-Agri: similarity + outcome-aware case retrieval + planner/executor

## Temporal evaluation
Use the oldest observations for memory/training and the newest observations for evaluation. Never randomly mix future observations into historical memory.

## Ablations
- Remove memory
- Random retrieval
- Similarity-only retrieval
- Similarity + outcome-aware retrieval
- Different K values: 1, 3, 4, 5, 10
- Memory sizes: 100, 500, 1k, 5k, 10k when dataset size allows

## Metrics
### ML
- Risk: accuracy, macro-F1, precision, recall, ROC-AUC where applicable
- Yield: MAE, RMSE, R2

### Retrieval
- Recall@K
- Precision@K
- MRR
- outcome-positive fraction among retrieved cases

### Advisory
- relevance 1-5
- locality 1-5
- crop specificity 1-5
- actionability 1-5
- evidence consistency 1-5
- hallucination rate

## Critical methodological rule
Historical yield outcomes do not establish that a recommendation caused the outcome. Do not convert observational rows into fabricated historical interventions. For causal intervention experiments, collect verified action/outcome records or explicitly run a simulation with a declared intervention policy.

## Suggested paper contribution
Outcome-aware, weight-free continual adaptation: the memory changes, not the LLM parameters.
