def llm_only_baseline(query, row, llm_client):
    context = {k: row[k] for k in [
        "state", "district", "crop", "season", "rainfall_anomaly",
        "temperature_anomaly", "water_stress", "moisture_deficit"
    ] if k in row}
    risk = {"category": "Unknown", "score": None}
    execution = {"candidate_interventions": [], "retrieved_case_ids": [], "retrieved_cases": []}
    return llm_client.generate(query, context, risk, execution)

def similarity_only_score(case):
    return case["similarity"]

def outcome_aware_score(case):
    return case["final_score"]
