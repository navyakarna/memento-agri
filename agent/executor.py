from agent.interventions import candidate_interventions

class Executor:
    def run(self, plan):
        context = plan["context"]
        risk = plan["risk"]
        actions = candidate_interventions(context, risk["category"])
        return {
            "candidate_interventions": actions,
            "retrieved_case_ids": [c["case_id"] for c in plan["cases"]],
            "retrieved_cases": plan["cases"]
        }
