class Planner:
    def plan(self, query, context, risk, cases):
        return {
            "goal": "Provide a localized, evidence-aware climate-resilience advisory.",
            "steps": [
                "Interpret the farmer's crop, location and climate context.",
                "Assess climate risk.",
                "Compare the situation with retrieved historical cases.",
                "Prefer evidence with stronger outcome signals, while not treating correlation as causation.",
                "Combine retrieved evidence with transparent agricultural rules.",
                "Generate practical recommendations and communicate uncertainty."
            ],
            "query": query,
            "context": context,
            "risk": risk,
            "cases": cases
        }
