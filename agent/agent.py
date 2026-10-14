import pandas as pd
from agent.planner import Planner
from agent.executor import Executor
from agent.llm import LLMClient

class MementoAgriAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.planner = Planner()
        self.executor = Executor()
        self.llm = LLMClient()

    @staticmethod
    def row_to_context(row):
        if isinstance(row, pd.Series):
            row = row.to_dict()
        keys = [
            "state", "district", "crop", "season", "year", "soil_type",
            "rainfall", "temperature", "humidity", "ph",
            "wind_speed", "solar_radiation",
            "rainfall_anomaly", "temperature_anomaly", "humidity_anomaly",
            "wind_anomaly", "solar_anomaly",
            "water_stress", "moisture_deficit", "irrigation_percentage"
        ]
        out = {}
        for k in keys:
            v = row.get(k, None)
            if pd.isna(v) if v is not None else False:
                v = None
            out[k] = v.item() if hasattr(v, "item") else v
        # In the processed dataset the "rainfall" column is a static
        # per-crop reference value; the observed rainfall for that
        # state and year is in rainfall_real_mm.
        real_rain = row.get("rainfall_real_mm", None)
        if real_rain is not None and not pd.isna(real_rain):
            out["rainfall"] = float(real_rain)
        return out

    def advise(self, query, row):
        context = self.row_to_context(row)
        risk_score = row.get("climate_risk", None)
        risk_score = (
            0.0 if risk_score is None or pd.isna(risk_score)
            else float(risk_score)
        )
        risk = {
            "score": risk_score,
            "category": str(row.get("risk_category", "Unknown"))
        }
        query_case = {
            "context": context,
            "risk": risk,
            "outcome": {"yield_anomaly": 0}
        }
        cases = self.retriever.retrieve(query_case)
        plan = self.planner.plan(query, context, risk, cases)
        execution = self.executor.run(plan)
        advisory = self.llm.generate(query, context, risk, execution)
        return {
            "advisory": advisory,
            "context": context,
            "risk": risk,
            "cases": cases,
            "plan": plan,
            "execution": execution
        }

    def record_experience(self, cases, action, reward, verified=False):
        for case in cases:
            if case["case_id"] == action.get("case_id"):
                case["experience"] = {
                    "action": action.get("action"),
                    "reward": reward,
                    "verified": verified
                }
