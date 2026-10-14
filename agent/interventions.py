def candidate_interventions(context, risk_category):
    """Transparent rule layer. These are candidate actions, not claimed historical actions.

    Anomalies are fractional deviations from normal (-0.25 = 25% below normal).
    """
    crop = str(context.get("crop", "")).lower()
    irrigation = context.get("irrigation_percentage")
    rainfall_anomaly = float(context.get("rainfall_anomaly", 0) or 0)
    temp_anomaly = float(context.get("temperature_anomaly", 0) or 0)

    actions = []

    # Drought-side actions: rainfall deficit, or high risk that is not caused by excess rain.
    drought_risk = risk_category in {"High", "Severe"} and rainfall_anomaly <= 0.20
    if drought_risk or rainfall_anomaly < -0.15:
        actions += [
            "Prioritize irrigation scheduling according to local water availability.",
            "Monitor soil moisture and avoid unnecessary irrigation losses.",
            "Consider locally recommended drought/heat-tolerant varieties for the next sowing window."
        ]

    if rainfall_anomaly > 0.20:
        actions += [
            "Keep drainage channels clear and watch low-lying fields for waterlogging.",
            "Inspect the crop for fungal and moisture-related diseases after heavy rain."
        ]

    if temp_anomaly > 0.10:
        actions += [
            "Monitor the crop during heat-stress periods and follow locally validated heat-management practices."
        ]

    if crop in {"wheat", "maize", "rice"}:
        actions.append("Use crop- and growth-stage-specific recommendations from a local agricultural extension source.")

    # Irrigation coverage is not in every dataset; only use the rule when it is known.
    if irrigation is not None and float(irrigation) < 30:
        actions.append("Prioritize water-conservation practices because irrigation access appears limited.")

    # Deduplicate while preserving order.
    return list(dict.fromkeys(actions))
