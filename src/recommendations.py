from __future__ import annotations


def retention_recommendations(customer: dict) -> list[str]:
    """Return practical, non-causal retention ideas based on the supplied profile."""
    actions: list[str] = []
    tenure = int(customer.get("tenure", 0))

    if customer.get("Contract") == "Month-to-month":
        actions.append("Offer a clear, optional 12-month plan with a transparent loyalty benefit.")
    if tenure <= 12:
        actions.append("Start an onboarding check-in and make setup, billing, and support options easy to find.")
    if customer.get("TechSupport") == "No":
        actions.append("Offer a short support-plan trial or a proactive technical health check.")
    if customer.get("OnlineSecurity") == "No" and customer.get("InternetService") != "No":
        actions.append("Explain the available online-security add-on and offer a trial where appropriate.")
    if customer.get("PaymentMethod") == "Electronic check":
        actions.append("Explain automatic payment options and any available convenience benefits without pressure.")
    if float(customer.get("MonthlyCharges", 0)) >= 80:
        actions.append("Review the bill with the customer and suggest a lower-cost plan or right-sized bundle.")
    if customer.get("InternetService") == "Fiber optic":
        actions.append("Check service quality and recent support history before offering an upgrade or add-on.")
    if customer.get("PaperlessBilling") == "Yes":
        actions.append("Use a concise digital bill summary so plan value and recurring charges stay easy to understand.")

    if not actions:
        actions.append("Maintain a positive experience with regular service check-ins and clear plan information.")
    return actions[:5]


def segment_playbook(segment: str) -> tuple[str, str]:
    playbook = {
        "Contract": (
            "Contract flexibility",
            "Explain the value of longer-term plans, but pair any offer with a clear comparison and an easy-to-understand exit policy.",
        ),
        "tenure": (
            "Early customer experience",
            "Prioritise first-year onboarding, first-bill clarity, installation follow-up, and quick access to support.",
        ),
        "PaymentMethod": (
            "Payment convenience",
            "Make payment options simple to compare and resolve failed or confusing payments quickly.",
        ),
        "MonthlyCharges": (
            "Perceived value",
            "Review plan fit, recurring add-ons, and bill clarity; offer a right-sized plan when the current package is not a good match.",
        ),
        "TechSupport": (
            "Support experience",
            "Make troubleshooting easy to access and proactively follow up after unresolved technical issues.",
        ),
        "InternetService": (
            "Service experience",
            "Monitor service quality and help customers understand which services are included in their plan.",
        ),
    }
    return playbook.get(segment, ("Customer experience", "Use targeted check-ins and clear plan information to improve customer confidence."))
