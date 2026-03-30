"""
Deterministic synthetic ticket generator for reproducible benchmarks.
All randomness is seeded so baseline scores are reproducible.
"""

from __future__ import annotations

import random
from typing import List, Optional

from env.models import (
    SupportTicket, Department, Priority, SentimentLabel
)


# ---------------------------------------------------------------------------
# Template pools
# ---------------------------------------------------------------------------

BILLING_TICKETS = [
    ("Incorrect charge on my account", "I was charged twice for my subscription this month. Please refund the extra $29.99 immediately. This is unacceptable."),
    ("Upgrade invoice question", "Hi, I upgraded from Pro to Enterprise last week and got an invoice but the proration seems wrong. Can someone check?"),
    ("Failed payment", "My payment keeps failing but my card is valid. I've tried 3 times. Please help."),
    ("Request for annual invoice", "Could you send me a consolidated annual invoice for our records? We need it for accounting."),
    ("Unexpected price increase", "My bill went up by $50 this month with no notice. What changed? I want an explanation before I pay."),
]

TECHNICAL_TICKETS = [
    ("API returning 500 errors", "Our integration has been throwing 500 errors since 2 AM UTC. This is breaking production for our clients. We need urgent help."),
    ("Webhook not firing", "Webhooks stopped firing about 3 hours ago. I've checked our endpoint and it's healthy. Is something wrong on your end?"),
    ("Login loop bug", "After updating my password I'm stuck in a redirect loop. Cleared cache, tried incognito, still broken."),
    ("Export feature broken", "The CSV export button does nothing when clicked. Chrome + Windows 11. Started today."),
    ("Performance degradation", "Our dashboard has been loading very slowly (10-15s) for the past 2 days. Usually takes under 2s."),
    ("SSO configuration help", "We're setting up SAML SSO with Okta and getting an assertion error. I can share logs if helpful."),
]

SHIPPING_TICKETS = [
    ("Order not delivered", "My order was supposed to arrive 5 days ago. Tracking shows it's been sitting at a distribution center. Please help."),
    ("Wrong item shipped", "I ordered the blue version but received the red one. Need a replacement ASAP."),
    ("Missing package", "Tracking says delivered but nothing is here. My neighbor hasn't seen it either."),
    ("Express shipping charged but used standard", "I paid for 2-day shipping but it took 7 days. Please refund the shipping difference."),
    ("International customs delay", "My order has been stuck in customs for 2 weeks. What can you do to help?"),
]

RETURNS_TICKETS = [
    ("Return label request", "I'd like to return my purchase from last week. How do I get a return label?"),
    ("Refund status", "I returned my item 10 days ago and still haven't received my refund. Order #RET-4892."),
    ("Exchange request", "Product arrived damaged. I'd like an exchange, not a refund. Please advise on next steps."),
    ("Return window exception", "I know I'm past the 30-day window but the product failed after only light use. Can you make an exception?"),
]

GENERAL_TICKETS = [
    ("How to export data", "Where can I find the option to export all my data? I've looked in settings but can't find it."),
    ("Account deletion request", "I'd like to permanently delete my account and all associated data per GDPR."),
    ("Feature request", "It would be great if you could add dark mode. Many users in your community are asking for it too."),
    ("Partnership inquiry", "We're a mid-sized SaaS company interested in a reseller partnership. Who should I contact?"),
    ("Two-factor authentication setup", "I want to enable 2FA on my account. The docs say to go to Security Settings but I don't see that option."),
]

ALL_POOLS = {
    Department.BILLING: BILLING_TICKETS,
    Department.TECHNICAL: TECHNICAL_TICKETS,
    Department.SHIPPING: SHIPPING_TICKETS,
    Department.RETURNS: RETURNS_TICKETS,
    Department.GENERAL: GENERAL_TICKETS,
}

CUSTOMER_TIERS = ["free", "pro", "enterprise"]
TIER_WEIGHTS = [0.5, 0.35, 0.15]

TAGS_BY_DEPT = {
    Department.BILLING: ["payment", "invoice", "refund", "subscription"],
    Department.TECHNICAL: ["bug", "api", "integration", "performance", "auth"],
    Department.SHIPPING: ["delivery", "tracking", "lost", "damaged"],
    Department.RETURNS: ["return", "refund", "exchange", "warranty"],
    Department.GENERAL: ["account", "feature", "data", "question"],
}

# Sentiment distribution varies by department
SENTIMENT_BY_DEPT = {
    Department.BILLING:  [SentimentLabel.ANGRY, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL, SentimentLabel.POSITIVE],
    Department.TECHNICAL: [SentimentLabel.NEGATIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL, SentimentLabel.ANGRY],
    Department.SHIPPING:  [SentimentLabel.NEGATIVE, SentimentLabel.ANGRY, SentimentLabel.NEUTRAL, SentimentLabel.NEUTRAL],
    Department.RETURNS:   [SentimentLabel.NEUTRAL, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL, SentimentLabel.POSITIVE],
    Department.GENERAL:   [SentimentLabel.POSITIVE, SentimentLabel.NEUTRAL, SentimentLabel.NEUTRAL, SentimentLabel.NEGATIVE],
}


def generate_tickets(
    n: int,
    seed: int = 42,
    dept_distribution: Optional[dict] = None,
    force_departments: Optional[List[Department]] = None,
) -> List[SupportTicket]:
    """
    Generate n synthetic support tickets deterministically.

    Args:
        n: Number of tickets to generate.
        seed: RNG seed for reproducibility.
        dept_distribution: Optional dict mapping Department → weight.
        force_departments: If set, cycle through these departments in order.
    """
    rng = random.Random(seed)
    tickets = []

    depts = list(ALL_POOLS.keys())
    if dept_distribution:
        dept_weights = [dept_distribution.get(d, 1) for d in depts]
    else:
        dept_weights = [1] * len(depts)

    for i in range(n):
        if force_departments:
            dept = force_departments[i % len(force_departments)]
        else:
            dept = rng.choices(depts, weights=dept_weights, k=1)[0]

        subject, body = rng.choice(ALL_POOLS[dept])
        tier = rng.choices(CUSTOMER_TIERS, weights=TIER_WEIGHTS, k=1)[0]
        sentiment = rng.choice(SENTIMENT_BY_DEPT[dept])
        prev = rng.randint(0, 8)
        wait = rng.randint(0, 480)  # 0–8 hours
        num_tags = rng.randint(1, 3)
        tags = rng.sample(TAGS_BY_DEPT[dept], min(num_tags, len(TAGS_BY_DEPT[dept])))

        tickets.append(SupportTicket(
            ticket_id=f"TKT-{seed:04d}-{i:04d}",
            subject=subject,
            body=body,
            customer_tier=tier,
            sentiment=sentiment,
            previous_contacts=prev,
            wait_time_minutes=wait,
            tags=tags,
            metadata={"dept_hint": dept.value},  # used by grader only
        ))

    return tickets
