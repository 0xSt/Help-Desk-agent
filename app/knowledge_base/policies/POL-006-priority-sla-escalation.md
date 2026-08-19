# POL-006: Ticket Priority, SLA and Escalation Policy

**Version:** 3.5
**Effective Date:** 2025-01-05
**Owner:** IT Service Management
**Applies To:** All IT Help Desk tickets, including those triaged by the AI Help Desk Agent

## 1. Purpose

This is the master policy governing ticket priority levels, service level agreement (SLA) targets, and the conditions under which a ticket must be escalated (handed off) from automated handling to a human IT agent. Every other policy's escalation criteria are consistent with, and subordinate to, this policy. Where another policy and this policy appear to conflict, this policy governs.

## 2. Priority Levels and SLA Targets

| Priority | Definition | Response Target | Resolution Target |
|---|---|---|---|
| **P1 – Critical** | Business-critical outage, active security incident, multi-user impact, or complete inability to work with no workaround | 30 minutes | 4 business hours |
| **P2 – High** | Single user blocked from critical work; suspected security issue with limited scope; time-sensitive business need | 2 business hours | 8 business hours |
| **P3 – Medium** | Non-blocking issue; user can continue working with a workaround or reduced functionality | 4 business hours | 2 business days |
| **P4 – Low** | How-to questions, minor cosmetic issues, non-urgent requests | 1 business day | 5 business days |

## 3. Mandatory Escalation Triggers

The following conditions require escalation to a human agent **regardless of the AI agent's confidence level**:

1. **Security-classified tickets** — any ticket in the Security category (phishing, malware, lost/stolen device, account compromise indicators) per POL-005.
2. **Involuntary termination offboarding** — per POL-002 Section 6.
3. **Sensitive system access requests** missing required dual approval — per POL-002 Section 4.
4. **Spend-threshold approvals** — hardware, software, or license requests exceeding the thresholds defined in POL-003 and POL-004 without documented approval attached.
5. **Explicit human request** — the requester asks to speak with a human agent, expresses that automated help is not working, or explicitly declines an AI-provided resolution.
6. **Repeated recurrence** — the same underlying issue has been reopened by the same requester more than twice within a 7-day window.
7. **Multi-user or infrastructure-wide impact** — an issue reported by one user that investigation suggests may be affecting multiple users or core infrastructure (e.g., a VPN gateway outage).
8. **SLA breach risk** — a ticket approaching 80% of its resolution target time without a resolution in place.

## 4. Confidence-Based Escalation

For tickets not covered by a mandatory trigger in Section 3, the AI agent must escalate to a human agent when:

- Its classification confidence score for category/priority falls below **0.65**, or
- Its retrieval from both knowledge bases (past tickets and policies) returns no result above the minimum similarity threshold configured for the system (i.e., it has no clear precedent or policy basis for a resolution), or
- The retrieved policy guidance is ambiguous, contradictory, or does not clearly cover the specific scenario described.

In these cases the agent should not guess; escalation is preferred over a low-confidence automated resolution.

## 5. Escalation Handoff Procedure

When a ticket is escalated:

1. The ticket status changes to `ESCALATED` and is placed into the human agent queue.
2. The full context — original request, AI classification, retrieved past-ticket examples, retrieved policy excerpts, draft resolution (if any), and the specific escalation reason — is attached to the ticket so the human agent does not have to start from scratch.
3. The requester is informed that their ticket has been passed to a human agent, with an updated expected response time based on the ticket's priority.
4. Once the human agent resolves the ticket, the outcome should be captured so it can be added to the historical ticket knowledge base for future reference.

## 6. De-escalation / Return to Automated Handling

A ticket handled by a human agent is not automatically returned to the AI agent. However, once resolved, it becomes a candidate example for the past-ticket knowledge base, improving future automated handling of similar cases.

## 7. Related Policies

This policy references and is referenced by all other IT Help Desk policies. See POL-001 through POL-005, POL-007, and POL-008 for domain-specific procedures that feed into these escalation rules.
