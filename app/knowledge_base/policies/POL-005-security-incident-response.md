# POL-005: Security Incident Response Policy

**Version:** 4.0
**Effective Date:** 2025-01-05
**Owner:** IT Service Management (in coordination with the Security function)
**Applies To:** All employees, contractors, and interns

## 1. Purpose

This policy defines mandatory handling for tickets involving suspected phishing, malware, lost or stolen devices, or any other suspected security compromise. It exists to guarantee that these categories are never resolved purely by automated first-line triage.

## 2. Core Principle: Mandatory Human Escalation

**Every ticket in this policy's scope must be escalated to a human agent.** An AI or automated system may perform initial containment steps explicitly listed below (e.g., triggering a remote wipe, disconnecting a device from the network) because these are time-sensitive and reversible-risk-reducing, but it must never independently close or fully resolve a security-classified ticket without human review, and must never tell the requester "this is not a real threat" without a human's confirmation.

## 3. Phishing Reports

1. Do not require the requester to prove the email is malicious; treat all reports as legitimate until reviewed.
2. If the requester reports having clicked a link or entered credentials, treat as a **suspected compromise** — escalate immediately with P2 priority, and advise a precautionary password change while investigation proceeds.
3. If the requester reports the email without having interacted with it, still escalate for review, but this may be handled at P2–P3 depending on the sophistication of the attempt.
4. Low-risk, clearly non-targeted spam (e.g., generic prize scams) may be classified as low-risk and handled with sender blocking, but should still be logged and reviewed by security on a rolling basis rather than fully bypassing review.

## 4. Suspected Malware

1. Any device showing strong indicators of active malware (unexpected popups, browser hijacking, unexplained resource usage, ransomware notices) must be disconnected from the network immediately as a containment step.
2. Escalate immediately at **P1** if there are strong indicators of active infection or ransomware.
3. Escalate at **P2** for lower-confidence signals (e.g., an antivirus tool flagged and quarantined a file automatically before execution).
4. Devices confirmed infected are reimaged; data is restored from the most recent backup only after the device is confirmed clean.

## 5. Lost or Stolen Devices

1. Trigger remote lock/wipe capability immediately upon report, regardless of the likelihood the device will be recovered.
2. Escalate to security incident response at **P1** if the device is a laptop or contains significant company data; **P2** for mobile devices with more limited data exposure.
3. Deactivate associated badge or physical access credentials immediately if the device or accompanying items (e.g., an access badge) were also lost or stolen.
4. Coordinate with the employee on filing a police report where physical theft is involved, for the incident record.

## 6. Account Compromise Indicators

Escalate immediately, without attempting standard password-reset resolution under POL-001, if any of the following are present:

- The account lockout was triggered by a suspicious-activity flag from the identity provider (not just failed password attempts).
- The requester reports login activity they do not recognize.
- The requester reports having entered credentials into a suspicious site.

## 7. Post-Incident Requirements

All security-classified tickets must be logged with a full timeline for the compliance audit trail, regardless of how quickly they were resolved. Resolution speed does not exempt a ticket from escalation or from audit logging.

## 8. Escalation Criteria (Summary)

Every ticket categorized under Security (phishing, malware, lost/stolen device, account compromise indicators) is escalated to a human agent. This is a blanket rule and is not subject to AI confidence-score override — high AI confidence in a resolution does **not** permit skipping escalation for this category.

## 9. Related Policies

- POL-001: Password and Account Access Management Policy
- POL-002: Access Request and Provisioning Policy
- POL-006: Ticket Priority, SLA and Escalation Policy
- POL-008: Acceptable Use and Data Handling Policy
