# POL-001: Password and Account Access Management Policy

**Version:** 3.2
**Effective Date:** 2025-01-15
**Owner:** IT Service Management
**Applies To:** All employees, contractors, and interns with a company account

## 1. Purpose

This policy defines how password resets, account lockouts, and multi-factor authentication (MFA) issues are handled by the IT Help Desk, and sets identity-verification standards that must be followed before any credential is reset or reissued.

## 2. Scope

This policy covers first-line account access issues that do not involve a suspected security compromise. If there is any indication of unauthorized access, stolen credentials, or account compromise, the ticket must instead be handled under **POL-005 (Security Incident Response Policy)**.

## 3. Identity Verification Requirements

Before performing any password reset, unlock, or MFA reissue, the Help Desk (human or automated) must verify the requester's identity using at least one of the following:

- Confirmation through the requester's registered personal recovery email or phone number.
- Verbal confirmation from the requester's direct manager (required for phone/email channel requests where self-service verification is not possible).
- Employee badge or ID number cross-checked against the HR directory.

Identity verification may **not** be skipped, even for urgent or executive requests. Urgency justifies expediting the process, not skipping verification.

## 4. Password Reset Procedure

1. Verify identity per Section 3.
2. Confirm the account is not already locked due to repeated failed attempts (see Section 5).
3. Issue a reset via the self-service link (preferred) or a temporary password with forced change at next login.
4. Confirm the requester can successfully log in before closing the ticket.

Standard password resets are **Priority 3 (P3)** unless the requester demonstrates a time-critical business need (e.g., blocking an imminent meeting, deadline, or onboarding session), in which case they may be handled as **P2**.

## 5. Account Lockout Procedure

Accounts lock automatically after 5 consecutive failed login attempts within a 15-minute window. To unlock:

1. Verify identity per Section 3.
2. Confirm the lockout reason (failed attempts, expired credential, or suspicious pattern flagged by the identity provider).
3. If the lockout was triggered by a suspicious pattern flag (as opposed to ordinary mistyped passwords), escalate to POL-005 for security review before unlocking.
4. Otherwise, unlock the account and advise the requester on any root cause found (e.g., outdated saved password in a browser).

## 6. Multi-Factor Authentication (MFA) Issues

- Lost or broken MFA devices: verify identity using an enhanced method (video confirmation plus badge/ID, or in-person verification) before issuing a temporary backup code. Backup codes are valid for 24 hours only and must prompt a follow-up task for the employee to re-enroll a new device.
- Time-sync or code-mismatch issues: guide the requester to enable automatic time synchronization on their device before making any account changes.

## 7. Escalation Criteria

Escalate to a human agent (do not resolve automatically) when any of the following apply:

- The account lockout was triggered by a suspicious-activity flag rather than ordinary failed attempts.
- The requester cannot be verified through any of the standard methods in Section 3.
- The same account has required more than 2 password resets or unlocks within a 7-day period (possible sign of a deeper issue or compromise).
- The request involves an executive-level account and a security-sensitive system (see POL-002).

## 8. Related Policies

- POL-002: Access Request and Provisioning Policy
- POL-005: Security Incident Response Policy
- POL-006: Ticket Priority, SLA and Escalation Policy
