# POL-002: Access Request and Provisioning Policy

**Version:** 2.4
**Effective Date:** 2025-02-01
**Owner:** IT Service Management
**Applies To:** All employees, contractors, and interns

## 1. Purpose

This policy governs how new accounts are created, how access to systems and shared resources is granted, and how access is adjusted when an employee changes roles or departments.

## 2. Scope

Covers new account provisioning, additional access requests to existing accounts, contractor account setup, and access changes due to role transfers. Removal of access due to termination is covered by **POL-001 Section 5 (lockouts)** for security-flagged cases and **the Offboarding procedure in this policy (Section 6)** for standard exits.

## 3. Standard Access Requests

Requests for access to non-sensitive shared resources (department shared drives, standard collaboration tools, standard software from the approved catalog) may be granted after:

1. Confirming the request comes from the employee or their manager.
2. Confirming the resource is appropriate for the requester's department and role.

These requests do not require additional written approval and may be resolved directly. Standard priority is **P3**.

## 4. Sensitive System Access

The following systems are classified as **sensitive** and require **dual approval** (requesting manager **and** the system/data owner) before access is granted, regardless of role or seniority:

- Financial reporting and reconciliation systems
- HR information systems containing personal employee data
- Production infrastructure and production databases
- Legal case management and contract repositories
- Any system storing customer payment information

A ticket requesting access to a sensitive system **must be escalated to a human agent** if manager approval alone is present but system-owner approval is missing, or if no approval documentation is attached at all. The AI agent may draft the access request and identify which approvals are missing, but must not grant sensitive access on its own authority.

## 5. Contractor Accounts

Contractor accounts must:

- Be tied to a signed contractor onboarding form specifying scope of access.
- Be scoped to only the resources explicitly listed on the form (least-privilege principle).
- Be created with an automatic expiry date no longer than 90 days, renewable with a new approval.

## 6. Role Transfer and Offboarding

When an employee changes roles or departments:

1. Confirm the new role assignment with the requester's manager.
2. Grant access matching the standard access profile for the new role.
3. Revoke access associated with the previous role that is not part of the new profile, unless a documented business reason for retaining it is provided.

When an employee separates from the company (standard, voluntary exit):

1. Confirm termination date with HR.
2. Schedule account deactivation for end of business on the last working day.
3. Configure mailbox forwarding to the departing employee's manager for 30 days, unless HR specifies otherwise.
4. Generate an equipment return checklist.

**Involuntary terminations** always require immediate (same-hour) deactivation of all accounts, VPN, and badge access upon receipt of a signed HR notice, and must be logged as an escalated action for the compliance audit trail regardless of how quickly it is completed — see **POL-006** for escalation classification.

## 7. Escalation Criteria

- Missing required dual approval for sensitive system access.
- Any offboarding request tied to an involuntary termination.
- Any request where the requester's stated role does not match the HR directory record.
- Contractor access requests lacking a signed onboarding form.

## 8. Related Policies

- POL-001: Password and Account Access Management Policy
- POL-005: Security Incident Response Policy
- POL-006: Ticket Priority, SLA and Escalation Policy
- POL-008: Acceptable Use and Data Handling Policy
