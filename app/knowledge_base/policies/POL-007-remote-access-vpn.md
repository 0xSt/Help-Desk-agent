# POL-007: Remote Access and VPN Policy

**Version:** 1.6
**Effective Date:** 2025-02-15
**Owner:** IT Service Management
**Applies To:** All employees using VPN or remote network access

## 1. Purpose

This policy governs troubleshooting and access rules for VPN connectivity, remote network access, and related connectivity issues (Wi-Fi, network drives) encountered while working remotely or while traveling.

## 2. Standard VPN Troubleshooting Sequence

1. Confirm the issue is isolated to one user before assuming a client-side fault (check for known gateway-wide issues first).
2. For connection timeouts: confirm the required VPN port is not blocked by the local network (common at client sites, hotels, and public Wi-Fi); recommend the alternate gateway configured for external/travel use if the primary gateway is blocked.
3. For "connected but unreachable internal resources" symptoms: reset the VPN client's DNS configuration, as this is the most common root cause.
4. For persistent failures after basic steps: escalate to network engineering.

## 3. Multi-User or Infrastructure Impact

If more than one user reports VPN issues within a short window, or if a single report includes signals suggesting a broader outage (e.g., a whole department affected, or timing tied to a business-critical process such as month-end close), this must be treated as a potential infrastructure incident and escalated immediately as **P1** per POL-006 Section 3, rather than handled as an individual connectivity ticket.

## 4. Wi-Fi Connectivity

1. Check for known access point issues in the requester's location before troubleshooting the device.
2. For intermittent drops: reset the saved network profile and update the wireless adapter driver.
3. For weak signal in a specific physical location: log a facilities ticket for access point placement review; provide a temporary wired connection as a workaround where available.
4. Guest Wi-Fi access codes may be issued directly upon request without additional approval, using the standard time-limited guest access process.

## 5. Shared Drive and Network Performance

1. Check for scheduled jobs (backups, syncs) that may be consuming bandwidth during business hours before assuming a network fault.
2. Check for recent permission or group membership changes if the issue is "access denied" rather than "slow performance."

## 6. Escalation Criteria

- Any indication of multi-user or infrastructure-wide impact (see Section 3).
- VPN or connectivity issues tied to a business-critical time window (financial close, product launch, executive travel) that cannot be resolved within the standard SLA response target.
- Persistent failures that remain unresolved after the standard troubleshooting sequence has been fully attempted.

## 7. Related Policies

- POL-006: Ticket Priority, SLA and Escalation Policy
- POL-005: Security Incident Response Policy (for suspected unauthorized VPN access)
