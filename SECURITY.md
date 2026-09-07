# Security Policy

## Supported versions

This is a single-maintainer research project. Only the latest release on `main`
receives fixes.

| Version | Supported |
|---------|-----------|
| 1.0.x   | yes       |
| < 1.0   | no        |

## Reporting a vulnerability

Please report security issues privately to **sulastomatolog@gmail.com** with
`SECURITY` in the subject line. Do not open a public issue for anything that
looks exploitable.

Useful things to include: affected file or module, the version or commit, what
an attacker gains, and a minimal reproduction.

Expect an acknowledgement within 7 days. There is no bug-bounty programme.

## Scope

This repository contains research and simulation code plus the CCEA
Cloud/Agent boundary. The parts where a report matters most:

- `ccea/`, `packages/agent/` — local vault, risk controls, kill switch, artifact
  signature verification.
- `packages/cloud/` — control plane, authentication, telemetry redaction and DLP.
- `adapters/` — venue credentials handling.

Out of scope: trading losses, model quality, backtest realism, and anything that
requires the attacker to already control the machine running the Agent.

## Handling credentials

The Agent is customer-hosted and holds broker credentials locally. Never commit
a real `.env`; `.env.example` lists the variable names only.
