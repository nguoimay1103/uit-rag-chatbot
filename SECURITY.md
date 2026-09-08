# Security Policy

## Reporting a vulnerability

Do not disclose exploitable vulnerabilities or credentials in a public issue.
Use GitHub's private vulnerability reporting or a private Security Advisory for
this repository.

Include the affected component, reproduction steps, impact, and a suggested
mitigation when available. Do not include real student data in the report.

## Credential handling

- Keep OpenAI, Qdrant, Supabase service-role, and JWT secrets in environment
  variables managed by the deployment platform.
- Never place service-role credentials in `frontend/index.html`.
- Rotate a credential immediately if it appears in Git history or logs.
- Enforce Supabase Row Level Security for all user-owned chat data.
