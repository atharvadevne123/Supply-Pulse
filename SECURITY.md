# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✓         |

## Reporting a Vulnerability

Please report security vulnerabilities to **devneatharva@gmail.com**.

Do **not** open public GitHub issues for security vulnerabilities.

We will acknowledge your report within 48 hours and provide a fix timeline within 7 business days.

## Security Practices

- All API endpoints validate input via Pydantic models
- Rate limiting is enforced per-IP (default 300 req/min)
- Database credentials must be set via environment variables — never hardcoded
- The `.env.example` file contains placeholder values only
- No PII is stored in prediction logs (only numeric feature vectors)
