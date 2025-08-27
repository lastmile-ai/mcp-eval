# Security Policy

## Supported Versions

We generally support the latest minor release line. Security fixes are applied to the most recent release. If you rely on older versions, consider pinning and testing, but please update when possible.

## Reporting a Vulnerability

If you discover a security issue, please report it responsibly. Do not open a public issue.

- Email: security@lastmileai.dev
- Alternatively: open a private report via GitHub Security Advisories

Please include:
- Affected version(s) and environment (OS, Python version)
- Reproduction steps or proof of concept
- Impact assessment and any suggested mitigations

We will acknowledge receipt within 72 hours, and aim to provide an initial assessment within 7 days. Once the fix is available, we will coordinate a disclosure timeline with you.

## Security Best Practices for Users

- Keep dependencies updated (see `pyproject.toml`)
- Prefer API tokens over passwords; rotate credentials regularly
- Validate and sanitize any external inputs when integrating
- Review `docs/security.mdx` for additional guidance if available

## Coordinated Disclosure

We follow responsible disclosure. Please refrain from sharing details publicly until a fix is released and users have had reasonable time to update.
