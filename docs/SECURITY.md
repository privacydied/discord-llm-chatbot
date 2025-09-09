# Security

> **Fill me in**
> - [ ] Confirm token rotation policy and retention periods.
> - [ ] Document actual dependency scanning setup.

## Secrets
- Store tokens and API keys in `.env` files or secret managers.
- Rotate credentials regularly and revoke unused tokens.
- Never commit secrets to version control.

## Privileged Intents
- Request only the intents you need.
- Disable unused intents to reduce exposure.

## Webhook & Interaction Verification
If hosting an interactions endpoint, verify signatures using Discord's public key.

## Dependency & Supply Chain
- Use `pip install -r requirements.txt` with pinned versions.
- Enable Dependabot or similar tools for automated updates.
- Consider generating an SBOM for releases.

## Data Retention
- Logs are stored locally in `logs/` and may contain user IDs.
- Prune logs regularly and honour deletion requests.
