# Deployment

> **Fill me in**
> - [ ] Provide tested commands for each platform.
> - [ ] Document health probes and scaling thresholds.

## Docker
- Use the `docker run` or `docker compose` examples from [Installation](INSTALLATION.md#docker).
- Expose port `8000` if Prometheus metrics are enabled.
- Restart policy: `unless-stopped`.

## Render
- Create a new Web Service using the Docker image.
- Add environment variables via the dashboard.
- Configure health checks to hit `/metrics` if enabled.

## Fly.io
- Use `fly launch` to generate `fly.toml`.
- Scale with `fly scale count <n>`.
- Store secrets with `fly secrets set`.

## Heroku
- Deploy using the container registry:
```bash
heroku container:push web -a <app>
heroku container:release web -a <app>
```

## VPS
- Provision the host with Python and Docker.
- Run the bot as a systemd service for automatic restarts.

## Kubernetes
- Create a Deployment and Service manifest.
- Mount secrets as env vars or files.
- Use liveness and readiness probes hitting the metrics endpoint if available.

## Scaling & Sharding
- Scale vertically by allocating more CPU/RAM.
- For large guild counts, enable sharding via `SHARD_COUNT`.
