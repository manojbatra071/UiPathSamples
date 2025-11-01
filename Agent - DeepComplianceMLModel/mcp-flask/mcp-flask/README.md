# MCP Flask Middleware (Docker-ready)

This repo contains a small Flask microservice (MCP) that forwards inference requests to a Hugging Face Inference Endpoint. It's Docker-ready and intended to act as an integration layer for DeepComplianceAgent.

## Quick start (local, dev)

1. Copy `.env.example` to `.env` and fill `HF_ENDPOINT_URL` and `HF_API_KEY`.
2. Build & run with docker-compose:

```bash
chmod +x entrypoint.sh
docker-compose up --build
```

3. Health: `GET http://localhost:8080/health`
4. Inference: `POST http://localhost:8080/infer` with JSON body (same payload you would send to HF).

## Deploying to production
- Use container registry (Docker Hub, ECR) and run on your cloud provider (ECS, ECS/Fargate, VM, k8s).
- Use a secrets manager rather than `.env` for HF API keys.
- Add TLS termination/load balancer in front of the MCP service.

## Environment variables
See `.env.example`.

## Security & ops notes
- Do not commit `.env` to source control.
- Limit outgoing concurrency to avoid HF overage costs.
- Add authentication to `/infer` (API key, mTLS) if used in production.
