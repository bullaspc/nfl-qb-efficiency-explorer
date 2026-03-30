#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID="your-gcp-project-id"   # ← EDIT THIS
REGION="us-central1"
REPO="nfl-qb-explorer"
SERVICE="nfl-qb-explorer"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE"

# ── One-time setup (safe to re-run) ───────────────────────────────────────────
gcloud config set project "$PROJECT_ID"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="NFL QB Explorer images" 2>/dev/null || true

gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

# ── Build & deploy ─────────────────────────────────────────────────────────────
TAG=$(git rev-parse --short HEAD)

echo "Building image: $IMAGE:$TAG"
gcloud builds submit \
  --tag "$IMAGE:$TAG" \
  --project "$PROJECT_ID"

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE" \
  --image "$IMAGE:$TAG" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 600 \
  --min-instances 0 \
  --max-instances 3

echo ""
echo "✅  Deployed successfully!"
echo "URL: $(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')"
