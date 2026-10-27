#!/usr/bin/env bash
# Deploys the fraud-detection API to Azure Container Apps, with an Azure
# Database for PostgreSQL Flexible Server for prediction logging.
#
# Prerequisites:
#   - Azure CLI, logged in (az login)
#   - a trained model exported to model/ (python -m src.export_model)
#   - PG_PASSWORD set (8+ chars, upper + lower case + digit)
#
# Usage (from the repository root; on Windows run it from Git Bash):
#   PG_PASSWORD='...' ./deploy/azure/deploy.sh
#   WITH_POSTGRES=0 ./deploy/azure/deploy.sh   # API only, no prediction logging
# Re-running it rebuilds the image and updates the running app.
# Remove everything with ./deploy/azure/teardown.sh
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-fraud-detection-rg}"
LOCATION="${LOCATION:-polandcentral}"
APP_NAME="${APP_NAME:-fraud-detection-api}"
ENV_NAME="${ENV_NAME:-fraud-detection-env}"
WITH_POSTGRES="${WITH_POSTGRES:-1}"
PG_ADMIN="${PG_ADMIN:-fraudadmin}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

# Registry and Postgres server names must be globally unique: derive them from the subscription id.
SUB_SUFFIX="$(az account show --query id -o tsv | tr -d '\r-' | cut -c1-8)"
ACR_NAME="${ACR_NAME:-fraudapi${SUB_SUFFIX}}"
PG_SERVER="${PG_SERVER:-fraud-pg-${SUB_SUFFIX}}"
IMAGE="${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG}"

if [[ ! -f model/MLmodel ]]; then
  echo "model/ not found: run 'python -m src.export_model' after training first." >&2
  exit 1
fi
if [[ "$WITH_POSTGRES" == "1" && -z "${PG_PASSWORD:-}" ]]; then
  echo "Set PG_PASSWORD (or WITH_POSTGRES=0 to deploy without a database)." >&2
  exit 1
fi

echo "==> Registering providers and the containerapp CLI extension"
az extension add --name containerapp --upgrade --only-show-errors
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.DBforPostgreSQL; do
  az provider register --namespace "$ns" --wait
done

echo "==> Resource group $RESOURCE_GROUP ($LOCATION)"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "==> Container registry $ACR_NAME, building $IMAGE in the cloud"
if ! az acr show --name "$ACR_NAME" --output none 2>/dev/null; then
  az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --output none
fi
az acr build --registry "$ACR_NAME" --image "${APP_NAME}:${IMAGE_TAG}" --file deploy/azure/Dockerfile .

DATABASE_URL=""
if [[ "$WITH_POSTGRES" == "1" ]]; then
  echo "==> PostgreSQL Flexible Server $PG_SERVER (Burstable B1ms)"
  if ! az postgres flexible-server show --resource-group "$RESOURCE_GROUP" --name "$PG_SERVER" --output none 2>/dev/null; then
    # --public-access 0.0.0.0 allows connections from Azure services only (no public IPs).
    az postgres flexible-server create \
      --resource-group "$RESOURCE_GROUP" --name "$PG_SERVER" --location "$LOCATION" \
      --tier Burstable --sku-name Standard_B1ms --storage-size 32 --version 16 \
      --admin-user "$PG_ADMIN" --admin-password "$PG_PASSWORD" \
      --public-access 0.0.0.0 --yes --output none
    az postgres flexible-server db create \
      --resource-group "$RESOURCE_GROUP" --server-name "$PG_SERVER" --database-name fraud_detection --output none
  fi
  # Windows (Git Bash) venvs ship python.exe but no python3.exe, so try both.
  for PY in python3 python; do "$PY" -c 'pass' 2>/dev/null && break; done
  ENCODED_PW="$("$PY" -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$PG_PASSWORD")"
  DATABASE_URL="postgresql+psycopg2://${PG_ADMIN}:${ENCODED_PW}@${PG_SERVER}.postgres.database.azure.com:5432/fraud_detection?sslmode=require"
fi

echo "==> Container Apps environment $ENV_NAME"
if ! az containerapp env show --resource-group "$RESOURCE_GROUP" --name "$ENV_NAME" --output none 2>/dev/null; then
  az containerapp env create --resource-group "$RESOURCE_GROUP" --name "$ENV_NAME" --location "$LOCATION" --output none
fi

echo "==> Container app $APP_NAME"
if az containerapp show --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --output none 2>/dev/null; then
  if [[ -n "$DATABASE_URL" ]]; then
    az containerapp secret set --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" \
      --secrets "database-url=${DATABASE_URL}" --output none
  fi
  az containerapp update --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --image "$IMAGE" --output none
else
  # ${arr[@]+...} below keeps an empty array safe under set -u on macOS bash 3.2.
  DB_ARGS=()
  if [[ -n "$DATABASE_URL" ]]; then
    DB_ARGS=(--secrets "database-url=${DATABASE_URL}" --env-vars "DATABASE_URL=secretref:database-url")
  fi
  # Scales to zero when idle, so the app costs nothing between demo requests.
  az containerapp create \
    --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --environment "$ENV_NAME" \
    --image "$IMAGE" --registry-server "${ACR_NAME}.azurecr.io" --registry-identity system \
    --target-port 8000 --ingress external \
    --min-replicas 0 --max-replicas 1 --cpu 0.5 --memory 1.0Gi \
    ${DB_ARGS[@]+"${DB_ARGS[@]}"} --output none
fi

FQDN="$(az containerapp show --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" \
  --query properties.configuration.ingress.fqdn -o tsv | tr -d '\r')"
echo
echo "Deployed: https://${FQDN}"
echo "  health: https://${FQDN}/health"
echo "  docs:   https://${FQDN}/docs"
