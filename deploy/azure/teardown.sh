#!/usr/bin/env bash
# Deletes every Azure resource created by deploy.sh (they all live in one resource group).
set -euo pipefail
RESOURCE_GROUP="${RESOURCE_GROUP:-fraud-detection-rg}"
az group delete --name "$RESOURCE_GROUP" --yes --no-wait
echo "Deleting resource group $RESOURCE_GROUP in the background."
