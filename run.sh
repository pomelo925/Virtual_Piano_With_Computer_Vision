#!/bin/bash

export DISPLAY=:0
xhost +local:root

usage() {
  echo "usage: $0 [service]"
  echo "service: dev | deploy"
  exit 1
}

# 參數檢查
if [ $# -ne 1 ]; then
  usage
fi

SERVICE=$1

case "$SERVICE" in
  dev|deploy)
    ;;
  *)
    echo "Invalid service: $SERVICE"
    usage
    ;;
esac

echo "[VIRTUAL-PIANO] Starting $SERVICE service ..."
docker compose -f docker/compose.yml run --rm $SERVICE bash
