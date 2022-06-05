#!/usr/bin/env bash

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run \
    --rm \
    -it \
    -u root \
    --gpus all \
    -v "$(pwd)":/workspaces/keras-training-kit \
    -w /workspaces/keras-training-kit \
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GID=$(id -g)" \
    -e "PYTHONPATH=/workspaces/keras-training-kit" \
    keras-training-kit:devel \
    bash /workspaces/keras-training-kit/docker/_docker_init.sh

