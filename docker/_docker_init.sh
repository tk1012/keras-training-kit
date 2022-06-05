#!/usr/bin/env bash

# overwrite uid and gid
usermod -u "$HOST_UID" developer
groupmod -g "$HOST_GID" developer

# keep some environments
{
    echo "export PYTHONPATH=${PYTHONPATH}"
    echo "export PYTHONIOENCODING=utf-8"
    echo "export PATH=${PATH}"
    echo "cd /workspaces/keras-training-kit"
} >> /home/developer/.bashrc

# change to the developer
chown developer:developer -R /home/developer
su - developer
