FROM mcr.microsoft.com/vscode/devcontainers/cpp:ubuntu-22.04
 
# Install additional system packages.
RUN apt-get update && \
     export DEBIAN_FRONTEND=noninteractive && \
     apt-get -y install --no-install-recommends build-essential gcc cmake libtiff5-dev cmake
