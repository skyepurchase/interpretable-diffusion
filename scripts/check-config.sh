#!/bin/bash

set -euo pipefail

if git diff --cached --name-only | grep -qxE ".*configs/.*\.yaml" &&
    git ls-files | grep -qxE ".*configs/.*\.yaml"; then
    exit 1
fi
exit 0
