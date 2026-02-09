#!/bin/bash

set -euo pipefail

if git diff --cached --name-only --diff-filter=MR | grep -qxE ".*configs/.*\.yaml"; then
    exit 1
fi
exit 0
