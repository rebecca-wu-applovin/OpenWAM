#!/usr/bin/env bash
# Shared helpers for the run_*.sh launchers. Source it: `source "$(dirname "$0")/base.sh"`.
#
# Run index: a global counter in /root/.core/index (override with CORE_INDEX_FILE) holding the
# last index handed out (plain integer). Each launch takes the next one and prefixes it, zero-padded to
# 4 digits (e.g. 0094), to the wandb run name.

CORE_INDEX_FILE="${CORE_INDEX_FILE:-/root/.core/index}"

# next_run_index: bump the counter and print the new value.
#   Input: none (reads/writes ${CORE_INDEX_FILE}; a missing file counts as 0).
#   Output: stdout, the new index zero-padded to 4 digits (e.g. 0094); the file holds it unpadded (94).
next_run_index() {
    mkdir -p "$(dirname "${CORE_INDEX_FILE}")"
    (
        # flock: two launches started at once must not get the same index.
        flock -x 9
        local cur
        cur="$(tr -dc '0-9' < "${CORE_INDEX_FILE}" 2>/dev/null || true)"
        # 10#: read as decimal even if the file ever holds a zero-padded value (bash treats 0094 as octal).
        local next=$(( 10#${cur:-0} + 1 ))
        echo "${next}" > "${CORE_INDEX_FILE}"
        printf '%04d\n' "${next}"
    ) 9>>"${CORE_INDEX_FILE}.lock"
}
