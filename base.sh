#!/usr/bin/env bash
# Shared helpers for the run_*.sh launchers. Source it: `source "$(dirname "$0")/base.sh"`.
#
# Run index: a global counter in /root/.core/index (override with CORE_INDEX_FILE) holding the
# last index handed out. Each launch takes the next one and prefixes it to the wandb run name.

CORE_INDEX_FILE="${CORE_INDEX_FILE:-/root/.core/index}"

# next_run_index: bump the counter and print the new value.
#   Input: none (reads/writes ${CORE_INDEX_FILE}; a missing file counts as 0).
#   Output: stdout, the new index (integer); the file then holds that same value.
next_run_index() {
    mkdir -p "$(dirname "${CORE_INDEX_FILE}")"
    (
        # flock: two launches started at once must not get the same index.
        flock -x 9
        local cur
        cur="$(tr -dc '0-9' < "${CORE_INDEX_FILE}" 2>/dev/null || true)"
        local next=$(( ${cur:-0} + 1 ))
        echo "${next}" > "${CORE_INDEX_FILE}"
        echo "${next}"
    ) 9>>"${CORE_INDEX_FILE}.lock"
}
