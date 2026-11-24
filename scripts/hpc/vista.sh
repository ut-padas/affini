#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One shared SSH connection to TACC Vista, opened once with the password and token
# and then reused by every later ssh, scp, rsync or script, without a prompt.
#
# Usage (from the laptop, never on Vista):
#     scripts/hpc/vista.sh open        open the shared connection if it is not open (prompts once)
#     scripts/hpc/vista.sh             log in to a login node through it (opens it first if needed)
#     scripts/hpc/vista.sh run CMD...  run one command on a login node; fails, without a prompt, when
#                                      the connection is closed (for scripts and agents)
#     scripts/hpc/vista.sh check       exit 0 if the connection is open, 1 otherwise
#     scripts/hpc/vista.sh close       close it (every session riding on it ends too)
#
# The TACC user name is asked once and kept in scripts/hpc/vista.local beside this script, a file
# git ignores, so each person who clones the repository has their own; TACC_USER overrides it. Only
# the user name is kept: TACC asks for the password and a 6-digit token that changes every 30 s on
# every new connection, so a login cannot be automated, `run` refuses to prompt, and a person opens
# the connection with `open`. The host and the socket's path can be changed with VISTA_HOST and
# VISTA_SOCKET (one socket per user name by default). The connection lives in the background for
# VISTA_PERSIST (default 12h) after the last session on it ends; a laptop that sleeps or changes
# network drops it, and `open` is needed again.
# -----------------------------------------------------------------------------
set -euo pipefail

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vista.local"
if [[ -z "${TACC_USER:-}" && -f "$LOCAL" ]]; then
    TACC_USER="$(sed -n 's/^TACC_USER=//p' "$LOCAL" | head -n 1)"
fi
if [[ -z "${TACC_USER:-}" ]]; then
    if [[ ! -t 0 ]]; then
        echo "Vista: no TACC user name; run '$0 open' once in a terminal, or set TACC_USER" >&2
        exit 1
    fi
    read -r -p "TACC user name: " TACC_USER
    [[ -n "$TACC_USER" ]] || exit 1
    printf 'TACC_USER=%s\n' "$TACC_USER" > "$LOCAL"
    echo "Vista: saved in $LOCAL (ignored by git)"
fi
HOST="${VISTA_HOST:-vista.tacc.utexas.edu}"
SOCKET="${VISTA_SOCKET:-$HOME/.ssh/cm-vista-$TACC_USER}"
PERSIST="${VISTA_PERSIST:-12h}"
TARGET="$TACC_USER@$HOST"

OPTS=(-o ControlPath="$SOCKET")
# OpenSSH 10.1+ warns on every connection that Vista's key exchange is not post-quantum; the warning
# says nothing the user can act on here, so it is silenced where the client knows the option.
if ssh -G -o WarnWeakCrypto=no localhost >/dev/null 2>&1; then
    OPTS+=(-o WarnWeakCrypto=no)
fi

is_open() { ssh "${OPTS[@]}" -O check "$TARGET" >/dev/null 2>&1; }

open() {
    if is_open; then
        echo "Vista: the shared connection is already open ($SOCKET)"
        return
    fi
    mkdir -p "$(dirname "$SOCKET")" && chmod 700 "$(dirname "$SOCKET")"
    echo "Vista: opening the shared connection for $PERSIST (password, then the TACC token)"
    ssh "${OPTS[@]}" -o ControlMaster=yes -o ControlPersist="$PERSIST" -fN "$TARGET"
    is_open && echo "Vista: open; later ssh, scp and this script reuse it without a prompt"
}

case "${1:-login}" in
    open)  open ;;
    login) open; ssh "${OPTS[@]}" "$TARGET" ;;
    run)
        shift
        [[ $# -gt 0 ]] || { echo "usage: $0 run CMD..." >&2; exit 2; }
        is_open || { echo "Vista: the shared connection is closed; run '$0 open' first" >&2; exit 1; }
        ssh "${OPTS[@]}" -o BatchMode=yes "$TARGET" "$@"
        ;;
    check)
        if is_open; then echo "Vista: open ($SOCKET)"; else echo "Vista: closed"; exit 1; fi
        ;;
    close) ssh "${OPTS[@]}" -O exit "$TARGET" 2>/dev/null && echo "Vista: closed" || echo "Vista: was not open" ;;
    *) echo "usage: $0 [open|login|run CMD...|check|close]" >&2; exit 2 ;;
esac
