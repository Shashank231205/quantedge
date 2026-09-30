#!/usr/bin/env bash
# Failover drill: prove the HA stack survives losing a machine.
#
#   1. API: drive steady traffic through the load balancer, stop one replica
#      gracefully, then hard-kill the other, and count failed requests.
#   2. Scheduler: find the leader, hard-kill it, and time how long until the
#      standby reports leadership.
#
# Exits non-zero if any request fails or the standby does not take over within
# the deadline. Runs in CI on every push; locally:  bash ops/failover_test.sh

set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose --profile ha"
LB=http://localhost:8080
KEY=${API_KEY:-quantedge-dev-key}
TAKEOVER_DEADLINE=30
LOG=$(mktemp)

step() { echo; echo "==> $*"; }
fail() { echo "FAIL: $*" >&2; $COMPOSE logs --tail 50 >&2 || true; exit 1; }

# wait_for <description> <seconds> <predicate> [args...]
# The predicate is re-run every second, so it must be a command, not a value.
wait_for() {
    local what=$1 deadline=$2; shift 2
    for _ in $(seq "$deadline"); do
        if "$@" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    fail "timed out waiting for $what"
}

lb_ready() { curl -fsS "$LB/health/ready"; }

backends_are() {
    local n
    n=$(curl -fsS http://localhost:8404/metrics \
        | awk '/^haproxy_backend_active_servers\{proxy="api"/ {print $2}')
    [[ "$n" == "$1" ]]
}

leader_of() {  # 1 if the scheduler container holds leadership, else 0
    docker exec "$1" curl -fsS http://localhost:9101/metrics 2>/dev/null \
        | awk '/^quantedge_scheduler_is_leader / {print int($2)}' || echo 0
}

is_leader() { [[ "$(leader_of "$1")" == "1" ]]; }

one_leader() {
    local n=0 c
    for c in "$@"; do n=$((n + $(leader_of "$c"))); done
    (( n == 1 ))
}

load() {  # load <seconds>: one request every 50ms, status codes to $LOG
    local end=$((SECONDS + $1))
    while (( SECONDS < end )); do
        curl -s -o /dev/null -w '%{http_code}\n' --max-time 10 \
            -H "X-API-Key: $KEY" "$LB/v1/system/info" >>"$LOG" || echo 000 >>"$LOG"
        sleep 0.05
    done
}

under_load() {  # under_load <docker verb> <container>: returns "failed total"
    : >"$LOG"
    load 20 &
    local pid=$!
    sleep 5
    docker "$1" "$2" >/dev/null
    wait "$pid"
    echo "$(grep -vc '^200$' "$LOG" || true) $(wc -l <"$LOG")"
}

step "Starting HA stack"
$COMPOSE up -d --build
wait_for "load balancer" 180 lb_ready
wait_for "two healthy replicas" 60 backends_are 2

mapfile -t REPLICAS < <($COMPOSE ps -q api)
(( ${#REPLICAS[@]} == 2 )) || fail "expected 2 api replicas, found ${#REPLICAS[@]}"

step "API: graceful stop of one replica under load"
read -r GRACEFUL_FAILED GRACEFUL_TOTAL < <(under_load stop "${REPLICAS[0]}")
echo "requests=$GRACEFUL_TOTAL failed=$GRACEFUL_FAILED"
docker start "${REPLICAS[0]}" >/dev/null
wait_for "replica to rejoin" 60 backends_are 2

step "API: hard kill of one replica under load"
read -r KILL_FAILED KILL_TOTAL < <(under_load kill "${REPLICAS[1]}")
echo "requests=$KILL_TOTAL failed=$KILL_FAILED"
docker start "${REPLICAS[1]}" >/dev/null

step "Scheduler: hard kill of the leader"
mapfile -t SCHEDULERS < <($COMPOSE ps -q scheduler)
(( ${#SCHEDULERS[@]} == 2 )) || fail "expected 2 schedulers, found ${#SCHEDULERS[@]}"
wait_for "exactly one scheduler leader" 60 one_leader "${SCHEDULERS[@]}"

if is_leader "${SCHEDULERS[0]}"; then
    LEADER=${SCHEDULERS[0]}; STANDBY=${SCHEDULERS[1]}
else
    LEADER=${SCHEDULERS[1]}; STANDBY=${SCHEDULERS[0]}
fi

START=$SECONDS
docker kill "$LEADER" >/dev/null
wait_for "standby takeover" "$TAKEOVER_DEADLINE" is_leader "$STANDBY"
TAKEOVER=$((SECONDS - START))
docker start "$LEADER" >/dev/null

step "Summary"
printf '  %-32s %s\n' \
    "graceful stop: failed / total" "$GRACEFUL_FAILED / $GRACEFUL_TOTAL" \
    "hard kill:     failed / total" "$KILL_FAILED / $KILL_TOTAL" \
    "scheduler takeover" "${TAKEOVER}s (deadline ${TAKEOVER_DEADLINE}s)"

(( GRACEFUL_FAILED == 0 )) || fail "$GRACEFUL_FAILED requests failed during a graceful stop"
(( KILL_FAILED == 0 )) || fail "$KILL_FAILED requests failed during a hard kill"
echo "PASS"
