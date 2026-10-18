#!/bin/bash
# guard.sh LIMIT_GB CMD...: run CMD and kill its whole process tree when the
# summed RSS of the tree passes LIMIT_GB. Exit status: CMD's, or 99 when killed.
#
# Use it for anything heavy (complete SNAP graphs, parallel checks, A/B timing
# loops), one at a time. An unguarded leak once took the machine past 64 GB and
# rebooted it; another run filled it to 28 GB next to the user's experiments.
set -u
limit_gb=$1; limit_kb=$(( limit_gb * 1024 * 1024 )); shift
"$@" & root=$!

tree() {  # the pids of the tree under $1, $1 included
  local pids=$1 next=$1
  while [ -n "$next" ]; do
    next=$(ps -A -o pid=,ppid= | awk -v p=" $next " 'index(p, " " $2 " ") {print $1}' | tr '\n' ' ')
    next=${next% }
    [ -n "$next" ] && pids="$pids $next"
  done
  echo "$pids"
}

while kill -0 $root 2>/dev/null; do
  pids=$(tree $root)
  rss=$(ps -o rss= -p ${pids// /,} 2>/dev/null | awk '{s += $1} END {print s + 0}')
  if [ "$rss" -gt "$limit_kb" ]; then
    echo "guard: tree RSS ${rss} KB > ${limit_gb} GB, killing $pids" >&2
    kill -9 $pids 2>/dev/null
    wait $root 2>/dev/null
    exit 99
  fi
  sleep 0.5
done
wait $root
