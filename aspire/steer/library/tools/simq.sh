#!/usr/bin/env bash
# Run simulator jobs from a queue file, at most $P (default 2) at a time; each line: <program> <suite> <task> <episodes> [extra args]
# Usage: library/tools/simq.sh queue.txt [P]   -> appends one result line per job to outputs/simq.log
cd "$(dirname "$0")/../.."
P=${2:-2}
grep -v '^#' "$1" | grep -v '^\s*$' | xargs -P "$P" -L 1 bash -c '
  r=$(.venv/bin/python scripts/run_program.py --program "$0" --suite "$1" --task "$2" --episodes "$3" "${@:4}" 2>&1 | tail -1)
  echo "$(date +%H:%M:%S) $r" >> outputs/simq.log; echo "$r"'
