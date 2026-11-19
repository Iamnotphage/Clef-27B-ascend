H11 sample-based capability experiment — auditable execution plan skeleton (PREPARATION ONLY)
==============================================================================================
Status: prepared_pending_source_pin; device_go=false; not deployed; nothing executed.

Scope
- One H11 sample-based run: msprof sample-based 100 Hz, delay 10 s, duration 120 s, task-time l1,
  ai-core on, aic-metrics PipeUtilization, followed by the original H11 native app argv
  (torchrun 2 ranks + run_probe H11 M128 K256 N512 seed 20261119, BF16, flags=8 KEEP_ALL).
- Child argv is a single list handed to the audited supervisor's existing --child-argv interface;
  no shell, no argument splitting, no --application, no --dynamic, no --pid, no --instr.
- The msprof collector is part of the child argv (single process tree). Native run flags/semantics
  are unchanged from the frozen H11 app.
- Adapter is the OLD FROZEN probe_adapter.so d181b15a... (verified read-only); no new .so build or switch;
  release candidate 83598d6e is NOT used. Supervisor is 8f4b843 existing bytes d2576c63... with only
  its path placed in a fresh isolated dir; no derived core changes.

Observation end vs task/pending endpoint
- observation window: ~10 s delay + 120 s duration = about 130 s after launch (EARLY pending only).
- task/pending endpoint: actual task completion, timeout, or outer 240 s termination if artifacts exist.
- attribution uses the explicit task endpoint and reason; duration arrival is NOT flush and NOT task
  termination; if the endpoint is unknown, the result is capability_unproven.

Fresh layout (all paths planned, none created)
- cell: /workspace/yangchen/work/grouped-deepdive-cell-h11-sprofile-v1
- run / msprof-out / probe-temp / probe-out / logs / archive / restore under that cell.
- runid/window/collect begin-end identity to be created at execution time by the existing tools.

Budgets and gates
- stop 120 s, child 240 s, comm 90 s, restore 900 s; supervisor settle default 30 s — explicitly noted
  as NOT proving coverage of the old exit delay (attempt2 residual behaviour).
- Fail-closed gates: caller env + preflight ALL PASS + collect begin success before any stop;
  continuous two-empty residual before restore; exactly one begin_restore; no second restore tree;
  no same-cell retry. Collector wait/unflushed/rank-exits-first remains fail-closed.
- Success requires: valid AIC sample + kernel identity + sample time within this submit-end window +
  this run's actual stream/task; otherwise capability_unproven. timeline-only/empty/truncated are not success.
- Collector help flags/units: aic-freq 100 Hz; delay/duration seconds; no documented conflicts;
  duration end semantics unproven.

Files
- child-argv.json: exact child argv list (msprof + native app).
- window-argv-skeleton.json: full supervisor argv; $PENDING_WINDOW_PATH and $PENDING_FRESH_IDENTITY_FILE
  must be pinned at execution time.
- plan-skeleton.json: plan fields, budgets, gates, pending list, status.
- identity-snapshot-now.txt: informational current read (not final).
- restore-source.txt: restore argv source and adapted template (not executed).
- pending.txt: unresolved pins (revised 35 files/integrity/Linux incremental, fresh identity, adapter pin).

No real msprof / torch / device / service stop / deployment occurred while preparing this skeleton.
