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

Observation end vs task/pending endpoint (machine-readable)
- observation_start_ns/observation_end_ns: delay+duration (~10-130 s); end_kind=observation_end_only.
- attribution_window: window_basis=task_or_pending_end; task_or_pending_end_ns + endpoint_reason
  (unknown|timeout|pending_end|actual_task_end); fail_closed_if_missing=true.
- duration arrival is NOT flush and NOT task termination; observation_end cannot substitute for task end;
  unknown endpoint => capability_unproven.

Fresh layout (all paths planned, none created)
- cell: /workspace/yangchen/work/grouped-deepdive-cell-h11-sprofile-v1
- run / msprof-out / probe-temp / probe-out / logs / archive / restore under that cell.
- runid/window/collect begin-end identity to be created at execution time by the existing tools.

Budgets and gates (machine-readable in plan-skeleton.json)
- stop 120 s, child 240 s, comm 90 s, restore 900 s; supervisor settle default 30 s — explicitly noted
  as NOT proving coverage of the old exit delay (attempt2 residual behaviour).
- settle decision is 300 s: settle_budget_s records requested 300, supported_by_pinned_supervisor=false,
  requires_new_frozen_supervisor=true, new supervisor SHA pending (27 minimal CLI; default stays 30;
  two-empty/kill/single-restore unchanged). d2576c63 is now a historical baseline, not bytes-frozen.
  300 s is a conservative bound only; old exit delay remains unproven. This blocks the final package until pinned.
- wall_clock_budget_s: preflight 60 + collect begin 60 + stop 120 + child 240 + settle 300 + restore 900 +
  collect end 120 = 1800 s, plus 300 s explicit margin => outer deadline 2100 s; collector observation
  10+120 s and app arm 240 s/comm 90 s are inside child and not added again. Lifecycle evidence must keep kill time,
  per-round same-snapshot state/pid/starttime, first-empty/two-empty times and deadline_hit.
- pre_stop_gates (ordered): caller_env -> preflight_ALL_PASS -> collect_begin_rc0; stop_after collect_begin_rc0.
  preflight/collect/wrapper new-cell copies are pending (constructor paths), with source SHAs recorded
  (3ea4e890 / 2bedc64a / 0af37af4); their derived-copy SHAs are produced in the final merge after 35 fix/32 review.
- continuous two-empty residual before restore; exactly one begin_restore; no second restore tree; no same-cell retry;
  collector wait/unflushed/rank-exits-first remains fail-closed.
- collect end/archive/SUMS paths are encoded; archive SUMS at <archive>/SHA256SUMS; end may repeat to collect late flush.
- Success requires: valid AIC sample + kernel identity + sample time within this submit-end window +
  this run's actual stream/task; otherwise capability_unproven. timeline-only/empty/truncated are not success.
- Collector help flags/units: aic-freq 100 Hz; delay/duration seconds; no documented conflicts;
  duration end semantics unproven.

Files
- child-argv.json: exact child argv list (msprof + native app).
- build_window_argv.py: canonical generator for the supervisor argv (reads child-argv.json only;
  optional --check-argv asserts exact equality, preventing duplicate/drifting embedded child argv).
  $PENDING_WINDOW_PATH and $PENDING_FRESH_IDENTITY_FILE must be pinned at execution time.
- plan-skeleton.json: plan fields, budgets, gates, pending list, status.
- identity-snapshot-now.txt: informational current read (not final).
- restore-source.txt: restore argv source and adapted template (not executed).
- pending.txt: unresolved pins (revised 35 files/integrity/Linux incremental, fresh identity, adapter pin).

No real msprof / torch / device / service stop / deployment occurred while preparing this skeleton.
Note: the 7fcba47 Linux 34/34 run was executed under 32 incremental A BEFORE the settle-300 decision
(evidence commit 6eca211); it is an intermediate result, superseded, and NOT readiness. No further runs of
that intermediate source are permitted; wait for the integrated 27+35 fixed source.
