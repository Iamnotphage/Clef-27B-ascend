import json, sys
point = sys.argv[1]
try:
    report = json.load(open(point + "/benchmark.json"))
except Exception as exc:
    print("NO REPORT:", exc); sys.exit(0)
s = report.get("summary", {})
lat = s.get("successful_request_latency_ms") or {}
hbm = ((report.get("hbm") or {}).get("sampled_peak_hbm") or {})
out = {
  "label": point.split("/")[-1],
  "passed": report.get("passed"),
  "requested": s.get("requested_requests"),
  "completed": s.get("completed_requests"),
  "successful": s.get("successful_requests"),
  "failed": s.get("failed_requests"),
  "success_rate": s.get("success_rate"),
  "wall_seconds": s.get("measurement_wall_seconds"),
  "rps": s.get("successful_requests_per_second"),
  "decisions_per_second": s.get("decisions_per_second"),
  "input_tokens_per_second": s.get("input_tokens_per_second"),
  "p50_ms": lat.get("p50"), "p95_ms": lat.get("p95"), "p99_ms": lat.get("p99"),
  "min_duration_satisfied": s.get("minimum_measurement_duration_satisfied"),
  "peak_http_concurrency": s.get("observed_peak_http_concurrency"),
  "hbm_max_total_used_mib": hbm.get("max_total_used_mib"),
}
json.dump(out, open(point + "/point-summary.json", "w"), indent=1)
print(json.dumps(out))
