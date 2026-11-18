#!/bin/bash
# usage: scrape_metrics.sh <outfile>
curl -s -m 10 http://127.0.0.1:8000/metrics > "$1"
echo "scraped $(wc -c < "$1") bytes -> $1"
