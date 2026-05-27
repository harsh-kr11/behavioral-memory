# Demo Setup & Run Guide

## Quick Start

```bash
git pull origin main
uv sync --extra agent --extra eval
uv run python demo/showcase.py
```

The demo initializes in ~2 seconds, then walks through 4 acts. Press Enter between acts to advance.

## Run Options

```bash
# Full demo (all 4 acts with pauses between them)
uv run python demo/showcase.py

# Skip pauses (continuous output)
uv run python demo/showcase.py --no-pause

# Run specific acts only
uv run python demo/showcase.py --act 2              # strategy comparison only
uv run python demo/showcase.py --act 1 --act 3      # memory + gatekeeper (instant, no LLM)
uv run python demo/showcase.py --act 2 --act 3      # comparison + gatekeeper

# Custom query for Act 2
uv run python demo/showcase.py --act 2 --query "Build a daily revenue pipeline and store for the dashboard"

# Different model
uv run python demo/showcase.py --model gemini-2.5-pro

# Via Makefile
make showcase
```

## What Each Act Does

| Act | Name | LLM Calls | Time | What It Shows |
|-----|------|-----------|------|---------------|
| 1 | Memory Inspector | 0 | Instant | All 12 seed traces in the store + embedding neighborhood |
| 2 | Strategy Comparison | 3 | ~15-20s | Same query through zero-shot / static / dynamic with diff |
| 3 | Gatekeeper Challenge | 0 | Instant | Poisoned traces fed through 3 validation gates |
| 4 | Custom Query REPL | 3 per query | ~10s each | Interactive — type any query, type `quit` to exit |

## Sample Queries for Act 4

These are designed to show clear differences between zero-shot and dynamic retrieval. Each one triggers specific domain conventions that the seed traces teach.

### Best demo queries (show largest differences)

```
Build a daily revenue pipeline and store results for the dashboard
```
Zero-shot wraps everything in a single `schedule_task`. Dynamic builds the full `query_database -> store_results` pipeline with cache target.

```
Schedule a weekly fulfillment rate report for the ops team
```
Zero-shot produces a single `schedule_task`. Dynamic builds `query_database -> generate_report -> schedule_task` with the correct fulfillment formula.

```
Get net order values excluding discounts and store for the dashboard
```
Dynamic retrieves the net order value trace and applies `total_amount - discount` with `transform_data`, stores to cache.

```
Archive all valid orders from last quarter as CSV
```
Zero-shot may only exclude cancelled. Dynamic excludes both cancelled AND returned, uses append mode.

### Revenue queries (tests quantity * unit_price convention)

```
Get monthly revenue breakdown and send to the finance team
```

```
Calculate total revenue by product category and archive the results
```

### Alert queries (tests Slack #data-alerts convention)

```
Monitor order cancellation rates and alert if above threshold
```

```
Check fulfillment metrics and notify the ops team
```

### Pipeline queries (tests query -> transform -> store pattern)

```
Build a customer lifetime value pipeline and cache for the dashboard
```

```
Create a product performance ranking and store for analytics
```

### Report queries (tests markdown_table + email convention)

```
Generate a weekly summary of returned orders and email to management
```

```
Find top 10 customers by spending and create a report
```

### Scheduling queries (tests daily + notify_on_failure convention)

```
Set up automated daily tracking of new customer signups
```

```
Schedule a daily report of order volumes by region
```

## Tips

- Type **one query at a time** in Act 4 and press Enter. Do not paste multiple lines at once.
- Type `quit`, `exit`, or `q` to end Act 4.
- If a query produces identical plans across strategies, try one of the pipeline or scheduling queries above — those show the biggest differences.
- The talk narration script is at `demo/TALK_SCRIPT.md`.
