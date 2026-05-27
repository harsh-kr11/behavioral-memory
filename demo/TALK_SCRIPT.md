# Demo Talk Script — Behavioral Memory for Tool Orchestration

**Run command:** `uv run python demo/showcase.py`

**Total runtime:** ~30 seconds of compute, rest is your narration.

---

## OPENING (before running anything)

> We're looking at a common problem with AI agents that use tools. Whether it's database queries, API calls, sending notifications — agents that orchestrate multiple tools tend to make the same mistakes over and over. Every time a new task comes in, the agent starts from zero. It sees a list of available tools and their parameters, and it guesses the best plan.
>
> The problem is that tool schemas tell you *what* a tool accepts, but not *how* your organization uses it. Which SQL column is revenue? Does "completed order" mean status equals 'completed', or does it mean shipped and delivered? Do alerts go to email or Slack? These are domain conventions — institutional knowledge that lives in people's heads.
>
> What we built is **behavioral memory** — a vector store of validated execution traces. Every time an agent successfully completes a task, we store the task description paired with the exact tool chain that worked. When a new task comes in, we embed the query, search for semantically similar past successes, and inject them into the prompt as reference examples. The LLM follows proven patterns instead of guessing.
>
> Let me show you what this looks like.

*Run the demo:* `uv run python demo/showcase.py`

The initialization takes about 1-2 seconds. It loads a Gemini model, initializes the embedding model, registers 7 MCP tool schemas, and seeds the memory with 12 validated traces.

---

## ACT 1 — Memory Inspector

*The table of 12 traces appears on screen.*

> This is the behavioral memory store. These 12 traces are the agent's institutional knowledge. Each row is a validated execution trace — a task description paired with the ordered sequence of tool calls that accomplished it.
>
> Look at trace number 1: "Get Q1 revenue data and send a report to stakeholders." The tool chain is `query_database` then `generate_report` then `send_notification`. That's three steps. And the source column says "seed" — these traces were validated by domain experts and seeded into memory as ground truth.
>
> Now look at the embedding neighborhood below the table. We took the first trace and searched the vector store for its nearest neighbors. The most similar trace is "Monthly valid orders trend and email the report" at 0.747 cosine similarity. These are the kinds of traces that would get retrieved when a similar query comes in.
>
> What matters here is that each trace encodes domain conventions that are not in the tool schemas. For example, revenue is computed from `order_items` using `quantity * unit_price` — not from the `total_amount` column in the orders table. A "completed" order means status is 'shipped' or 'delivered', not a literal status called 'completed'. Alerts go to Slack channel `#data-alerts`, not email. Reports use `markdown_table` format. Dashboard data gets stored to cache, not to a database table.
>
> None of this is discoverable from the tool definitions alone. The agent has to learn it from examples.

*Press Enter to continue.*

---

## ACT 2 — Side-by-Side Strategy Comparison

*The Act 2 panel appears with the query.*

> Now we run the core experiment. We take a single query — "Calculate average basket size per customer segment and alert the data team" — and run it through three strategies.
>
> The first strategy is **zero-shot**: the model sees only the query and the tool schemas. No examples, no memory. This is how most agents work today.

*Zero-shot results appear with pipeline timing.*

> Notice the pipeline logging on the left — we can see exactly what's happening. The prompt is about 1,050 tokens with zero reference examples. The model takes a few seconds and produces a plan.
>
> Look at what zero-shot chose: `query_database`, then `generate_report`, then `send_notification` via email. It treated this as a reporting task.

> The second strategy is **static few-shot**: the model gets the same three fixed examples for every query, regardless of what you ask. This is the traditional approach to few-shot prompting.

*Static few-shot results appear.*

> Similar result — `query_database`, `generate_report`, `send_notification` via email. The fixed examples didn't help here because they weren't relevant to basket sizes.

> Now the third strategy — **dynamic retrieval** — our approach. Watch the pipeline.

*Dynamic retrieval results scroll with timing.*

> This is where behavioral memory kicks in. The system embeds the query, searches the vector store, and finds 6 candidate traces. The top match at 0.81 similarity is "Get basket sizes per order and alert about unusual patterns" — that's a direct hit. It also retrieves the fulfillment metrics trace and the pipeline trace.
>
> Three traces are selected within the token budget — 1,709 out of 3,500 allowed tokens. These get injected into the prompt as reference examples.
>
> Now look at the plan: `query_database`, then `transform_data`, then `send_notification` via **Slack to `#data-team`**.

*Point at the comparison table.*

> The comparison table makes it clear. Zero-shot and static both chose `generate_report` at step 2. Dynamic chose `transform_data`. Why? Because the retrieved trace for basket sizes taught the model that basket size means item count — a quantity aggregation, not a report. And the alert convention says use Slack, not email.

*Point at the structural diff.*

> The diff confirms it: `generate_report` was removed, `transform_data` was added. Same input query, materially different plan — because the agent had access to relevant past experience.
>
> This is the thesis of the paper: when you retrieve semantically similar validated traces and inject them into the prompt, the LLM follows proven domain conventions instead of guessing.

*Press Enter to continue.*

---

## ACT 3 — Gatekeeper Challenge

*The Act 3 panel appears.*

> The obvious next question is: what stops bad data from entering memory? If someone stores a trace with wrong conventions, doesn't the whole system degrade?
>
> That's what the gatekeeper pipeline addresses. Before any trace enters behavioral memory, it passes through three validation gates.

*Candidate #1 appears — broken dependency.*

> Candidate 1 is a trace for "Get product data and generate report." The tool chain is just `generate_report` — but look at the parameters: it references `source_step: s0_nonexistent`. That step doesn't exist in the trace.
>
> Gate 1, Schema Validation, checks that all tools exist and required parameters are present. This passes — `generate_report` is a real tool with valid params.
>
> Gate 2, Sandbox Execution, does a dry-run data-flow check. It walks through the steps and verifies that every `source_step` reference points to a step that has already produced output. Step `s1` references `s0_nonexistent` which doesn't exist — **Gate 2 fails**. The trace is rejected. It never enters memory.

*Candidate #2 appears — wrong convention.*

> Candidate 2 is more subtle. "Get quarterly revenue and send report." The tool chain is `query_database` then `generate_report`. Structurally, this is perfectly valid — real tools, valid parameters, correct data flow.
>
> Gate 1 passes. Gate 2 passes. Gate 3, Semantic Dedup, checks if this trace is too similar to an existing one. The nearest trace in memory has a similarity score of 0.843, which is below the 0.95 threshold — so it's not a duplicate. **All three gates pass. The trace is admitted.**
>
> But here's the thing: this trace uses `total_amount` for revenue instead of `quantity * unit_price`, and it uses CSV format instead of `markdown_table`. It's structurally valid but semantically wrong.

*The Limitation panel appears.*

> We're transparent about this. The gatekeeper validates **structure** — does the plan make sense as a sequence of tool calls? It does not validate **domain semantics** — is this the right business logic? That's a fundamentally harder problem.
>
> This is why the seed traces matter. They establish the correct conventions in memory. The gatekeeper prevents broken plans from entering. The seed traces ensure the conventions are right from the start. And in production, the feedback loop — where human-approved executions flow back through the gatekeeper into memory — continuously reinforces correct patterns.

*Press Enter to continue.*

---

## ACT 4 — Custom Query (if time permits)

*The REPL prompt appears.*

> Now I can take any query. Type a task description, and we'll run it through all three strategies in real time. You'll see the retrieval scores, the pipeline timing, and the plan comparison.

**Good queries to demonstrate live:**

- `Build a daily revenue pipeline and store results for the dashboard`
  - *Shows: zero-shot just wraps everything in schedule_task; dynamic retrieves the pipeline pattern and builds query -> transform -> store with cache target*

- `Schedule a weekly fulfillment rate report for the ops team`
  - *Shows: zero-shot collapses everything into a single schedule_task call; dynamic retrieves the fulfillment metrics trace and builds the full workflow first, then wraps it in schedule_task*

- `Get net order values excluding discounts and store for the dashboard`
  - *Shows: zero-shot may skip the discount subtraction; dynamic retrieves the net order value trace and applies total_amount - discount*

- `Archive all valid orders from last quarter as CSV`
  - *Shows: zero-shot may only exclude cancelled; dynamic retrieves the valid orders trace and excludes both cancelled AND returned, uses append mode*

*Type `quit` when done.*

---

## CLOSING (after demo ends)

> To summarize what we just saw:
>
> First, behavioral memory gives agents institutional knowledge. Instead of guessing which tools to use and how to use them, the agent retrieves validated patterns from past successes.
>
> Second, semantic retrieval means the right examples surface for the right query. A question about basket sizes retrieves the basket size trace, not a random revenue trace. This is the key advantage over static few-shot prompting.
>
> Third, the gatekeeper pipeline maintains memory quality. Structurally broken traces are caught and rejected before they can contaminate retrieval results.
>
> And fourth, the entire system is model-agnostic and framework-agnostic. The behavioral memory layer sits between your agent and the LLM. It works with any LangChain-compatible model and any MCP tool ecosystem. You bring your tools, you bring your model — behavioral memory provides the institutional knowledge layer.
>
> The benchmark results across 30 evaluation tasks show that dynamic retrieval improves plan correctness by 15-25 percentage points over zero-shot, with statistically significant results confirmed by McNemar's test. The full results are in the paper, Section IV.

---

## Quick Reference — What to Point At

| Moment | What to highlight |
|--------|------------------|
| Act 1 table | "Each row is a validated trace — task + tool chain" |
| Act 1 neighborhood | "Vector similarity — these traces are nearby in embedding space" |
| Act 2 pipeline log | "Real-time: embed query, retrieve traces, assemble prompt, call model" |
| Act 2 similarity bars | "0.81 similarity — the system found the right basket size trace" |
| Act 2 comparison table | "Same query, different plan — memory changed the tool selection" |
| Act 2 diff | "Red = removed, Green = added — structural difference in one glance" |
| Act 3 candidate #1 | "Broken data flow — Gate 2 catches it, trace rejected" |
| Act 3 candidate #2 | "Structurally valid but semantically wrong — passes all gates" |
| Act 3 limitation panel | "We're honest about what the gatekeeper does and doesn't catch" |

---

## Troubleshooting During the Talk

| Problem | Fix |
|---------|-----|
| Slow LLM response | "The model is thinking — you can see the pipeline stages updating in real time" |
| Plans look identical across strategies | Run with a different query (basket size, pipeline, scheduling queries show the biggest differences) |
| Act 4 query concatenation | Type one query at a time, press Enter, wait for results before typing the next |
| API key error | Make sure `.env` has `GOOGLE_API_KEY=your-key` and restart the script |
