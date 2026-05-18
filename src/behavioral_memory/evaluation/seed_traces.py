"""12 seed traces that teach domain conventions to behavioral memory.

These encode orchestration conventions NOT obvious from tool schemas alone.
They are loaded into the vector store before the dynamic retrieval evaluation.
See the paper Section IV.A for design rationale.

Conventions taught:
  - "revenue" -> computed from order_items (quantity * unit_price), NOT orders.total_amount
  - "completed orders" -> status IN ('shipped', 'delivered')
  - "valid orders" -> exclude both 'cancelled' AND 'returned'
  - "report" -> generate_report with markdown_table + send via email
  - "alert" -> send_notification via slack to #data-alerts
  - "archive" -> store_results to csv_file with append mode
  - "dashboard data" -> store_results to cache (not database_table)
  - "pipeline" -> must include query -> transform -> store
  - "scheduled report" -> schedule_task with daily interval + notify_on_failure=true
  - "basket size" -> number of items (quantity), not dollar amount
  - "fulfillment rate" -> delivered / (total - cancelled)
  - "net order value" -> total_amount - discount
"""

from __future__ import annotations

from behavioral_memory.core.schemas import ExecutionTrace, ToolCall


def _build_trace(task: str, chain: list[dict], explanation: str) -> ExecutionTrace:
    """Helper to convert raw dict data into an ExecutionTrace."""
    tool_chain = [
        ToolCall(
            step_id=step["step_id"],
            tool_name=step["tool"],
            parameters=step["params"],
        )
        for step in chain
    ]
    return ExecutionTrace(
        task_description=task,
        tool_chain=tool_chain,
        validated=True,
        source="seed",
        metadata={"explanation": explanation},
    )


SEED_TRACES_RAW: list[dict] = [
    {
        "task": "Get Q1 revenue data and send a report to stakeholders",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN orders o ON o.order_id = oi.order_id WHERE o.status NOT IN ('cancelled', 'returned') AND o.order_date >= '2025-01-01' AND o.order_date < '2025-04-01';"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Q1 Revenue Report"}},
            {"step_id": "step_3", "tool": "send_notification", "params": {"channel": "email", "recipient": "stakeholders@company.com", "subject": "Q1 Revenue Report", "body": "Q1 revenue report is attached.", "attach_step": "step_2"}},
        ],
        "explanation": "Revenue is always from order_items (quantity * unit_price), not total_amount. Reports use markdown_table and are delivered via email.",
    },
    {
        "task": "Get completed order statistics and cache for the dashboard",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT COUNT(*) AS completed FROM orders WHERE status IN ('shipped', 'delivered');"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "cache", "target_name": "dashboard_completed_stats"}},
        ],
        "explanation": "'Completed' orders = status IN ('shipped', 'delivered'). Dashboard data goes to cache.",
    },
    {
        "task": "Archive the current valid order data as CSV",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT * FROM orders WHERE status NOT IN ('cancelled', 'returned') ORDER BY order_date;"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "csv_file", "target_name": "valid_orders.csv", "mode": "append"}},
        ],
        "explanation": "'Valid orders' excludes BOTH cancelled AND returned. Archiving uses csv_file with append.",
    },
    {
        "task": "Alert the team about the current fulfillment metrics",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT ROUND(COUNT(*) FILTER (WHERE status = 'delivered')::numeric / NULLIF(COUNT(*) FILTER (WHERE status != 'cancelled'), 0) * 100, 1) AS fulfillment_rate_pct FROM orders;"}},
            {"step_id": "step_2", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#data-alerts", "subject": "Fulfillment Metrics", "body": "Current fulfillment metrics are ready.", "attach_step": "step_1"}},
        ],
        "explanation": "Fulfillment rate = delivered / (total - cancelled). Alerts use Slack #data-alerts.",
    },
    {
        "task": "Build a pipeline for net order value analysis and store for dashboard",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT o.order_id, (o.total_amount - o.discount) AS net_value FROM orders o WHERE o.status NOT IN ('cancelled', 'returned');"}},
            {"step_id": "step_2", "tool": "transform_data", "params": {"source_step": "step_1", "operation": "aggregate", "params": {"metrics": {"net_value": "avg"}}}},
            {"step_id": "step_3", "tool": "store_results", "params": {"source_step": "step_2", "target": "cache", "target_name": "dashboard_net_value"}},
        ],
        "explanation": "Net order value = total_amount - discount. Pipelines include transform. Dashboard -> cache.",
    },
    {
        "task": "Get product category performance and archive the data",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON p.product_id = oi.product_id JOIN orders o ON o.order_id = oi.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY p.category ORDER BY revenue DESC;"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "csv_file", "target_name": "category_performance.csv", "mode": "append"}},
        ],
        "explanation": "Product performance uses revenue from order_items. Archiving uses csv_file with append.",
    },
    {
        "task": "Schedule a daily report of active customer counts",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT COUNT(DISTINCT c.customer_id) AS active_customers FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status NOT IN ('cancelled', 'returned');"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Active Customer Count"}},
            {"step_id": "step_3", "tool": "schedule_task", "params": {"task_name": "daily_active_customers", "workflow_steps": [{"tool": "query_database"}, {"tool": "generate_report"}], "interval": "daily", "notify_on_failure": True}},
        ],
        "explanation": "Active = non-cancelled, non-returned orders. Scheduled reports use daily + notify_on_failure=true.",
    },
    {
        "task": "Get repeat customer data and generate a report about it",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT c.name, c.email, COUNT(o.order_id) AS order_count FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY c.customer_id, c.name, c.email HAVING COUNT(o.order_id) > 1;"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Repeat Customers Report"}},
        ],
        "explanation": "Repeat customers have > 1 valid order. Reports use markdown_table.",
    },
    {
        "task": "Get basket sizes per order and alert about unusual patterns",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT o.order_id, SUM(oi.quantity) AS item_count FROM orders o JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY o.order_id;"}},
            {"step_id": "step_2", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#data-alerts", "subject": "Basket Size Analysis", "body": "Basket size data is ready for review.", "attach_step": "step_1"}},
        ],
        "explanation": "Basket size = number of items (quantity), not dollar amount. Alerts -> Slack #data-alerts.",
    },
    {
        "task": "Find products with no sales and generate a report",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT p.name, p.category, p.price FROM products p LEFT JOIN order_items oi ON oi.product_id = p.product_id WHERE oi.item_id IS NULL ORDER BY p.name;"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Unsold Products Report"}},
        ],
        "explanation": "Use LEFT JOIN + IS NULL for 'never sold'. Reports use markdown_table.",
    },
    {
        "task": "Customer ranking pipeline: get spending data, transform, and cache",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT c.name, c.segment, SUM(oi.quantity * oi.unit_price) AS total_spending FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY c.customer_id, c.name, c.segment ORDER BY total_spending DESC;"}},
            {"step_id": "step_2", "tool": "transform_data", "params": {"source_step": "step_1", "operation": "compute", "params": {"new_column": "spending_rank", "expression": "RANK() OVER (ORDER BY total_spending DESC)"}}},
            {"step_id": "step_3", "tool": "store_results", "params": {"source_step": "step_2", "target": "cache", "target_name": "dashboard_customer_ranking"}},
        ],
        "explanation": "Customer spending uses order_items revenue. Pipelines include transform. Dashboard -> cache.",
    },
    {
        "task": "Monthly valid orders trend and email the report",
        "tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT DATE_TRUNC('month', o.order_date)::date AS month, COUNT(*) AS valid_orders FROM orders o WHERE o.status NOT IN ('cancelled', 'returned') AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' GROUP BY month ORDER BY month;"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Monthly Valid Orders"}},
            {"step_id": "step_3", "tool": "send_notification", "params": {"channel": "email", "recipient": "manager@company.com", "subject": "Monthly Valid Orders Report", "body": "Valid orders trend report is attached.", "attach_step": "step_2"}},
        ],
        "explanation": "Valid orders exclude cancelled AND returned. Reports go via email.",
    },
]


def get_seed_traces() -> list[ExecutionTrace]:
    """Return all 12 seed traces as ExecutionTrace objects."""
    return [
        _build_trace(raw["task"], raw["tool_chain"], raw["explanation"])
        for raw in SEED_TRACES_RAW
    ]
