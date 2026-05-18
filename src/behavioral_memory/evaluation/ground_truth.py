"""Ground truth: 30 tasks with gold tool chains for evaluation.

10 simple (1 step), 10 moderate (2-3 steps), 10 challenging (3-6 steps).
Convention-sensitive terms are taught by the 12 seed traces in seed_traces.py.

See the paper Section IV.A for the full benchmark design rationale.
"""

from __future__ import annotations

EVALUATION_TASKS: list[dict] = [
    # ═══════════════════════════════════════════
    #  SIMPLE — single-tool operations (10)
    # ═══════════════════════════════════════════
    {
        "task_id": 0,
        "task": "Get the total number of customers from the database",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT COUNT(*) AS customer_count FROM customers;"}},
        ],
    },
    {
        "task_id": 1,
        "task": "Fetch the current exchange rate from the forex API",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "fetch_api_data", "params": {"url": "https://api.exchangerate.host/latest", "method": "GET"}},
        ],
    },
    {
        "task_id": 2,
        "task": "Send a Slack message to the team about the deployment",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#team", "subject": "Deployment Update", "body": "Deployment completed successfully."}},
        ],
    },
    {
        "task_id": 3,
        "task": "Get all products and their prices from the database",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT name, price FROM products ORDER BY name;"}},
        ],
    },
    {
        "task_id": 4,
        "task": "Store the sales summary as a JSON file",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "store_results", "params": {"source_step": "previous_step", "target": "json_file", "target_name": "sales_summary.json"}},
        ],
    },
    {
        "task_id": 5,
        "task": "Schedule a weekly backup task",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "schedule_task", "params": {"task_name": "weekly_backup", "workflow_steps": [], "interval": "weekly"}},
        ],
    },
    {
        "task_id": 6,
        "task": "Get the count of orders in the system",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT COUNT(*) AS order_count FROM orders;"}},
        ],
    },
    {
        "task_id": 7,
        "task": "Send an email notification about the monthly report being ready",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "send_notification", "params": {"channel": "email", "recipient": "team@company.com", "subject": "Monthly Report Ready", "body": "The monthly report has been generated and is ready for review."}},
        ],
    },
    {
        "task_id": 8,
        "task": "Generate a text summary from the analysis results",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "generate_report", "params": {"source_step": "previous_step", "format": "summary_text", "title": "Analysis Summary"}},
        ],
    },
    {
        "task_id": 9,
        "task": "Query the database for delivered orders",
        "difficulty": "simple",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT * FROM orders WHERE status = 'delivered';"}},
        ],
    },
    # ═══════════════════════════════════════════
    #  MODERATE — multi-step with conventions (10)
    # ═══════════════════════════════════════════
    {
        "task_id": 10,
        "task": "Get the total revenue and generate a report",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT SUM(oi.quantity * oi.unit_price) AS total_revenue FROM order_items oi JOIN orders o ON o.order_id = oi.order_id WHERE o.status NOT IN ('cancelled', 'returned');"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Total Revenue Report"}},
        ],
    },
    {
        "task_id": 11,
        "task": "Retrieve valid orders and store them as a CSV archive",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT * FROM orders WHERE status NOT IN ('cancelled', 'returned') ORDER BY order_date;"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "csv_file", "target_name": "valid_orders_archive.csv", "mode": "append"}},
        ],
    },
    {
        "task_id": 12,
        "task": "Get revenue by category and send an alert about it",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON p.product_id = oi.product_id JOIN orders o ON o.order_id = oi.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY p.category ORDER BY revenue DESC;"}},
            {"step_id": "step_2", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#data-alerts", "subject": "Revenue by Category", "body": "Revenue breakdown by product category is ready.", "attach_step": "step_1"}},
        ],
    },
    {
        "task_id": 13,
        "task": "Build a pipeline to get completed orders count and cache it for the dashboard",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT COUNT(*) AS completed_orders FROM orders WHERE status IN ('shipped', 'delivered');"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "cache", "target_name": "dashboard_completed_orders"}},
        ],
    },
    {
        "task_id": 14,
        "task": "Fetch customer data, filter active customers, and generate a report",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT DISTINCT c.customer_id, c.name, c.email, c.segment FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status NOT IN ('cancelled', 'returned');"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Active Customers Report"}},
        ],
    },
    {
        "task_id": 15,
        "task": "Get order data and email the report to the manager",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT o.order_id, o.order_date, o.status, o.total_amount FROM orders o WHERE o.status NOT IN ('cancelled', 'returned') ORDER BY o.order_date DESC;"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Order Status Report"}},
            {"step_id": "step_3", "tool": "send_notification", "params": {"channel": "email", "recipient": "manager@company.com", "subject": "Order Status Report", "body": "Please find the latest order status report attached.", "attach_step": "step_2"}},
        ],
    },
    {
        "task_id": 16,
        "task": "Query product revenue data and save it to the dashboard cache",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON p.product_id = oi.product_id JOIN orders o ON o.order_id = oi.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY p.name ORDER BY revenue DESC;"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "cache", "target_name": "dashboard_product_revenue"}},
        ],
    },
    {
        "task_id": 17,
        "task": "Get repeat customer data and archive it as CSV",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT c.customer_id, c.name, c.email, COUNT(o.order_id) AS order_count FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY c.customer_id, c.name, c.email HAVING COUNT(o.order_id) > 1;"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "csv_file", "target_name": "repeat_customers_archive.csv", "mode": "append"}},
        ],
    },
    {
        "task_id": 18,
        "task": "Get the fulfillment rate and alert the operations team",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT ROUND(COUNT(*) FILTER (WHERE status = 'delivered')::numeric / NULLIF(COUNT(*) FILTER (WHERE status != 'cancelled'), 0) * 100, 1) AS fulfillment_rate_pct FROM orders;"}},
            {"step_id": "step_2", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#data-alerts", "subject": "Fulfillment Rate Update", "body": "Current order fulfillment rate has been calculated.", "attach_step": "step_1"}},
        ],
    },
    {
        "task_id": 19,
        "task": "Get net order values and store as dashboard data",
        "difficulty": "moderate",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT o.order_id, (o.total_amount - o.discount) AS net_value FROM orders o WHERE o.status NOT IN ('cancelled', 'returned');"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "cache", "target_name": "dashboard_net_order_values"}},
        ],
    },
    # ═══════════════════════════════════════════
    #  CHALLENGING — complex multi-step (10)
    # ═══════════════════════════════════════════
    {
        "task_id": 20,
        "task": "Build a complete revenue analysis pipeline: query revenue by category, transform to add percentage, generate a report, email it, and archive the data",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON p.product_id = oi.product_id JOIN orders o ON o.order_id = oi.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY p.category ORDER BY revenue DESC;"}},
            {"step_id": "step_2", "tool": "transform_data", "params": {"source_step": "step_1", "operation": "compute", "params": {"new_column": "revenue_pct", "expression": "revenue / SUM(revenue) * 100"}}},
            {"step_id": "step_3", "tool": "generate_report", "params": {"source_step": "step_2", "format": "markdown_table", "title": "Revenue by Category Analysis"}},
            {"step_id": "step_4", "tool": "send_notification", "params": {"channel": "email", "recipient": "stakeholders@company.com", "subject": "Revenue by Category Report", "body": "Please find the revenue analysis report attached.", "attach_step": "step_3"}},
            {"step_id": "step_5", "tool": "store_results", "params": {"source_step": "step_2", "target": "csv_file", "target_name": "revenue_by_category_archive.csv", "mode": "append"}},
        ],
    },
    {
        "task_id": 21,
        "task": "Create a customer segmentation pipeline: get customer spending, aggregate by segment, generate report, cache for dashboard, and alert the team",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT c.segment, COUNT(DISTINCT c.customer_id) AS customers, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY c.segment ORDER BY total_revenue DESC;"}},
            {"step_id": "step_2", "tool": "transform_data", "params": {"source_step": "step_1", "operation": "compute", "params": {"new_column": "avg_revenue_per_customer", "expression": "total_revenue / customers"}}},
            {"step_id": "step_3", "tool": "generate_report", "params": {"source_step": "step_2", "format": "markdown_table", "title": "Customer Segmentation Report"}},
            {"step_id": "step_4", "tool": "store_results", "params": {"source_step": "step_2", "target": "cache", "target_name": "dashboard_customer_segmentation"}},
            {"step_id": "step_5", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#data-alerts", "subject": "Customer Segmentation Updated", "body": "Customer segmentation analysis has been refreshed.", "attach_step": "step_3"}},
        ],
    },
    {
        "task_id": 22,
        "task": "Set up a scheduled daily report: query valid orders with revenue, generate a markdown report, and schedule it to run daily with failure notifications",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT o.order_id, o.order_date, o.status, SUM(oi.quantity * oi.unit_price) AS revenue FROM orders o JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY o.order_id, o.order_date, o.status ORDER BY o.order_date DESC;"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Daily Valid Orders Report"}},
            {"step_id": "step_3", "tool": "schedule_task", "params": {"task_name": "daily_valid_orders_report", "workflow_steps": [{"tool": "query_database"}, {"tool": "generate_report"}], "interval": "daily", "notify_on_failure": True}},
        ],
    },
    {
        "task_id": 23,
        "task": "Build a product performance pipeline: get product revenue, transform to compute margin, generate report, and archive",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT p.name, p.price, p.cost, SUM(oi.quantity * oi.unit_price) AS revenue FROM products p JOIN order_items oi ON oi.product_id = p.product_id JOIN orders o ON o.order_id = oi.order_id WHERE o.status NOT IN ('cancelled', 'returned') AND p.cost IS NOT NULL AND p.cost > 0 GROUP BY p.product_id, p.name, p.price, p.cost ORDER BY revenue DESC;"}},
            {"step_id": "step_2", "tool": "transform_data", "params": {"source_step": "step_1", "operation": "compute", "params": {"new_column": "margin_pct", "expression": "(price - cost) / price * 100"}}},
            {"step_id": "step_3", "tool": "generate_report", "params": {"source_step": "step_2", "format": "markdown_table", "title": "Product Performance Report"}},
            {"step_id": "step_4", "tool": "store_results", "params": {"source_step": "step_2", "target": "csv_file", "target_name": "product_performance_archive.csv", "mode": "append"}},
        ],
    },
    {
        "task_id": 24,
        "task": "Create a monthly revenue trend pipeline: query monthly revenue, cache for dashboard, generate chart config, and email stakeholders",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT DATE_TRUNC('month', o.order_date)::date AS month, SUM(oi.quantity * oi.unit_price) AS monthly_revenue FROM orders o JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status IN ('shipped', 'delivered') AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' GROUP BY month ORDER BY month;"}},
            {"step_id": "step_2", "tool": "store_results", "params": {"source_step": "step_1", "target": "cache", "target_name": "dashboard_monthly_revenue_2025"}},
            {"step_id": "step_3", "tool": "generate_report", "params": {"source_step": "step_1", "format": "chart_config", "title": "Monthly Revenue Trend 2025"}},
            {"step_id": "step_4", "tool": "send_notification", "params": {"channel": "email", "recipient": "stakeholders@company.com", "subject": "Monthly Revenue Trend Report", "body": "Monthly revenue trend for 2025 is available.", "attach_step": "step_3"}},
        ],
    },
    {
        "task_id": 25,
        "task": "Build an order health monitoring pipeline: get valid orders per month, compute growth, generate report, alert ops, and archive",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT DATE_TRUNC('month', o.order_date)::date AS month, COUNT(*) AS valid_orders FROM orders o WHERE o.status NOT IN ('cancelled', 'returned') AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' GROUP BY month ORDER BY month;"}},
            {"step_id": "step_2", "tool": "transform_data", "params": {"source_step": "step_1", "operation": "compute", "params": {"new_column": "mom_growth_pct", "expression": "(valid_orders - LAG(valid_orders)) / LAG(valid_orders) * 100"}}},
            {"step_id": "step_3", "tool": "generate_report", "params": {"source_step": "step_2", "format": "markdown_table", "title": "Order Health Monitor"}},
            {"step_id": "step_4", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#data-alerts", "subject": "Order Health Update", "body": "Monthly order health report is ready.", "attach_step": "step_3"}},
            {"step_id": "step_5", "tool": "store_results", "params": {"source_step": "step_2", "target": "csv_file", "target_name": "order_health_archive.csv", "mode": "append"}},
        ],
    },
    {
        "task_id": 26,
        "task": "Create a basket analysis workflow: query basket sizes by segment, transform to rank, generate report, cache for dashboard",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT c.segment, ROUND(AVG(sub.item_count)::numeric, 1) AS avg_basket_size FROM (SELECT o.order_id, o.customer_id, SUM(oi.quantity) AS item_count FROM orders o JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY o.order_id, o.customer_id) sub JOIN customers c ON c.customer_id = sub.customer_id GROUP BY c.segment ORDER BY avg_basket_size DESC;"}},
            {"step_id": "step_2", "tool": "transform_data", "params": {"source_step": "step_1", "operation": "sort", "params": {"column": "avg_basket_size", "order": "desc"}}},
            {"step_id": "step_3", "tool": "generate_report", "params": {"source_step": "step_2", "format": "markdown_table", "title": "Basket Size Analysis by Segment"}},
            {"step_id": "step_4", "tool": "store_results", "params": {"source_step": "step_2", "target": "cache", "target_name": "dashboard_basket_analysis"}},
        ],
    },
    {
        "task_id": 27,
        "task": "Set up an automated customer churn alert: query at-risk customers, generate report, send alert, schedule weekly",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT c.name, c.email, c.segment, COUNT(o.order_id) AS total_orders FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.name, c.email, c.segment HAVING COUNT(o.order_id) = COUNT(CASE WHEN o.status IN ('cancelled', 'returned') THEN 1 END);"}},
            {"step_id": "step_2", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Customer Churn Risk Report"}},
            {"step_id": "step_3", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#data-alerts", "subject": "Customer Churn Alert", "body": "Customers with potential churn risk identified.", "attach_step": "step_2"}},
            {"step_id": "step_4", "tool": "schedule_task", "params": {"task_name": "weekly_churn_alert", "workflow_steps": [{"tool": "query_database"}, {"tool": "generate_report"}, {"tool": "send_notification"}], "interval": "weekly", "notify_on_failure": True}},
        ],
    },
    {
        "task_id": 28,
        "task": "Build a cross-channel reporting pipeline: query top customers, enrich with CRM API, generate report, email and archive",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT c.name, c.segment, SUM(oi.quantity * oi.unit_price) AS total_spending FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status NOT IN ('cancelled', 'returned') GROUP BY c.customer_id, c.name, c.segment ORDER BY total_spending DESC LIMIT 10;"}},
            {"step_id": "step_2", "tool": "fetch_api_data", "params": {"url": "https://crm.company.com/api/customer-scores", "method": "GET"}},
            {"step_id": "step_3", "tool": "transform_data", "params": {"source_step": "step_1", "operation": "join", "params": {"right_step": "step_2", "on": "name", "how": "left"}}},
            {"step_id": "step_4", "tool": "generate_report", "params": {"source_step": "step_3", "format": "markdown_table", "title": "Top Customer Spending Report"}},
            {"step_id": "step_5", "tool": "send_notification", "params": {"channel": "email", "recipient": "stakeholders@company.com", "subject": "Top Customer Report", "body": "Top customer spending report is ready.", "attach_step": "step_4"}},
            {"step_id": "step_6", "tool": "store_results", "params": {"source_step": "step_3", "target": "csv_file", "target_name": "top_customers_archive.csv", "mode": "append"}},
        ],
    },
    {
        "task_id": 29,
        "task": "Create a full inventory monitoring pipeline: query unsold products, fetch supplier API, generate report, alert team, cache, and schedule weekly",
        "difficulty": "challenging",
        "gold_tool_chain": [
            {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT p.name, p.category, p.price, p.stock_qty FROM products p LEFT JOIN order_items oi ON oi.product_id = p.product_id WHERE oi.item_id IS NULL ORDER BY p.name;"}},
            {"step_id": "step_2", "tool": "fetch_api_data", "params": {"url": "https://supplier.company.com/api/restock-status", "method": "GET"}},
            {"step_id": "step_3", "tool": "generate_report", "params": {"source_step": "step_1", "format": "markdown_table", "title": "Unsold Products Inventory Report"}},
            {"step_id": "step_4", "tool": "send_notification", "params": {"channel": "slack", "recipient": "#data-alerts", "subject": "Inventory Alert: Unsold Products", "body": "Products with zero orders have been identified.", "attach_step": "step_3"}},
            {"step_id": "step_5", "tool": "store_results", "params": {"source_step": "step_1", "target": "cache", "target_name": "dashboard_unsold_products"}},
            {"step_id": "step_6", "tool": "schedule_task", "params": {"task_name": "weekly_inventory_monitor", "workflow_steps": [{"tool": "query_database"}, {"tool": "fetch_api_data"}, {"tool": "generate_report"}, {"tool": "send_notification"}, {"tool": "store_results"}], "interval": "weekly", "notify_on_failure": True}},
        ],
    },
]
