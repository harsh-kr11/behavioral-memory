"""Seven benchmark MCP tool definitions for evaluation.

These simulate a realistic multi-tool MCP ecosystem for an enterprise
data operations platform, as described in the paper's evaluation
(Section IV.A). Used for both the reference agent's benchmark and
the gatekeeper's structural validation.

Tool categories:
  - Data retrieval:  query_database, fetch_api_data
  - Transformation:  transform_data
  - Storage:         store_results
  - Communication:   send_notification, generate_report
  - System:          schedule_task
"""

from __future__ import annotations

from behavioral_memory.core.schemas import ToolSchema

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "query_database",
        "description": (
            "Execute a read-only SQL query against the operational database. "
            "Returns structured rows. Supports PostgreSQL syntax. "
            "Use for retrieving raw business data from tables: customers, "
            "orders, order_items, products, reviews."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PostgreSQL SELECT query to execute",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to return (default: 100)",
                    "default": 100,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Query timeout in seconds (default: 30)",
                    "default": 30,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_api_data",
        "description": (
            "Fetch data from an external REST API endpoint. "
            "Supports GET requests with optional query parameters and headers. "
            "Use for retrieving external data like exchange rates, weather, "
            "or third-party service data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL of the API endpoint",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method (default: GET)",
                    "default": "GET",
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters as key-value pairs",
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP headers as key-value pairs",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "transform_data",
        "description": (
            "Apply a transformation operation to a dataset. "
            "Supports: aggregate (sum, avg, count, min, max), filter, sort, "
            "join (merge two datasets), pivot, and compute (add calculated fields). "
            "Input data comes from previous tool outputs referenced by step_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_step": {
                    "type": "string",
                    "description": "Step ID of the source data (from a previous tool call)",
                },
                "operation": {
                    "type": "string",
                    "enum": ["aggregate", "filter", "sort", "join", "pivot", "compute"],
                    "description": "Transformation operation to apply",
                },
                "params": {
                    "type": "object",
                    "description": "Operation-specific parameters",
                },
            },
            "required": ["source_step", "operation", "params"],
        },
    },
    {
        "name": "store_results",
        "description": (
            "Persist processed data to a destination. "
            "Supports targets: database_table, csv_file, json_file, cache. "
            "Use after data has been retrieved and/or transformed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_step": {
                    "type": "string",
                    "description": "Step ID of the data to store",
                },
                "target": {
                    "type": "string",
                    "enum": ["database_table", "csv_file", "json_file", "cache"],
                    "description": "Storage destination type",
                },
                "target_name": {
                    "type": "string",
                    "description": "Name of the destination (table name, file path, or cache key)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "description": "Write mode (default: overwrite)",
                    "default": "overwrite",
                },
            },
            "required": ["source_step", "target", "target_name"],
        },
    },
    {
        "name": "send_notification",
        "description": (
            "Send a notification through a specified channel. "
            "Supports channels: email, slack, webhook. "
            "Use for alerting stakeholders about completed analyses, "
            "anomalies, or report availability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["email", "slack", "webhook"],
                    "description": "Notification channel",
                },
                "recipient": {
                    "type": "string",
                    "description": "Recipient address (email, Slack channel, or webhook URL)",
                },
                "subject": {
                    "type": "string",
                    "description": "Notification subject or title",
                },
                "body": {
                    "type": "string",
                    "description": "Notification body content",
                },
                "attach_step": {
                    "type": "string",
                    "description": "Optional step ID whose output to attach",
                },
            },
            "required": ["channel", "recipient", "subject", "body"],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "Generate a formatted report from processed data. "
            "Supports formats: summary_text, markdown_table, csv, chart_config. "
            "Use to produce human-readable output from analysis results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_step": {
                    "type": "string",
                    "description": "Step ID of the data to report on",
                },
                "format": {
                    "type": "string",
                    "enum": ["summary_text", "markdown_table", "csv", "chart_config"],
                    "description": "Output format for the report",
                },
                "title": {
                    "type": "string",
                    "description": "Report title",
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include timestamps and source info (default: true)",
                    "default": True,
                },
            },
            "required": ["source_step", "format", "title"],
        },
    },
    {
        "name": "schedule_task",
        "description": (
            "Schedule a workflow or tool chain for recurring execution. "
            "Supports intervals: hourly, daily, weekly, monthly, cron. "
            "Use for setting up automated pipelines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_name": {
                    "type": "string",
                    "description": "Unique name for the scheduled task",
                },
                "workflow_steps": {
                    "type": "array",
                    "description": "List of tool call specifications to execute in order",
                    "items": {"type": "object"},
                },
                "interval": {
                    "type": "string",
                    "enum": ["hourly", "daily", "weekly", "monthly", "cron"],
                    "description": "Execution frequency",
                },
                "cron_expression": {
                    "type": "string",
                    "description": "Cron expression (required if interval is 'cron')",
                },
                "notify_on_failure": {
                    "type": "boolean",
                    "description": "Send alert on task failure (default: true)",
                    "default": True,
                },
            },
            "required": ["task_name", "workflow_steps", "interval"],
        },
    },
]


def get_tool_schemas() -> list[ToolSchema]:
    """Convert raw tool definitions into ToolSchema models."""
    schemas: list[ToolSchema] = []
    for td in TOOL_DEFINITIONS:
        input_schema = td.get("input_schema", {})
        schemas.append(
            ToolSchema(
                name=td["name"],
                description=td["description"],
                parameters_schema=input_schema,
                required_params=input_schema.get("required", []),
            )
        )
    return schemas


def get_tool_names() -> list[str]:
    return [t["name"] for t in TOOL_DEFINITIONS]
