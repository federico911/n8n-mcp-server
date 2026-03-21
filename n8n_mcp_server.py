#!/usr/bin/env python3
"""
n8n MCP Server — Cloud HTTP version
Provides 10 tools for creating, managing and debugging n8n workflows
via the n8n REST API. Runs as a cloud service (Railway/Render).
"""

import os
import json
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

# ── Server init ──────────────────────────────────────────────────────────────
# Disable DNS rebinding protection: this server runs on Railway (cloud),
# not localhost — the default whitelist only allows 127.0.0.1/localhost.
mcp = FastMCP(
    "n8n_mcp",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)

# ── Config from environment ──────────────────────────────────────────────────
N8N_URL = os.environ.get("N8N_URL", "").rstrip("/")
N8N_API_KEY = os.environ.get("N8N_API_KEY", "")

if not N8N_URL:
    raise ValueError("N8N_URL environment variable is required (e.g. https://yourname.app.n8n.cloud)")
if not N8N_API_KEY:
    raise ValueError("N8N_API_KEY environment variable is required")

# ── Shared HTTP helpers ──────────────────────────────────────────────────────
def _headers() -> dict:
    return {
        "X-N8N-API-KEY": N8N_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

async def _request(method: str, endpoint: str, **kwargs) -> dict:
    """Reusable async HTTP call to n8n REST API."""
    url = f"{N8N_URL}/api/v1/{endpoint}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(method, url, headers=_headers(), **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

def _error(e: Exception) -> str:
    """Convert exceptions into clear, actionable error messages."""
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 401:
            return "Error: Invalid API key. Verify N8N_API_KEY is correct."
        if status == 403:
            return "Error: Forbidden. Your n8n plan may not support the REST API."
        if status == 404:
            return "Error: Resource not found. Check the ID."
        if status == 422:
            try:
                return f"Error: Validation failed — {json.dumps(e.response.json())}"
            except Exception:
                return "Error: Validation failed (422)."
        return f"Error: HTTP {status} — {e.response.text[:300]}"
    if isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. Try again."
    return f"Error: {type(e).__name__}: {str(e)[:300]}"

# ── Input models ─────────────────────────────────────────────────────────────
class WorkflowIdInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    workflow_id: str = Field(..., description="The n8n workflow ID (e.g. '123')")

class CreateWorkflowInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    workflow_json: str = Field(..., description="Complete workflow definition as a JSON string")

class UpdateWorkflowInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    workflow_id: str = Field(..., description="The n8n workflow ID to update")
    workflow_json: str = Field(..., description="Complete updated workflow JSON string")

class ListWorkflowsInput(BaseModel):
    limit: Optional[int] = Field(default=20, description="Max workflows to return (1-100)", ge=1, le=100)
    active: Optional[bool] = Field(default=None, description="Filter: True=active only, False=inactive only, None=all")

class ListExecutionsInput(BaseModel):
    workflow_id: Optional[str] = Field(default=None, description="Filter executions by workflow ID")
    status: Optional[str] = Field(default=None, description="Filter by status: 'success', 'error', 'waiting'")
    limit: Optional[int] = Field(default=10, description="Max executions to return (1-100)", ge=1, le=100)

class ExecutionIdInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    execution_id: str = Field(..., description="The execution ID")

class DeployFromFileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    file_path: str = Field(..., description="Absolute path to a JSON file containing the workflow definition")

class UpdateFromFileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    workflow_id: str = Field(..., description="The n8n workflow ID to update")
    file_path: str = Field(..., description="Absolute path to a JSON file containing the updated workflow definition")

# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="n8n_list_workflows",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def n8n_list_workflows(params: ListWorkflowsInput) -> str:
    """List all n8n workflows with their IDs, names and active status.

    Use this first to discover existing workflows before creating or modifying them.

    Returns:
        str: Formatted list of workflows with ID, name and active status.
             Example: "- ID: 42 | 🟢 Active | Name: LinkedIn Agent"
    """
    try:
        query = f"?limit={params.limit}"
        if params.active is not None:
            query += f"&active={'true' if params.active else 'false'}"
        data = await _request("GET", f"workflows{query}")
        workflows = data.get("data", [])
        if not workflows:
            return "No workflows found."
        lines = [f"Found {len(workflows)} workflow(s):\n"]
        for w in workflows:
            icon = "🟢" if w.get("active") else "⚫"
            lines.append(f"- ID: {w['id']} | {icon} {'Active' if w.get('active') else 'Inactive'} | Name: {w['name']}")
        return "\n".join(lines)
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_get_workflow",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def n8n_get_workflow(params: WorkflowIdInput) -> str:
    """Get the full JSON definition of a specific n8n workflow by ID.

    Use to inspect an existing workflow before modifying it, or to understand
    its node structure and connections.

    Returns:
        str: Full workflow JSON as a formatted string.
    """
    try:
        data = await _request("GET", f"workflows/{params.workflow_id}")
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_create_workflow",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
)
async def n8n_create_workflow(params: CreateWorkflowInput) -> str:
    """Create a new n8n workflow from a JSON definition.

    The workflow_json must include: name, nodes (array), connections (object),
    and settings. Do NOT include 'id' (assigned by n8n automatically).

    Returns:
        str: Success message with the new workflow ID and name.
             Example: "✅ Workflow created! ID: 42 | Name: My Agent"
    """
    try:
        workflow = json.loads(params.workflow_json)
        data = await _request("POST", "workflows", json=workflow)
        return f"✅ Workflow created! ID: {data.get('id')} | Name: {data.get('name')}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {str(e)}"
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_update_workflow",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def n8n_update_workflow(params: UpdateWorkflowInput) -> str:
    """Update an existing n8n workflow by replacing its full JSON definition.

    Always fetch the current workflow first with n8n_get_workflow, modify the JSON,
    then call this tool. The workflow_id must exist.

    Returns:
        str: Success message with workflow ID and name.
    """
    try:
        workflow = json.loads(params.workflow_json)
        data = await _request("PUT", f"workflows/{params.workflow_id}", json=workflow)
        return f"✅ Workflow updated! ID: {data.get('id')} | Name: {data.get('name')}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {str(e)}"
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_delete_workflow",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True}
)
async def n8n_delete_workflow(params: WorkflowIdInput) -> str:
    """Permanently delete an n8n workflow. This action cannot be undone.

    Always confirm the workflow ID with n8n_list_workflows before deleting.

    Returns:
        str: Confirmation message.
    """
    try:
        await _request("DELETE", f"workflows/{params.workflow_id}")
        return f"✅ Workflow {params.workflow_id} deleted permanently."
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_activate_workflow",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def n8n_activate_workflow(params: WorkflowIdInput) -> str:
    """Activate an n8n workflow, enabling its webhook triggers and schedules.

    The workflow must have at least one trigger node (Webhook, Schedule, etc.)
    and valid credentials to be activated successfully.

    Returns:
        str: Confirmation message.
    """
    try:
        await _request("POST", f"workflows/{params.workflow_id}/activate")
        return f"✅ Workflow {params.workflow_id} is now ACTIVE — triggers are live."
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_deactivate_workflow",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def n8n_deactivate_workflow(params: WorkflowIdInput) -> str:
    """Deactivate an n8n workflow, pausing its webhook triggers and schedules.

    Returns:
        str: Confirmation message.
    """
    try:
        await _request("POST", f"workflows/{params.workflow_id}/deactivate")
        return f"✅ Workflow {params.workflow_id} is now INACTIVE — triggers paused."
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_list_executions",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def n8n_list_executions(params: ListExecutionsInput) -> str:
    """List recent workflow executions for debugging and monitoring.

    Use after running a workflow to check if it succeeded or failed.
    Filter by workflow_id and/or status to narrow results.

    Returns:
        str: List of executions with ID, status, workflow ID and start time.
    """
    try:
        query = f"?limit={params.limit}"
        if params.workflow_id:
            query += f"&workflowId={params.workflow_id}"
        if params.status:
            query += f"&status={params.status}"
        data = await _request("GET", f"executions{query}")
        executions = data.get("data", [])
        if not executions:
            return "No executions found."
        lines = [f"Found {len(executions)} execution(s):\n"]
        for ex in executions:
            status = ex.get("status", "unknown")
            icon = "✅" if status == "success" else "❌" if status == "error" else "⏳"
            lines.append(
                f"- ID: {ex['id']} | {icon} {status} | "
                f"Workflow: {ex.get('workflowId')} | "
                f"Started: {ex.get('startedAt', 'N/A')}"
            )
        return "\n".join(lines)
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_get_execution",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def n8n_get_execution(params: ExecutionIdInput) -> str:
    """Get full details of a specific execution, including error messages and node outputs.

    IMPORTANT: Uses includeData=true to get actual error details from n8n Cloud.
    Without this parameter n8n Cloud returns data: {} making debugging impossible.

    Returns:
        str: JSON with execution details, extracted errors per node, and status.
    """
    try:
        # includeData=true is REQUIRED on n8n Cloud to get actual error details.
        # Without it, n8n returns data: {} and debugging is impossible.
        data = await _request("GET", f"executions/{params.execution_id}?includeData=true")

        # Extract errors clearly from the deeply nested n8n execution structure.
        errors = []
        result_data = data.get("data", {}).get("resultData", {})
        run_data = result_data.get("runData", {})

        for node_name, node_runs in run_data.items():
            for run in (node_runs or []):
                if run.get("error"):
                    err = run["error"]
                    errors.append({
                        "node": node_name,
                        "error": err.get("message", str(err)),
                        "description": err.get("description", ""),
                        "httpCode": err.get("httpCode", ""),
                    })

        # Top-level workflow error (e.g. uncaught exception)
        top_error = result_data.get("error")
        if top_error:
            errors.append({"node": "workflow", "error": str(top_error)})

        nodes_executed = list(run_data.keys()) if run_data else []

        summary = {
            "id": data.get("id"),
            "status": data.get("status"),
            "workflowId": data.get("workflowId"),
            "startedAt": data.get("startedAt"),
            "stoppedAt": data.get("stoppedAt"),
            "nodes_executed": nodes_executed,
            "last_node": nodes_executed[-1] if nodes_executed else None,
            "errors": errors,
            "error_count": len(errors),
        }
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_trigger_test",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
)
async def n8n_trigger_test(params: WorkflowIdInput) -> str:
    """Trigger a test execution of a workflow programmatically.

    Use this to test a workflow without requiring the user to click anything in n8n.
    Returns the execution ID so you can immediately call n8n_get_execution to check results.

    Returns:
        str: Execution ID to use with n8n_get_execution, or an error message.
    """
    try:
        data = await _request("POST", f"workflows/{params.workflow_id}/run", json={})
        exec_id = data.get("executionId") or data.get("id")
        return f"✅ Test triggered. Execution ID: {exec_id} — call n8n_get_execution with this ID."
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 405):
            return (
                "Note: programmatic trigger not supported for this workflow type on your n8n plan. "
                "Use n8n_list_executions to retrieve recent executions for analysis instead."
            )
        return _error(e)
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_deploy_from_file",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
)
async def n8n_deploy_from_file(params: DeployFromFileInput) -> str:
    """Create a new n8n workflow by reading its JSON definition from a local file.

    USE THIS instead of n8n_create_workflow when the workflow JSON is large (>10KB).
    The agent should write the JSON to a file using the Write tool, then call this
    tool with the file path. This bypasses all tool-call size limits.

    Workflow:
        1. Agent generates JSON and writes it to /tmp/workflow.json (Write tool)
        2. Agent calls n8n_deploy_from_file(file_path="/tmp/workflow.json")
        3. Server reads the file and deploys to n8n

    Returns:
        str: Success message with new workflow ID and name, or error details.
    """
    try:
        with open(params.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        workflow = json.loads(content)
        data = await _request("POST", "workflows", json=workflow)
        return f"✅ Workflow created from file! ID: {data.get('id')} | Name: {data.get('name')}"
    except FileNotFoundError:
        return f"Error: File not found at '{params.file_path}'. Write the JSON there first using the Write tool."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file — {str(e)}"
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_update_from_file",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def n8n_update_from_file(params: UpdateFromFileInput) -> str:
    """Update an existing n8n workflow by reading its JSON definition from a local file.

    USE THIS instead of n8n_update_workflow when the workflow JSON is large (>10KB).
    The agent should write the updated JSON to a file using the Write tool, then call
    this tool with the workflow_id and file path.

    Workflow:
        1. Agent fetches current workflow with n8n_get_workflow
        2. Agent modifies JSON and writes to /tmp/workflow_update.json (Write tool)
        3. Agent calls n8n_update_from_file(workflow_id="...", file_path="/tmp/workflow_update.json")

    Returns:
        str: Success message with workflow ID and name, or error details.
    """
    try:
        with open(params.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        workflow = json.loads(content)
        data = await _request("PUT", f"workflows/{params.workflow_id}", json=workflow)
        return f"✅ Workflow updated from file! ID: {data.get('id')} | Name: {data.get('name')}"
    except FileNotFoundError:
        return f"Error: File not found at '{params.file_path}'. Write the JSON there first using the Write tool."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file — {str(e)}"
    except Exception as e:
        return _error(e)


@mcp.tool(
    name="n8n_delete_execution",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True}
)
async def n8n_delete_execution(params: ExecutionIdInput) -> str:
    """Delete an execution record from n8n history.

    Returns:
        str: Confirmation message.
    """
    try:
        await _request("DELETE", f"executions/{params.execution_id}")
        return f"✅ Execution {params.execution_id} deleted."
    except Exception as e:
        return _error(e)


# ── OAuth discovery endpoints (required by MCP 2025 / Cowork) ────────────────
async def oauth_protected_resource(request: Request) -> JSONResponse:
    """Tell Cowork this server is public — no auth token needed."""
    base = str(request.base_url).rstrip("/")
    return JSONResponse({
        "resource": base,
        "bearer_methods_supported": []
    })

async def not_found(request: Request) -> JSONResponse:
    return JSONResponse({"error": "not_found"}, status_code=404)


# ── App assembly with proper lifespan propagation ────────────────────────────
mcp_asgi = mcp.streamable_http_app()

@asynccontextmanager
async def lifespan(app):
    async with mcp_asgi.router.lifespan_context(app):
        yield

app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/.well-known/oauth-protected-resource",     oauth_protected_resource),
        Route("/.well-known/oauth-protected-resource/mcp", oauth_protected_resource),
        Route("/.well-known/oauth-authorization-server",   not_found),
        Route("/register", not_found, methods=["POST"]),
        Mount("/", app=mcp_asgi),
    ]
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"🚀 n8n MCP Server starting on port {port}")
    print(f"🔗 n8n instance: {N8N_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)
