"""MCP server exposing ml-intern's read-only research tools.

Lets Claude Code, Codex CLI, and other MCP-aware clients use ml-intern's
Hugging Face papers/datasets/docs/repos and GitHub search tools without
running the ml-intern agent loop or providing an LLM API key. The host
agent (Claude Code / Codex) drives the conversation; this server only
provides tools.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.function_tool import FunctionTool

from agent.core.hf_tokens import resolve_hf_token


@dataclass
class _StubSession:
    """Minimal stand-in for agent.core.session.Session.

    Carries only the attributes the exposed handlers actually read.
    Tools that need event_queue / send_event / running-job tracking
    (hf_jobs, sandbox, notify, plan) are intentionally not exposed.
    """

    hf_token: str | None = None


def _resolve_session() -> _StubSession:
    return _StubSession(hf_token=resolve_hf_token(os.environ.get("HF_TOKEN")))


def _wrap_handler(
    handler: Callable[..., Awaitable[tuple[str, bool]]],
    *,
    pass_session: bool,
) -> Callable[..., Awaitable[str]]:
    """Adapt an ml-intern handler to a FastMCP-callable coroutine.

    ml-intern handlers return (text, success). FastMCP signals failure by
    raising ToolError, so we translate.
    """

    async def _call(**kwargs: Any) -> str:
        arguments = dict(kwargs)
        if pass_session:
            output, ok = await handler(arguments, session=_resolve_session())
        else:
            output, ok = await handler(arguments)
        if not ok:
            raise ToolError(output)
        return output

    return _call


def _register_tools(mcp: FastMCP) -> None:
    from agent.tools.dataset_tools import (
        HF_INSPECT_DATASET_TOOL_SPEC,
        hf_inspect_dataset_handler,
    )
    from agent.tools.docs_tools import (
        EXPLORE_HF_DOCS_TOOL_SPEC,
        HF_DOCS_FETCH_TOOL_SPEC,
        explore_hf_docs_handler,
        hf_docs_fetch_handler,
    )
    from agent.tools.github_find_examples import (
        GITHUB_FIND_EXAMPLES_TOOL_SPEC,
        github_find_examples_handler,
    )
    from agent.tools.github_list_repos import (
        GITHUB_LIST_REPOS_TOOL_SPEC,
        github_list_repos_handler,
    )
    from agent.tools.github_read_file import (
        GITHUB_READ_FILE_TOOL_SPEC,
        github_read_file_handler,
    )
    from agent.tools.hf_repo_files_tool import (
        HF_REPO_FILES_TOOL_SPEC,
        hf_repo_files_handler,
    )
    from agent.tools.hf_repo_git_tool import (
        HF_REPO_GIT_TOOL_SPEC,
        hf_repo_git_handler,
    )
    from agent.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler
    from agent.tools.web_search_tool import (
        WEB_SEARCH_TOOL_SPEC,
        web_search_handler,
    )

    entries: list[tuple[dict[str, Any], Callable[..., Any], bool]] = [
        (HF_PAPERS_TOOL_SPEC, hf_papers_handler, False),
        (WEB_SEARCH_TOOL_SPEC, web_search_handler, True),
        (EXPLORE_HF_DOCS_TOOL_SPEC, explore_hf_docs_handler, True),
        (HF_DOCS_FETCH_TOOL_SPEC, hf_docs_fetch_handler, True),
        (HF_INSPECT_DATASET_TOOL_SPEC, hf_inspect_dataset_handler, True),
        (HF_REPO_FILES_TOOL_SPEC, hf_repo_files_handler, True),
        (HF_REPO_GIT_TOOL_SPEC, hf_repo_git_handler, True),
        (GITHUB_FIND_EXAMPLES_TOOL_SPEC, github_find_examples_handler, False),
        (GITHUB_LIST_REPOS_TOOL_SPEC, github_list_repos_handler, False),
        (GITHUB_READ_FILE_TOOL_SPEC, github_read_file_handler, False),
    ]

    for spec, handler, needs_session in entries:
        fn = _wrap_handler(handler, pass_session=needs_session)
        fn.__name__ = spec["name"]
        mcp.add_tool(
            FunctionTool(
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
                fn=fn,
            )
        )


def _load_env() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    load_dotenv(override=False)


def main() -> None:
    _load_env()
    logging.basicConfig(level=logging.INFO)
    mcp = FastMCP(
        name="ml-intern",
        instructions=(
            "Hugging Face research tools from ml-intern: paper / dataset / "
            "docs discovery, HF repo file & git operations, and GitHub code "
            "search. All tools are read-only or scoped to the configured HF "
            "account. Set HF_TOKEN and GITHUB_TOKEN in the environment."
        ),
    )
    _register_tools(mcp)
    mcp.run()


if __name__ == "__main__":
    main()
