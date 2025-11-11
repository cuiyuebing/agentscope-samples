# -*- coding: utf-8 -*-
"""Standalone file download skill for the browser agent."""

from __future__ import annotations

import uuid
import json
from typing import Any

from agentscope.message import (
    Msg,
    TextBlock,
    ToolUseBlock,
)
from agentscope.tool import ToolResponse


async def file_download(
    browser_agent: Any,
    target_description: str,
) -> ToolResponse:
    """
    Download the target file. The current page should
    contain download-related element.

    Args:
        target_description (str): The description of the
        target file to download.

    Returns:
        ToolResponse: A structured response containing
        the download directory.
    """

    snapshot_chunks = (
        await browser_agent._get_snapshot_in_text()  # noqa: E501 # pylint: disable=protected-access
    )
    page_snapshot = snapshot_chunks[0] if snapshot_chunks else ""

    sys_prompt = (
        "You are a meticulous web automation specialist. "
        "Given the page snapshot, "
        "identify the download-related element on the webpage."
        "identify the exact element and its reference "
        "string (ref) that matches the elememt. "
        "Return ONLY a JSON object: "
        '{"element": <element description>, "ref": <ref string>}'
        ""
    )
    user_prompt = (
        f"Page snapshot:\n{page_snapshot}\n"
        f"Target description: {target_description}\n"
    )

    prompt = await browser_agent.formatter.format(
        msgs=[
            Msg("system", sys_prompt, role="system"),
            Msg("user", user_prompt, role="user"),
        ],
    )
    res = await browser_agent.model(prompt)
    if browser_agent.model.stream:
        async for chunk in res:
            model_text = chunk.content[0]["text"]
    else:
        model_text = res.content[0]["text"]

    try:
        if "```json" in model_text:
            model_text = model_text.replace("```json", "").replace(
                "```",
                "",
            )
        element_info = json.loads(model_text)
        element = element_info.get("element", "")
        ref = element_info.get("ref", "")
    except Exception:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Failed to parse element/ref from model output.",
                ),
            ],
            metadata={"success": False},
        )

    interaction_tool_call = ToolUseBlock(
        id=str(uuid.uuid4()),
        name="browser_click",
        input={"element": element, "ref": ref},
        type="tool_use",
    )
    tool_res = await browser_agent.toolkit.call_tool_function(
        interaction_tool_call,
    )
    tool_res_msg_text = ""
    # Async generator handling

    async for chunk in tool_res:
        # Turn into a tool result block
        tool_res_msg_text = chunk.content[0]["text"]

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    f"file downloaded for \nelement: {element}"
                    f" and \nref: {ref}"
                    f"tool_res_msg_text: {tool_res_msg_text}"
                ),
            ),
        ],
    )
