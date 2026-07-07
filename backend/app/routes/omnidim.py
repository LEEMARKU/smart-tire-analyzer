"""
OmniDimension Voice AI — API Routes
Provides endpoints to manage and query the voice AI technical support agent.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.services.omnidim_service import omnidim_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/support", tags=["Voice AI Support"])


@router.get("/voice-agent")
async def get_voice_agent_status():
    """Get the current status of the Voice AI support agent."""
    status = await omnidim_service.get_agent_status()
    return status


@router.post("/voice-agent/init")
async def init_voice_agent():
    """Initialize or get the Voice AI support agent on OmniDimension."""
    result = await omnidim_service.get_or_create_agent()
    return result
