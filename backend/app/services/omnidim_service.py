"""
OmniDimension Voice AI Service
Creates and manages a voice AI agent for Smart Tire Analyzer Technical Support.
Powered by Llama 3.3 70B via omnidim.io.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class OmnidimService:
    _client: Any = None
    _agent: Dict[str, Any] | None = None
    _initialized: bool = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        api_key = settings.OMNIDIM_API_KEY
        if not api_key:
            logger.warning("OMNIDIM_API_KEY is not set. Voice AI support will be unavailable.")
            return False

        try:
            from omnidimension import Client
            self._client = Client(api_key)
            self._initialized = True
            logger.info("OmniDimension client initialized successfully")
            return True
        except Exception as e:
            logger.error("Failed to initialize OmniDimension client: %s", e)
            return False

    async def get_or_create_agent(self) -> Dict[str, Any]:
        if not self._initialized:
            ok = await self.initialize()
            if not ok:
                return {"status": "error", "message": "Voice AI service not configured. Set OMNIDIM_API_KEY in .env"}

        if self._agent is not None:
            return self._agent

        try:
            response = self._client.agent.create(
                name="Smart Tire Analyser Technical Support",
                welcome_message="Hello, you have reached Smart Tire Analyser Support. May I have your ticket or username and a brief description of the issue?",
                context_breakdown=[
                    {
                        "title": "Identity & Purpose",
                        "body": (
                            "- You are the first-line technical support agent for Smart Tire Analyser.\n"
                            "- You assist developers, data scientists, end-users, and field technicians with onboarding, troubleshooting, and product education.\n"
                            "- Your goal is to resolve issues quickly, ensure smooth onboarding, and collect feedback for model improvement.\n"
                            "- Calls are successful when the issue is addressed or escalated, and the user feels supported."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Facts",
                        "body": (
                            "- Smart Tire Analyser is a technical tool for tire analysis using data models.\n"
                            "- Supported platforms and OS: Windows 10/11, macOS 12+, Ubuntu 20.04+.\n"
                            "- QUICK_START steps and starter dataset layout are available for onboarding.\n"
                            "- Model-selection tiers and feedback submission are part of the product features.\n"
                            "- Documentation links are available for all common workflows.\n"
                            "- Pricing, hardware repairs, and advanced model customization: NOT AVAILABLE — offer a callback from the specialist team."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Actions & Limits",
                        "body": (
                            "- CAN: verify user/ticket, guide onboarding, explain product features, walk through troubleshooting, provide documentation links, escalate complex issues, and record feedback.\n"
                            "- CANNOT: perform live remote access, process payments, handle hardware repairs, or provide custom model development — collect details and offer a specialist callback for these."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Flow: Greeting & Verification",
                        "body": (
                            "On call start:\n"
                            "1. Greet and request ticket or username and a brief issue description.\n"
                            "2. Determine if the caller is a new user, reporting a bug/error, or seeking feature guidance.\n"
                            "3. Move to the relevant branch based on their response."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Flow: Onboarding Branch",
                        "body": (
                            "For new users:\n"
                            "1. Verify the caller's OS and environment.\n"
                            "2. Guide them through QUICK_START steps.\n"
                            "3. Explain the starter dataset layout.\n"
                            "4. Share documentation links for setup.\n"
                            "5. Confirm successful onboarding or note any blockers for escalation."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Flow: Issue Diagnosis Branch",
                        "body": (
                            "For bug/error reports:\n"
                            "1. Gather detailed description, error messages, and recent actions.\n"
                            "2. Attempt to reproduce the issue using known scenarios.\n"
                            "3. Provide step-by-step resolution if possible.\n"
                            "4. If unresolved, escalate to the technical team and provide a reference number."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Flow: Product Education Branch",
                        "body": (
                            "For feature guidance:\n"
                            "1. Explain model-selection tiers and their use cases.\n"
                            "2. Demonstrate how to submit feedback within the product.\n"
                            "3. Provide relevant documentation links.\n"
                            "4. Answer additional feature questions or escalate if needed."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Flow: Closing",
                        "body": (
                            "At the end of every call:\n"
                            "1. Summarise the solution or next steps.\n"
                            "2. Confirm the caller's understanding and satisfaction.\n"
                            "3. Provide the ticket reference number.\n"
                            "4. Record call notes for follow-up."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Scope & Redirects",
                        "body": (
                            "- Hardware repair, pricing, or advanced model customization: say 'I don't have that information, but I can arrange for a specialist to call you back.'\n"
                            "- Medical, legal, or unrelated queries: say 'I'm only able to assist with Smart Tire Analyser technical support.'"
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "Guardrails",
                        "body": (
                            "- Never provide unsupported troubleshooting steps or guess at technical solutions.\n"
                            "- Do not share internal or proprietary information.\n"
                            "- Never promise a fix unless confirmed by the technical team."
                        ),
                        "is_enabled": True,
                    },
                    {
                        "title": "FAQ",
                        "body": (
                            "User: What operating systems are supported?\n"
                            "Agent: Smart Tire Analyser supports Windows 10 or 11, macOS 12 and above, and Ubuntu 20.04 or newer.\n"
                            "User: How do I get started as a new user?\n"
                            "Agent: I can guide you through the QUICK_START steps and share the starter dataset layout and documentation links.\n"
                            "User: I am getting an error when importing data. What should I do?\n"
                            "Agent: Please share the exact error message and your recent steps, and I will help you troubleshoot or escalate if needed.\n"
                            "User: Can you help with hardware repairs or pricing?\n"
                            "Agent: I don't have that information, but I can arrange for a specialist to call you back.\n"
                            "User: Where can I find more detailed documentation?\n"
                            "Agent: I can share links to our official documentation for setup, features, and troubleshooting."
                        ),
                        "is_enabled": True,
                    },
                ],
                call_type="Incoming",
                transcriber={
                    "provider": "Soniox",
                    "silence_timeout_ms": 400,
                },
                model={
                    "model": "llama-3.3-70b-versatile",
                    "temperature": 0.7,
                },
                voice={
                    "provider": "cartesia",
                    "voice_id": "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
                },
                languages=["English (India)", "Hindi"],
                interruption={
                    "enabled": True,
                    "min_words": 2,
                },
                noise_reduction=True,
                web_search={
                    "enabled": True,
                    "provider": "DuckDuckGo",
                },
                call_ending={
                    "max_duration_sec": 600,
                    "enabled": True,
                    "condition": "End the call when the user says goodbye, thank you, or indicates they are done with the conversation",
                    "message": "Thank you for calling. Have a great day! Goodbye.",
                },
                user_idle={
                    "threshold_sec": 10,
                    "first_message": None,
                    "second_message": None,
                    "last_message": "I will leave you for now. Have a nice day!",
                },
            )

            self._agent = {
                "status": "active",
                "name": "Smart Tire Analyser Technical Support",
                "model": "llama-3.3-70b-versatile",
                "languages": ["English (India)", "Hindi"],
                "response": response if isinstance(response, dict) else {"id": str(response)},
            }
            logger.info("OmniDimension voice agent created successfully")
            return self._agent

        except Exception as e:
            logger.error("Failed to create OmniDimension agent: %s", e)
            return {"status": "error", "message": f"Failed to create voice agent: {e}"}

    async def get_agent_status(self) -> Dict[str, Any]:
        if self._agent:
            return self._agent
        return {"status": "not_initialized", "message": "Voice AI agent has not been created yet. Call POST /support/voice-agent/init first."}

    async def is_ready(self) -> bool:
        return self._initialized and self._agent is not None


omnidim_service = OmnidimService()
