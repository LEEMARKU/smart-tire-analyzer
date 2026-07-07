import { NextRequest, NextResponse } from "next/server"
import { existsSync, readFileSync } from "fs"
import { join } from "path"

export const dynamic = "force-dynamic"
export const runtime = "nodejs"

type ClientChatMessage = {
  role: "user" | "assistant"
  content: string
}

type LiveChatProvider = "gemini" | "openai-compatible" | "ollama" | "llama3"

type ChatResult = {
  reply: string
  model: string
  provider: LiveChatProvider
}

type GeminiResponse = {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string
      }>
    }
    finishReason?: string
  }>
  error?: {
    message?: string
  }
  promptFeedback?: {
    blockReason?: string
  }
}

type OpenAiCompatibleResponse = {
  choices?: Array<{
    message?: {
      content?: unknown
    }
    finish_reason?: string
  }>
  error?: {
    message?: string
  }
  model?: string
}

type OllamaResponse = {
  message?: {
    content?: string
  }
}

loadRepositoryEnvFallback()

const REQUEST_TIMEOUT_MS = Number(
  process.env.LIVE_CHAT_REQUEST_TIMEOUT_MS
    || process.env.OLLAMA_REQUEST_TIMEOUT_MS
    || "45000",
)

const TIRE_ONLY_SYSTEM_PROMPT = `You are Smart Tire Analyzer's live chat assistant. You are an expert in tires, tire maintenance, tire analysis, and automotive safety.

STRICT RULES - YOU MUST FOLLOW THESE:
1. ONLY answer questions related to tires, tire maintenance, tire analysis, tread depth, tire wear patterns, tire damage, tire pressure, tire rotation, tire replacement, tire brands, tire sizing, tire safety, wheel alignment, wheel balancing, and automotive topics directly related to tires.
2. If a user asks ANY question NOT related to tires, you MUST politely refuse and redirect them to tire-related topics.
3. Do NOT answer general knowledge questions, programming questions, medical questions, legal questions, or any non-tire topics.
4. Do NOT pretend to have capabilities outside of tire expertise.
5. For safety-critical tire issues (bulges, sidewall damage, punctures, tread < 2mm), ALWAYS recommend immediate professional inspection and cautious driving.
6. Keep answers concise, practical, and friendly.
7. Do not claim to access private account data or support tickets.

Examples of topics you CAN answer:
- Tire tread depth analysis and measurement
- Tire wear patterns (center wear, edge wear, cupping, feathering, etc.)
- Tire damage identification (cracks, cuts, bulges, punctures, sidewall damage)
- Tire pressure recommendations and maintenance
- Tire rotation schedules and patterns
- When to replace tires
- Tire sizing and specifications
- Wheel alignment and balancing related to tire wear
- Tire safety and driving advice based on tire condition
- Smart Tire Analyzer app features and usage

Examples of topics you MUST refuse:
- General conversation, weather, news, sports
- Programming, coding, technical non-tire questions
- Medical, legal, financial advice
- Other automotive topics not directly related to tires (engine, transmission, brakes, etc.)
- Personal questions, creative writing, homework help

Response format for non-tire questions: "I'm a tire specialist assistant and can only help with tire-related questions. Could I assist you with something about your tires, such as tread depth, wear patterns, pressure, or tire safety?"`;

const SYSTEM_PROMPT = TIRE_ONLY_SYSTEM_PROMPT;

const RETRYABLE_PROVIDER_STATUSES = new Set([401, 403, 429, 500, 503])

class LiveChatError extends Error {
  status: number
  detail?: string

  constructor(message: string, status = 502, detail?: string) {
    super(message)
    this.name = "LiveChatError"
    this.status = status
    this.detail = detail
  }
}

function loadRepositoryEnvFallback() {
  const envPath = join(process.cwd(), "..", ".env")

  if (!existsSync(envPath)) {
    return
  }

  const lines = readFileSync(envPath, "utf8").split(/\r?\n/)
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#")) {
      continue
    }

    const [rawKey, ...rawValueParts] = trimmed.split("=")
    const key = rawKey.trim()
    if (!key || rawValueParts.length === 0) {
      continue
    }

    const value = rawValueParts.join("=").trim().replace(/^['"]|['"]$/g, "")

    if (!process.env[key]) {
      process.env[key] = value
    }
  }
}

function normalizeMessages(value: unknown): ClientChatMessage[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .map((message) => {
      if (!message || typeof message !== "object") {
        return null
      }

      const role = "role" in message ? message.role : null
      const content = "content" in message ? message.content : null

      if ((role !== "user" && role !== "assistant") || typeof content !== "string") {
        return null
      }

      const trimmedContent = content.trim()
      if (!trimmedContent) {
        return null
      }

      return {
        role,
        content: trimmedContent.slice(0, 2000),
      }
    })
    .filter((message): message is ClientChatMessage => Boolean(message))
    .slice(-10)
}

function getEnvList(...values: Array<string | undefined>): string[] {
  const items = values.flatMap((value) => (
    value
      ? value
        .split(",")
        .flatMap((item) => {
          const trimmed = item.trim()
          return trimmed ? [trimmed] : []
        })
      : []
  ))

  return Array.from(new Set(items))
}

function getEnvValue(...values: Array<string | undefined>): string | null {
  for (const value of values) {
    const trimmed = value?.trim()
    if (trimmed) {
      return trimmed
    }
  }

  return null
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "")
}

function resolveProvider(): LiveChatProvider {
  const configuredProvider = process.env.LIVE_CHAT_PROVIDER?.trim().toLowerCase()

  if (configuredProvider === "gemini") {
    return "gemini"
  }

  if (
    configuredProvider === "openai"
    || configuredProvider === "openai-compatible"
    || configuredProvider === "groq"
    || configuredProvider === "openrouter"
  ) {
    return "openai-compatible"
  }

  if (configuredProvider === "llama3" || configuredProvider === "llama3.3") {
    return "llama3"
  }

  if (configuredProvider === "ollama") {
    return "ollama"
  }

  if (getEnvList(process.env.LIVE_CHAT_API_KEY, process.env.GEMINI_API_KEY, process.env.GEMINI_API_KEYS).length > 0) {
    return "gemini"
  }

  if (
    getEnvValue(process.env.LIVE_CHAT_API_KEY, process.env.GROQ_API_KEY, process.env.OPENROUTER_API_KEY, process.env.OPENAI_API_KEY)
    && resolveOpenAiCompatibleBaseUrl()
  ) {
    return "openai-compatible"
  }

  return "ollama"
}

function resolveOpenAiCompatibleBaseUrl(): string | null {
  const configuredBaseUrl = getEnvValue(
    process.env.LIVE_CHAT_API_BASE_URL,
    process.env.OPENAI_BASE_URL,
    process.env.OPENAI_API_BASE_URL,
  )

  if (configuredBaseUrl) {
    return trimTrailingSlash(configuredBaseUrl)
  }

  if (process.env.GROQ_API_KEY) {
    return "https://api.groq.com/openai/v1"
  }

  if (process.env.OPENROUTER_API_KEY) {
    return "https://openrouter.ai/api/v1"
  }

  if (process.env.OPENAI_API_KEY) {
    return "https://api.openai.com/v1"
  }

  return null
}

async function postJson(url: string, headers: HeadersInit, payload: unknown): Promise<Response> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    return await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
      cache: "no-store",
    })
  } finally {
    clearTimeout(timeout)
  }
}

async function readJsonResponse<T>(response: Response): Promise<{ data: T | null; text: string }> {
  const text = await response.text()

  if (!text) {
    return { data: null, text: "" }
  }

  try {
    return { data: JSON.parse(text) as T, text }
  } catch {
    return { data: null, text }
  }
}

function buildErrorDetail(data: { error?: { message?: string } } | null, fallbackText: string): string {
  return (data?.error?.message || fallbackText || "No provider detail returned").slice(0, 500)
}

function normalizeGeminiModelPath(model: string): string {
  const modelPath = model.startsWith("models/") ? model : `models/${model}`
  return modelPath.split("/").map(encodeURIComponent).join("/")
}

function extractOpenAiCompatibleText(content: unknown): string {
  if (typeof content === "string") {
    return content
  }

  if (!Array.isArray(content)) {
    return ""
  }

  return content
    .map((part) => {
      if (typeof part === "string") {
        return part
      }

      if (!part || typeof part !== "object") {
        return ""
      }

      const text = "text" in part ? part.text : null
      return typeof text === "string" ? text : ""
    })
    .join("")
}

async function callGemini(messages: ClientChatMessage[]): Promise<ChatResult> {
  const apiKeys = getEnvList(
    process.env.LIVE_CHAT_API_KEY,
    process.env.GEMINI_API_KEY,
    process.env.GEMINI_API_KEYS,
  )
  const model = process.env.LIVE_CHAT_MODEL || process.env.GEMINI_MODEL || "gemini-2.5-flash"
  const baseUrl = trimTrailingSlash(process.env.LIVE_CHAT_API_BASE_URL || process.env.GEMINI_API_BASE_URL || "https://generativelanguage.googleapis.com/v1beta")

  if (apiKeys.length === 0) {
    throw new LiveChatError(
      "Live Chat is set to Gemini, but no Gemini API key is configured.",
      503,
      "Set GEMINI_API_KEY, GEMINI_API_KEYS, or LIVE_CHAT_API_KEY.",
    )
  }

  const payload = {
    systemInstruction: {
      parts: [{ text: SYSTEM_PROMPT }],
    },
    contents: messages.map((message) => ({
      role: message.role === "assistant" ? "model" : "user",
      parts: [{ text: message.content }],
    })),
    generationConfig: {
      temperature: 0.35,
      topP: 0.9,
      maxOutputTokens: 512,
    },
  }

  async function tryApiKey(index: number): Promise<ChatResult> {
    const apiKey = apiKeys[index]
    if (!apiKey) {
      throw new LiveChatError("Gemini is unavailable right now.", 503)
    }

    const response = await postJson(
      `${baseUrl}/${normalizeGeminiModelPath(model)}:generateContent`,
      {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey,
      },
      payload,
    )
    const { data, text } = await readJsonResponse<GeminiResponse>(response)

    if (!response.ok) {
      const error = new LiveChatError(
        `Gemini returned ${response.status} for ${model}.`,
        502,
        buildErrorDetail(data, text),
      )

      if (RETRYABLE_PROVIDER_STATUSES.has(response.status) && index < apiKeys.length - 1) {
        return tryApiKey(index + 1)
      }

      throw error
    }

    const reply = data?.candidates?.[0]?.content?.parts
      ?.map((part) => part.text || "")
      .join("")
      .trim()

    if (!reply) {
      const reason = data?.promptFeedback?.blockReason || data?.candidates?.[0]?.finishReason
      throw new LiveChatError(
        "Gemini did not return a chat response.",
        502,
        reason ? `Finish or block reason: ${reason}` : undefined,
      )
    }

    return {
      reply,
      model,
      provider: "gemini",
    }
  }

  return tryApiKey(0)
}

async function callOpenAiCompatible(messages: ClientChatMessage[]): Promise<ChatResult> {
  const baseUrl = resolveOpenAiCompatibleBaseUrl()
  const apiKey = getEnvValue(
    process.env.LIVE_CHAT_API_KEY,
    process.env.GROQ_API_KEY,
    process.env.OPENROUTER_API_KEY,
    process.env.OPENAI_API_KEY,
  )
  const model = process.env.LIVE_CHAT_MODEL || (process.env.GROQ_API_KEY ? "llama3-8b-8192" : null)

  if (!baseUrl || !apiKey || !model) {
    throw new LiveChatError(
      "Live Chat is set to a hosted LLM, but the API URL, API key, or model is missing.",
      503,
      "Set LIVE_CHAT_API_BASE_URL, LIVE_CHAT_API_KEY, and LIVE_CHAT_MODEL.",
    )
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  }

  const openRouterReferer = process.env.OPENROUTER_SITE_URL || process.env.NEXT_PUBLIC_SITE_URL
  if (openRouterReferer) {
    headers["HTTP-Referer"] = openRouterReferer
  }

  const appTitle = process.env.OPENROUTER_APP_TITLE || "Smart Tire Analyzer"
  headers["X-Title"] = appTitle

  const response = await postJson(
    `${baseUrl}/chat/completions`,
    headers,
    {
      model,
      stream: false,
      temperature: 0.35,
      top_p: 0.9,
      max_completion_tokens: 512,
      messages: [
        {
          role: "system",
          content: SYSTEM_PROMPT,
        },
        ...messages,
      ],
    },
  )
  const { data, text } = await readJsonResponse<OpenAiCompatibleResponse>(response)

  if (!response.ok) {
    throw new LiveChatError(
      `Hosted LLM returned ${response.status} for ${model}.`,
      502,
      buildErrorDetail(data, text),
    )
  }

  const reply = extractOpenAiCompatibleText(data?.choices?.[0]?.message?.content).trim()

  if (!reply) {
    throw new LiveChatError(
      "The hosted LLM did not return a chat response.",
      502,
      data?.choices?.[0]?.finish_reason ? `Finish reason: ${data.choices[0].finish_reason}` : undefined,
    )
  }

  return {
    reply,
    model: data?.model || model,
    provider: "openai-compatible",
  }
}

async function callOllama(messages: ClientChatMessage[]): Promise<ChatResult> {
  const configuredOllamaBaseUrl = process.env.OLLAMA_BASE_URL || process.env.OLLAMA_HOST
  const baseUrls = Array.from(
    new Set(
      [
        configuredOllamaBaseUrl,
        "http://127.0.0.1:11434",
        "http://host.docker.internal:11434",
      ]
        .filter((url): url is string => Boolean(url))
        .map(trimTrailingSlash),
    ),
  )
  const model = process.env.OLLAMA_MODEL || process.env.LIVE_CHAT_MODEL || "llama3:8b"
  const payload = {
    model,
    stream: false,
    messages: [
      {
        role: "system",
        content: SYSTEM_PROMPT,
      },
      ...messages,
    ],
    options: {
      temperature: 0.35,
      top_p: 0.9,
    },
  }

  async function tryBaseUrl(index: number, lastConnectionError = ""): Promise<ChatResult> {
    const baseUrl = baseUrls[index]
    if (!baseUrl) {
      throw new LiveChatError(
        "Unable to reach local Ollama.",
        503,
        lastConnectionError || "Set LIVE_CHAT_PROVIDER=gemini or LIVE_CHAT_PROVIDER=openai-compatible to avoid Ollama.",
      )
    }

    try {
      const response = await postJson(
        `${baseUrl}/api/chat`,
        {
          "Content-Type": "application/json",
        },
        payload,
      )
      const { data, text } = await readJsonResponse<OllamaResponse>(response)

      if (!response.ok) {
        throw new LiveChatError(
          `Ollama returned ${response.status}. Make sure ${model} is available locally.`,
          502,
          text.slice(0, 500),
        )
      }

      const reply = data?.message?.content?.trim()

      if (!reply) {
        throw new LiveChatError("Ollama did not return a chat response.", 502)
      }

      return {
        reply,
        model,
        provider: "ollama",
      }
    } catch (error) {
      return tryBaseUrl(
        index + 1,
        error instanceof Error ? error.message : "Unknown Ollama connection error",
      )
    }
  }

  return tryBaseUrl(0)
}

async function callLlama33(messages: ClientChatMessage[]): Promise<ChatResult> {
  const baseUrl = trimTrailingSlash(
    process.env.LLAMA3_BASE_URL
    || process.env.OLLAMA_BASE_URL
    || process.env.OLLAMA_HOST
    || "http://127.0.0.1:11434",
  )
  const model = process.env.LLAMA3_MODEL || "llama3.3:latest"

  const payload = {
    model,
    stream: false,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      ...messages,
    ],
    options: {
      temperature: 0.3,
      top_p: 0.9,
    },
  }

  const response = await postJson(
    `${baseUrl}/api/chat`,
    { "Content-Type": "application/json" },
    payload,
  )
  const { data, text } = await readJsonResponse<OllamaResponse>(response)

  if (!response.ok) {
    throw new LiveChatError(
      `Llama 3.3 returned ${response.status}. Make sure ${model} is available locally.`,
      502,
      text.slice(0, 500),
    )
  }

  const reply = data?.message?.content?.trim()

  if (!reply) {
    throw new LiveChatError("Llama 3.3 did not return a chat response.", 502)
  }

  return { reply, model, provider: "llama3" }
}

async function getChatCompletion(messages: ClientChatMessage[]): Promise<ChatResult> {
  const provider = resolveProvider()

  if (provider === "gemini") {
    return callGemini(messages)
  }

  if (provider === "openai-compatible") {
    return callOpenAiCompatible(messages)
  }

  if (provider === "llama3") {
    return callLlama33(messages)
  }

  return callOllama(messages)
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null) as { messages?: unknown } | null
  const messages = normalizeMessages(body?.messages)

  if (messages.length === 0) {
    return NextResponse.json(
      { error: "Please send at least one chat message." },
      { status: 400 },
    )
  }

  try {
    const result = await getChatCompletion(messages)
    return NextResponse.json(result)
  } catch (error) {
    if (error instanceof LiveChatError) {
      return NextResponse.json(
        {
          error: error.message,
          detail: error.detail,
        },
        { status: error.status },
      )
    }

    const detail = error instanceof Error ? error.message : "Unknown live chat error"
    return NextResponse.json(
      {
        error: "The live chat assistant is unavailable right now.",
        detail,
      },
      { status: 503 },
    )
  }
}
