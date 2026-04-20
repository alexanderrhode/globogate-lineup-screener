import type { Context } from "@netlify/functions"

const API_BASE = "https://api.globogate.de/external"
const API_TOKEN = process.env.GLOBOGATE_API_TOKEN || ""
const SITE_PASSWORD = process.env.SITE_PASSWORD || ""

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
}

const AUTH_HEADERS = {
  ...CORS_HEADERS,
  "WWW-Authenticate": 'Basic realm="GLOBOGATE Lineup Screener", charset="UTF-8"',
  "Content-Type": "text/plain; charset=utf-8",
}

function checkAuth(req: Request): boolean {
  // Wenn kein Passwort konfiguriert ist, offen lassen (lokales Dev)
  if (!SITE_PASSWORD) return true
  const auth = req.headers.get("authorization") || ""
  if (!auth.toLowerCase().startsWith("basic ")) return false
  let decoded = ""
  try { decoded = atob(auth.slice(6).trim()) } catch { return false }
  const colonIdx = decoded.indexOf(":")
  const provided = colonIdx >= 0 ? decoded.slice(colonIdx + 1) : decoded
  if (provided.length !== SITE_PASSWORD.length) return false
  let diff = 0
  for (let i = 0; i < provided.length; i++) {
    diff |= provided.charCodeAt(i) ^ SITE_PASSWORD.charCodeAt(i)
  }
  return diff === 0
}

export default async (req: Request, _context: Context) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS })
  }

  if (!checkAuth(req)) {
    return new Response("Authentication required", { status: 401, headers: AUTH_HEADERS })
  }

  const url = new URL(req.url)
  const endpoint = url.searchParams.get("endpoint")

  if (!endpoint) {
    return new Response(JSON.stringify({ error: "Missing endpoint parameter" }), {
      status: 400,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    })
  }

  if (!API_TOKEN) {
    return new Response(JSON.stringify({ error: "API token not configured" }), {
      status: 500,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    })
  }

  const allowed = ["/persons"]
  if (!allowed.some((a) => endpoint.startsWith(a))) {
    return new Response(JSON.stringify({ error: "Endpoint not allowed" }), {
      status: 403,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    })
  }

  try {
    const apiUrl = `${API_BASE}${endpoint}`
    const res = await fetch(apiUrl, {
      headers: { Authorization: `Bearer ${API_TOKEN}` },
    })

    const data = await res.text()

    return new Response(data, {
      status: res.status,
      headers: {
        ...CORS_HEADERS,
        "Content-Type": "application/json",
        "Cache-Control": "public, max-age=300",
      },
    })
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 502,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    })
  }
}

export const config = {
  path: "/.netlify/functions/api-proxy",
}
