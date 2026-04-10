import type { Context } from "@netlify/functions"

const API_BASE = "https://api.globogate.de/external"
const API_TOKEN = process.env.GLOBOGATE_API_TOKEN || ""

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
}

export default async (req: Request, _context: Context) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS })
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
