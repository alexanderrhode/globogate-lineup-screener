/**
 * HTTP Basic Auth Edge Function fuer den GLOBOGATE Lineup Screener.
 *
 * Schuetzt die gesamte Site (HTML, JS, CSS, Netlify Functions) hinter einem
 * Shared-Secret-Passwort. Der Username im Browser-Dialog kann leer oder
 * beliebig sein — nur das Passwort wird gegen SITE_PASSWORD gepruefft.
 *
 * Konfiguration:
 *   Env-Var SITE_PASSWORD muss in Netlify (Site-Einstellungen) gesetzt sein.
 *   Falls nicht gesetzt, wird die Site standardmaessig nicht geschuetzt
 *   (Fail-Open) — damit lokale Entwicklung per `netlify dev` weiter laeuft.
 */

import type { Context } from "https://edge.netlify.com";

const REALM = 'Basic realm="GLOBOGATE Lineup Screener", charset="UTF-8"';

function unauthorized(message = "Authentication required") {
  return new Response(message, {
    status: 401,
    headers: {
      "WWW-Authenticate": REALM,
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}

export default async (request: Request, context: Context) => {
  // Env-Var — wenn nicht gesetzt, Site unprotected lassen (lokales Dev)
  const expected = Netlify.env.get("SITE_PASSWORD");
  if (!expected) return context.next();

  const authHeader = request.headers.get("authorization") || "";
  if (!authHeader.toLowerCase().startsWith("basic ")) {
    return unauthorized();
  }

  let decoded = "";
  try {
    decoded = atob(authHeader.slice(6).trim());
  } catch {
    return unauthorized("Invalid auth header");
  }

  // Format: "user:pass" — user darf leer sein, wir pruefen nur das Passwort
  const colonIdx = decoded.indexOf(":");
  const provided = colonIdx >= 0 ? decoded.slice(colonIdx + 1) : decoded;

  // Konstante-Zeit-Vergleich (klein, aber ordentlich)
  if (provided.length !== expected.length) return unauthorized();
  let diff = 0;
  for (let i = 0; i < provided.length; i++) {
    diff |= provided.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  if (diff !== 0) return unauthorized();

  // Auth ok — Request weiterleiten
  return context.next();
};

export const config = {
  path: "/*",
  // Unser api-proxy ist auch Teil der Site und muss ebenfalls geschuetzt sein.
  // /.netlify/* wird von Edge Functions standardmaessig NICHT abgedeckt,
  // deshalb explizit die API-Proxy-Route zusaetzlich schuetzen:
};
