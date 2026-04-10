# GLOBOGATE Lineup Screener

Standalone Mini-Tool fuer Dropout-Risikobewertung vor Klassenzuweisung.

## Stack

- Vanilla HTML/JS/CSS (kein Framework, kein Build-Step)
- Rein client-side Scoring mit Logistic Regression (v4-Modell)
- Kein Supabase, kein Login

## Modell

v4 Logistic Regression, trainiert auf 4.906 aufgeloesten Pipeline-Journeys.
Drei laenderspezifische Modelle: Philippines (22 Features), Uzbekistan (19 Features), Colombia (10 Features).
OHNE Recruiter/Schule — nur Kandidaten-Stammdaten die zum Zeitpunkt des Line-Ups bekannt sind.

## Dateien

```
app/
  index.html   — Komplettes UI
  app.js       — Scoring-Logik, UI-Controller, CSV-Handler
  style.css    — GLOBOGATE Design System
```

## Deploy

Auto-Deploy via GitHub → Netlify. `git push` auf `main` = live.

## Design System

- Primary: #15005A (Globoblue)
- Accent: #5F50FF (Buttons)
- Mint: #DBFFDC (Success)
- Font: DM Sans
