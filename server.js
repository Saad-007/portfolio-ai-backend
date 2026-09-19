require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
const { GoogleGenAI } = require('@google/genai');

const app = express();

// --- CORS: restrict to your portfolio domain in production ---
// Set ALLOWED_ORIGIN in your .env, e.g. ALLOWED_ORIGIN=https://your-portfolio.vercel.app
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || '*';
app.use(cors({ origin: ALLOWED_ORIGIN }));
app.use(express.json());

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ---------------------------------------------------------
// EMBEDDING MODEL — boot + warm-up
// ---------------------------------------------------------
let extractor = null;
let modelReady = false;

(async () => {
  const { pipeline } = await import('@xenova/transformers');
  extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  // Warm-up: run one dummy embedding now so the FIRST real user
  // doesn't pay the cold-start cost (model weight loading, JIT, etc.)
  await extractor('warmup', { pooling: 'mean', normalize: true });

  modelReady = true;
  console.log("✅ Local Embedding Model Ready & Warmed Up.");
})();

// ---------------------------------------------------------
// SIMPLE IN-MEMORY CACHE (repeated / FAQ-style questions)
// Keyed by normalized message text -> { reply, timestamp }
// ---------------------------------------------------------
const responseCache = new Map();
const CACHE_MAX_SIZE = 200;
const CACHE_TTL_MS = 1000 * 60 * 60; // 1 hour

function getCacheKey(message) {
  return message.trim().toLowerCase().replace(/\s+/g, ' ');
}

function getFromCache(message) {
  const key = getCacheKey(message);
  const hit = responseCache.get(key);
  if (!hit) return null;
  if (Date.now() - hit.timestamp > CACHE_TTL_MS) {
    responseCache.delete(key);
    return null;
  }
  return hit.reply;
}

function setCache(message, reply) {
  const key = getCacheKey(message);
  if (responseCache.size >= CACHE_MAX_SIZE) {
    // Evict oldest entry (Maps preserve insertion order)
    const oldestKey = responseCache.keys().next().value;
    responseCache.delete(oldestKey);
  }
  responseCache.set(key, { reply, timestamp: Date.now() });
}

// ---------------------------------------------------------
// CONVERSATION MEMORY (multi-turn, per session)
// Frontend should send a stable `sessionId` (e.g. a random id
// generated once per browser tab and stored in memory/localStorage).
// If no sessionId is sent, the bot still works — just without
// cross-message memory (same as before).
// ---------------------------------------------------------
const sessions = new Map(); // sessionId -> [{ role, content }, ...]
const MAX_HISTORY_TURNS = 6; // last 6 exchanges per session
const SESSION_TTL_MS = 1000 * 60 * 30; // clear idle sessions after 30 min

function getHistory(sessionId) {
  if (!sessionId) return [];
  const entry = sessions.get(sessionId);
  if (!entry) return [];
  if (Date.now() - entry.lastSeen > SESSION_TTL_MS) {
    sessions.delete(sessionId);
    return [];
  }
  return entry.messages;
}

function pushHistory(sessionId, role, content) {
  if (!sessionId) return;
  const entry = sessions.get(sessionId) || { messages: [], lastSeen: Date.now() };
  entry.messages.push({ role, content });
  if (entry.messages.length > MAX_HISTORY_TURNS * 2) {
    entry.messages = entry.messages.slice(-MAX_HISTORY_TURNS * 2);
  }
  entry.lastSeen = Date.now();
  sessions.set(sessionId, entry);
}

// ---------------------------------------------------------
// SIMPLE RATE LIMITER (per IP, in-memory, no extra dependency)
// ---------------------------------------------------------
const RATE_LIMIT_WINDOW_MS = 60 * 1000; // 1 minute
const RATE_LIMIT_MAX = 15; // max requests per window per IP
const rateBuckets = new Map();

function isRateLimited(ip) {
  const now = Date.now();
  const bucket = rateBuckets.get(ip) || [];
  const recent = bucket.filter((t) => now - t < RATE_LIMIT_WINDOW_MS);
  recent.push(now);
  rateBuckets.set(ip, recent);
  return recent.length > RATE_LIMIT_MAX;
}

// ---------------------------------------------------------
// SYSTEM PROMPT
// ---------------------------------------------------------
const SYSTEM_PROMPT = `You are the AI portfolio assistant for Saad, an 8th-semester CS student and full-stack AI integration specialist.
You must answer strictly in the first person ("I", "my") acting as Saad.
Use ONLY the provided context to answer the question. Do not invent skills, metrics, or projects.

PERSONALITY & TONE:
Speak like a real person, not a corporate bot. Saad is a collaborative team player who genuinely enjoys contributing to and working alongside others on projects — he's not just a solo coder. He's a natural problem-solver: when something breaks or gets stuck, his instinct is to dig in and figure it out rather than give up or wait for someone else. He uses AI tools deliberately and effectively as part of his workflow — not as a crutch, but as a way to move faster and think more clearly. Outside of tech, he's a cricket and football fan and enjoys watching matches when he has time. Let this personality come through naturally in casual or personal questions — don't force it into every technical answer.

CRITICAL INSTRUCTION FOR PROJECT INQUIRIES:
When asked about your work, portfolio, recent projects, or what you are building, you MUST prioritize and mention these latest projects first before any others:
1. SuperSit (Commercial AI Desktop App — real-time posture detection)
2. Social Genius (AI-Powered iOS App)
3. ApplyMax (AI Career Copilot)
4. BodyMax (Multimodal AI Physique Assessment)
5. EduAIQuest (My ongoing Final Year Project)

If the retrieved context does not contain relevant information to answer the question, say so honestly
(e.g. "That's not something I've documented in my portfolio yet — feel free to email me directly and I can answer that personally.")
rather than guessing or inventing details.

If the user asks a question unrelated to software engineering, career, or the portfolio, politely decline and steer the conversation back to tech.`;

// ---------------------------------------------------------
// HELPER: run a promise with a timeout
// ---------------------------------------------------------
function withTimeout(promise, ms, timeoutMessage) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(timeoutMessage)), ms)),
  ]);
}

// ---------------------------------------------------------
// MAIN CHAT ENDPOINT
// ---------------------------------------------------------
app.post('/api/chat', async (req, res) => {
  const startTime = Date.now();
  try {
    const { message, sessionId } = req.body;

    if (!message || typeof message !== 'string' || !message.trim()) {
      return res.status(400).json({ error: "Message is required." });
    }
    if (message.length > 1000) {
      return res.status(400).json({ error: "Message is too long (max 1000 characters)." });
    }

    // Rate limiting
    const ip = req.headers['x-forwarded-for'] || req.socket.remoteAddress || 'unknown';
    if (isRateLimited(ip)) {
      return res.status(429).json({ error: "Too many requests — please slow down a bit." });
    }

    if (!modelReady || !extractor) {
      return res.status(503).json({ error: "AI model is still booting up, try again in a second." });
    }

    // Cache check (only for messages with no active session history,
    // since cached answers don't account for conversation context)
    const history = getHistory(sessionId);
    if (history.length === 0) {
      const cached = getFromCache(message);
      if (cached) {
        console.log(`⚡ Cache hit (${Date.now() - startTime}ms)`);
        pushHistory(sessionId, 'user', message);
        pushHistory(sessionId, 'model', cached);
        return res.json({ reply: cached, cached: true });
      }
    }

    // 1. Vectorize locally (0 latency, 0 cost, never goes down)
    const output = await extractor(message, { pooling: 'mean', normalize: true });
    const embedding = Array.from(output.data);

    // 2. Retrieve relevant context from Supabase
    const { data: documents, error } = await supabase.rpc('match_documents', {
      query_embedding: embedding,
      match_threshold: 0.25,
      match_count: 6,
    });

    if (error) throw error;

    // 3. Assemble context — explicitly flag when nothing relevant was found
    const context = documents && documents.length > 0
      ? documents.map((doc) => doc.content).join('\n\n')
      : "(No closely matching information was found in the knowledge base for this question.)";

    // 4. Build conversation-aware prompt
    const historyText = history
      .map((h) => `${h.role === 'user' ? 'User' : 'Saad'}: ${h.content}`)
      .join('\n');

    const fullPrompt = `Context information about Saad:\n${context}${
      historyText ? `\n\nConversation so far:\n${historyText}` : ''
    }\n\nUser Question: ${message}`;

    // 5. Generate response using Gemini, with a hard timeout so a slow
    //    API call never hangs the request indefinitely
    const response = await withTimeout(
      ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: fullPrompt,
        config: {
          systemInstruction: SYSTEM_PROMPT,
          temperature: 0.2,
        },
      }),
      12000,
      "The AI took too long to respond."
    );

    const reply = response.text;

    // Update memory + cache
    pushHistory(sessionId, 'user', message);
    pushHistory(sessionId, 'model', reply);
    if (history.length === 0) setCache(message, reply);

    console.log(`✅ Response in ${Date.now() - startTime}ms`);
    res.json({ reply });

  } catch (error) {
    console.error("RAG Pipeline Error:", error.message);
    const status = error.message?.includes('too long') ? 504 : 500;
    res.status(status).json({
      error: status === 504
        ? "The AI is taking longer than expected — please try again."
        : "Internal server error connecting to the AI system.",
    });
  }
});

// ---------------------------------------------------------
// HEALTH CHECK (useful for uptime monitors / quick debugging)
// ---------------------------------------------------------
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    modelReady,
    activeSessions: sessions.size,
    cachedResponses: responseCache.size,
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Local+Gemini RAG Server operational on port ${PORT}`));