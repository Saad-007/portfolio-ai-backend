require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(cors()); 
app.use(express.json());

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Boot up the Local AI Model 
let extractor;
(async () => {
    const { pipeline } = await import('@xenova/transformers');
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log("✅ Local Embedding Model Ready.");
})();

const SYSTEM_PROMPT = `You are the AI portfolio assistant for Saad, a 7th-semester CS student and MERN stack developer.
You must answer strictly in the first person ("I", "my") acting as Saad.
Use ONLY the provided context to answer the question. Do not invent skills or projects.
If the user asks a question unrelated to software engineering, career, or the portfolio, politely decline and steer the conversation back to tech.`;

app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "Message is required." });

    if (!extractor) return res.status(503).json({ error: "AI model is still booting up, try again in a second." });

    // 1. Vectorize locally (0 latency, 0 cost, never goes down)
    const output = await extractor(message, { pooling: 'mean', normalize: true });
    const embedding = Array.from(output.data);

    // 2. Retrieve relevant context from Supabase
    const { data: documents, error } = await supabase.rpc('match_documents', {
      query_embedding: embedding,
      match_threshold: 0.30, 
      match_count: 3,        
    });

    if (error) throw error;

    // 3. Assemble context
    const context = documents.map(doc => doc.content).join('\n\n');

    // 4. Generate response using Gemini
    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: `Context information about Saad:\n${context}\n\nUser Question: ${message}`,
        config: {
            systemInstruction: SYSTEM_PROMPT,
            temperature: 0.2
        }
    });

    res.json({ reply: response.text });

  } catch (error) {
    console.error("RAG Pipeline Error:", error);
    res.status(500).json({ error: "Internal server error connecting to the AI system." });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Local+Gemini RAG Server operational on port ${PORT}`));