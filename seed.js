require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

// Your core facts
const coreData = [
  {
    content: "I am a Computer Science student currently in my 7th semester, specializing in web development with a core interest in AI and blockchain.",
    metadata: { category: "bio" }
  },
  {
    content: "My 5-year aspiration is to have absolute command in Data Structures and Algorithms (DSA) and to become an AI Engineer.",
    metadata: { category: "goals" }
  },
  {
    content: "My professional experience includes working at the Digital Empowerment Network (DEN). Additionally, at Ziauddin Hospital, I built the frontend for a patient-type page portal.",
    metadata: { category: "experience" }
  },
  {
    content: "I built TeamSync, a collaborative whiteboard and video meeting platform. It uses React, Socket.io, and Firebase for the backend. I integrated Whisper and Ollama to generate AI meeting transcripts, diagrams, and summaries.",
    metadata: { category: "project", name: "TeamSync" }
  },
  {
    content: "I developed ResumeAI – an AI-Powered Resume Builder & Analyzer. It works in two ways: First, users can provide a simple prompt, and the AI will dynamically generate a professional, beautifully formatted resume. Second, users can use the analyzer feature to check their existing resumes, receiving detailed AI feedback and suggestions for improvement.",
    metadata: { category: "project", name: "ResumeAI" }
  },
  {
    content: "I created ShopPlus, a full-stack e-commerce architecture with seamless checkout flows and dynamic inventory management using React, Node.js, and Express.",
    metadata: { category: "project", name: "ShopPlus" }
  },
  {
    content: "I am the founder of Syntaq Systems, a web agency focused on high-end web development, strategic video editing, and AI automation workflows.",
    metadata: { category: "agency", name: "Syntaq Systems" }
  },
  
  // --- NEW MEMORY BLOCKS INSTALLED HERE ---
  {
    content: "If someone asks about my most interesting, favorite, or best project, I always talk about TeamSync. It was a fascinating engineering challenge because I had to integrate real-time collaborative whiteboards and video meetings with complex AI tools like Whisper and Ollama for automated meeting transcriptions and diagrams.",
    metadata: { category: "favorites", type: "interesting_project" }
  },
  {
    content: "When asked generally about my projects or portfolio, I highlight my top three core pieces: TeamSync (an AI-powered collaborative meeting platform), ResumeAI (an AI resume builder and analyzer), and ShopPlus (a full-stack MERN e-commerce architecture).",
    metadata: { category: "portfolio_summary", type: "top_projects" }
  },
  // ----------------------------------------

  {
    content: "If someone wants to contact me, hire me, or discuss a project, my direct email is saadsafeer223@gmail.com.",
    metadata: { category: "contact", type: "email" }
  },
  {
    content: "My professional LinkedIn profile can be found at www.linkedin.com/in/saad-safeer-11b9b227a. You can connect with me there for professional networking.",
    metadata: { category: "contact", type: "linkedin" }
  }
];

async function syncGitHubAndSeed() {
  console.log("1. Wiping old memory...");
  // Clears the table so we don't get duplicate answers
  await supabase.from('portfolio_knowledge').delete().neq('id', 0);

  console.log("2. Fetching live GitHub repositories for Saad-007...");
  try {
    const response = await fetch('https://api.github.com/users/Saad-007/repos');
    const repos = await response.json();
    
    // Add a fact about the total count
    coreData.push({
      content: `I currently have exactly ${repos.length} public repositories hosted on my GitHub profile.`,
      metadata: { category: "github_stats" }
    });

    // Add a fact for every single repository
    repos.forEach(repo => {
      if (!repo.fork) { // Skip projects you just copied, focus on your originals
        const desc = repo.description ? ` It is described as: ${repo.description}.` : '';
        const lang = repo.language ? ` The primary technology used is ${repo.language}.` : '';
        
        coreData.push({
          content: `On GitHub, one of my projects is named ${repo.name}.${desc}${lang}`,
          metadata: { category: "github_repo", name: repo.name }
        });
      }
    });
    console.log(`✅ Found and prepped ${repos.length} GitHub projects.`);
  } catch (error) {
    console.log("⚠️ Could not fetch GitHub data, proceeding with core data only.");
  }

  console.log("3. Booting up Local AI Model...");
  const { pipeline } = await import('@xenova/transformers');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  console.log("4. Vectorizing all knowledge...");
  for (const doc of coreData) {
    try {
      const output = await extractor(doc.content, { pooling: 'mean', normalize: true });
      const embedding = Array.from(output.data); 

      const { error } = await supabase
        .from('portfolio_knowledge')
        .insert({
          content: doc.content,
          metadata: doc.metadata,
          embedding: embedding,
        });

      if (error) throw error;
      console.log(`✅ Learned: ${doc.metadata.name || doc.metadata.category}`);
      
    } catch (error) {
      console.error(`❌ Failed to learn ${doc.metadata.category}:`, error);
    }
  }
  console.log("🧠 Neural Sync Complete. Your AI now knows your entire GitHub history.");
}

syncGitHubAndSeed();