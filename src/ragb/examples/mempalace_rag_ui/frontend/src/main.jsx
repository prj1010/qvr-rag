import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion } from "motion/react";
import {
  Archive,
  ArrowUp,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleHelp,
  FileText,
  FolderOpen,
  Gauge,
  Layers3,
  LibraryBig,
  LoaderCircle,
  MessageSquareText,
  Network,
  PanelLeft,
  Plus,
  Search,
  Settings2,
  Sparkles,
  UploadCloud,
  Volume2,
  X,
  Zap,
  Square,
} from "lucide-react";
import "./styles.css";

const MODELS = {
  Groq: "llama-3.3-70b-versatile",
  "NVIDIA NIM": "nvidia/nemotron-3-super-120b-a12b",
};

const ACCEPTED_FILES = ".pdf,.docx,.csv,.txt,.md,.markdown,.mdx";

async function apiRequest(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || "The API request failed.");
  }
  return payload;
}

function App() {
  const [provider, setProvider] = useState("Groq");
  const [model, setModel] = useState(MODELS.Groq);
  const [brainName, setBrainName] = useState("mempalace-quivr");
  const [wing, setWing] = useState("quivr-demo");
  const [palacePath, setPalacePath] = useState("~/.mempalace/palace");
  const [memoryCount, setMemoryCount] = useState(5);
  const [files, setFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [memoryContext, setMemoryContext] = useState("");
  const [indexStatus, setIndexStatus] = useState("No documents indexed yet.");
  const [status, setStatus] = useState({ tone: "neutral", text: "Ready when you are." });
  const [isIndexing, setIsIndexing] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [isRecalling, setIsRecalling] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    setModel(MODELS[provider]);
  }, [provider]);

  function addFiles(incoming) {
    const next = Array.from(incoming || []).filter((file) => {
      return ACCEPTED_FILES.split(",").some((extension) =>
        file.name.toLowerCase().endsWith(extension),
      );
    });
    setFiles((current) => {
      const merged = [...current, ...next];
      return merged.filter(
        (file, index, all) =>
          all.findIndex((candidate) => candidate.name === file.name && candidate.size === file.size) === index,
      );
    });
  }

  async function handleIndex() {
    if (!files.length) {
      setIndexStatus("Choose at least one document first.");
      return;
    }
    setIsIndexing(true);
    setStatus({ tone: "working", text: "Reading documents and building embeddings…" });
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    body.append("provider", provider);
    body.append("model_name", model);
    body.append("brain_name", brainName);
    try {
      const payload = await apiRequest("/api/index", { method: "POST", body });
      setIndexStatus(payload.message || "Documents indexed.");
      setStatus({ tone: "success", text: `${payload.chunks || "Your"} chunks are ready for questions.` });
    } catch (error) {
      setIndexStatus(`Indexing error: ${error.message}`);
      setStatus({ tone: "error", text: "Indexing needs attention." });
    } finally {
      setIsIndexing(false);
    }
  }

  async function handleAsk(event) {
    event?.preventDefault();
    if (!question.trim()) return;
    setIsAsking(true);
    setStatus({ tone: "working", text: "Searching documents and recalling memory…" });
    try {
      const payload = await apiRequest("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history: messages,
          wing,
          palace_path: palacePath,
          n_results: memoryCount,
        }),
      });
      setMessages(payload.history || messages);
      setMemoryContext(payload.context || "");
      setQuestion("");
      setStatus({ tone: payload.ok ? "success" : "error", text: payload.status || "Complete." });
    } catch (error) {
      setStatus({ tone: "error", text: `Request failed: ${error.message}` });
    } finally {
      setIsAsking(false);
    }
  }

  async function handleRecall() {
    if (!question.trim()) {
      setMemoryContext("Enter a question to search MemPalace memory.");
      return;
    }
    setIsRecalling(true);
    try {
      const payload = await apiRequest("/api/recall", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, wing, palace_path: palacePath, n_results: memoryCount }),
      });
      setMemoryContext(payload.context || "No memories found.");
      setStatus({ tone: payload.ok ? "success" : "error", text: "Memory recall complete." });
    } catch (error) {
      setStatus({ tone: "error", text: `Memory recall failed: ${error.message}` });
    } finally {
      setIsRecalling(false);
    }
  }

  function clearConversation() {
    setMessages([]);
    setMemoryContext("");
    setQuestion("");
    setStatus({ tone: "neutral", text: "Conversation cleared." });
  }

  return (
    <div className="app-shell">
      <div className="noise" />
      <div className="aurora aurora-one" />
      <div className="aurora aurora-two" />

      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><BrainCircuit size={21} /></div>
          <div>
            <div className="brand-name">mempalace <span>×</span> quivr</div>
            <div className="brand-caption">Long-term memory intelligence</div>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="live-pill"><span className="live-dot" /> Local workspace</div>
          <button className="icon-button" type="button" aria-label="Help"><CircleHelp size={18} /></button>
          <button className="avatar" type="button" aria-label="Profile">GS</button>
        </div>
      </header>

      <main className="page-container">
        <section className="hero-section">
          <motion.div
            className="eyebrow"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
          >
            <Sparkles size={14} /> Context-aware RAG workspace
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.06 }}
          >
            Your knowledge, <span>remembered.</span>
          </motion.h1>
          <motion.p
            className="hero-copy"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.12 }}
          >
            Ground every answer in your documents and the context that matters.
            Quivr retrieves the signal; MemPalace keeps the thread.
          </motion.p>
        </section>

        <section className="stats-grid" aria-label="Workspace overview">
          <StatCard icon={<LibraryBig size={18} />} label="Memory layer" value="MemPalace" detail="Persistent recall" accent="violet" />
          <StatCard icon={<Network size={18} />} label="Retrieval layer" value="Quivr RAG" detail="Semantic search" accent="cyan" />
          <StatCard icon={<Zap size={18} />} label="Active model" value={provider} detail={model} accent="amber" />
          <StatCard icon={<Gauge size={18} />} label="Workspace state" value={files.length ? `${files.length} file${files.length === 1 ? "" : "s"}` : "Empty"} detail={indexStatus} accent="green" />
        </section>

        <div className="workspace-grid">
          <aside className="settings-column">
            <div className="panel-heading">
              <div>
                <div className="section-kicker"><Settings2 size={14} /> Workspace setup</div>
                <h2>Configure your mind</h2>
              </div>
              <PanelLeft size={18} className="muted-icon" />
            </div>

            <div className="form-section">
              <label htmlFor="provider">LLM provider</label>
              <div className="select-wrap">
                <select id="provider" value={provider} onChange={(event) => setProvider(event.target.value)}>
                  <option>Groq</option>
                  <option>NVIDIA NIM</option>
                </select>
                <ChevronDown size={15} />
              </div>
            </div>
            <div className="form-section">
              <label htmlFor="model">Model</label>
              <input id="model" value={model} onChange={(event) => setModel(event.target.value)} />
            </div>
            <div className="form-section">
              <label htmlFor="brain-name">Brain name</label>
              <input id="brain-name" value={brainName} onChange={(event) => setBrainName(event.target.value)} />
            </div>

            <div className="divider" />

            <div className="panel-heading compact">
              <div>
                <div className="section-kicker"><Archive size={14} /> Memory settings</div>
                <h3>Keep the context</h3>
              </div>
            </div>
            <div className="form-section">
              <label htmlFor="wing">MemPalace wing</label>
              <input id="wing" value={wing} onChange={(event) => setWing(event.target.value)} />
            </div>
            <div className="form-section">
              <label htmlFor="palace-path">Palace path</label>
              <input id="palace-path" value={palacePath} onChange={(event) => setPalacePath(event.target.value)} />
            </div>
            <div className="form-section range-section">
              <div className="range-label"><label htmlFor="memory-count">Memories to recall</label><span>{memoryCount}</span></div>
              <input id="memory-count" type="range" min="1" max="10" value={memoryCount} onChange={(event) => setMemoryCount(Number(event.target.value))} />
            </div>
          </aside>

          <section className="main-column">
            <div className="panel upload-panel">
              <div className="panel-heading">
                <div>
                  <div className="section-kicker"><Layers3 size={14} /> Knowledge base</div>
                  <h2>Feed the retrieval engine</h2>
                </div>
                <div className="source-count">{files.length} selected</div>
              </div>
              <div
                className={`dropzone ${dragActive ? "drag-active" : ""}`}
                onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setDragActive(false)}
                onDrop={(event) => { event.preventDefault(); setDragActive(false); addFiles(event.dataTransfer.files); }}
                onClick={() => inputRef.current?.click()}
              >
                <input ref={inputRef} type="file" multiple accept={ACCEPTED_FILES} onChange={(event) => addFiles(event.target.files)} />
                <div className="upload-orb"><UploadCloud size={24} /></div>
                <div className="dropzone-title">Drop your sources here</div>
                <div className="dropzone-subtitle">PDF, DOCX, CSV, TXT, or Markdown · up to your imagination</div>
                <button className="secondary-button" type="button" onClick={(event) => { event.stopPropagation(); inputRef.current?.click(); }}><Plus size={16} /> Browse files</button>
              </div>
              <AnimatePresence initial={false}>
                {files.length > 0 && (
                  <motion.div className="file-list" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}>
                    {files.map((file) => (
                      <div className="file-chip" key={`${file.name}-${file.size}`}>
                        <FileText size={15} /><span>{file.name}</span><button type="button" onClick={() => setFiles((current) => current.filter((candidate) => candidate !== file))} aria-label={`Remove ${file.name}`}><X size={14} /></button>
                      </div>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
              <div className="upload-footer">
                <div className="index-copy"><span className={`status-dot ${isIndexing ? "pulse" : ""}`} />{indexStatus}</div>
                <motion.button className="primary-button" type="button" onClick={handleIndex} disabled={isIndexing} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                  {isIndexing ? <LoaderCircle className="spin" size={17} /> : <Sparkles size={17} />} {isIndexing ? "Indexing…" : "Index documents"}
                </motion.button>
              </div>
            </div>

            <div className="chat-panel panel">
              <div className="chat-heading">
                <div className="chat-title"><div className="chat-icon"><MessageSquareText size={18} /></div><div><div className="section-kicker">Conversation</div><h2>Ask your knowledge</h2></div></div>
                <button className="text-button" type="button" onClick={clearConversation}>Clear thread</button>
              </div>
              <div className={`status-banner ${status.tone}`}><span className="status-banner-icon">{status.tone === "success" ? <Check size={14} /> : status.tone === "working" ? <LoaderCircle className="spin" size={14} /> : <Sparkles size={14} />}</span>{status.text}</div>
              <div className="conversation-window">
                {messages.length === 0 ? (
                  <div className="empty-conversation">
                    <motion.div className="empty-orbit" animate={{ rotate: 360 }} transition={{ duration: 18, repeat: Infinity, ease: "linear" }}><div className="empty-orb"><BrainCircuit size={28} /></div></motion.div>
                    <h3>Start a conversation with your data</h3>
                    <p>Index a document, then ask a question. Every response is enriched with relevant memory from your palace.</p>
                    <div className="prompt-suggestions"><button type="button" onClick={() => setQuestion("What are the key ideas in these documents?")}>Summarize the key ideas <ArrowUp size={13} /></button><button type="button" onClick={() => setQuestion("What should I remember from this?")}>What should I remember? <ArrowUp size={13} /></button></div>
                  </div>
                ) : (
                  <div className="message-list">
                    {messages.map(([prompt, answer], index) => <Message key={`${prompt}-${index}`} prompt={prompt} answer={answer} />)}
                  </div>
                )}
              </div>
              <form className="question-composer" onSubmit={handleAsk}>
                <div className="composer-label"><Search size={14} /> Ask about your sources</div>
                <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What would you like to understand?" rows="2" />
                <div className="composer-footer"><span>Press Enter to send · Shift + Enter for a new line</span><div className="composer-actions"><button className="secondary-button small" type="button" onClick={handleRecall} disabled={isRecalling}>{isRecalling ? <LoaderCircle className="spin" size={15} /> : <Archive size={15} />} Recall memory</button><motion.button className="primary-button send-button" type="submit" disabled={isAsking || !question.trim()} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>{isAsking ? <LoaderCircle className="spin" size={17} /> : <ArrowUp size={17} />} {isAsking ? "Thinking…" : "Ask"}</motion.button></div></div>
              </form>
            </div>

            <div className="memory-panel panel">
              <div className="panel-heading compact"><div><div className="section-kicker"><Archive size={14} /> Retrieval trace</div><h3>MemPalace context</h3></div><div className="context-badge">{memoryContext ? "Context found" : "Waiting for recall"}</div></div>
              <div className={`context-box ${memoryContext ? "has-context" : ""}`}>{memoryContext || "Memory retrieved for your next question will appear here."}</div>
            </div>
          </section>
        </div>
      </main>

      <footer className="footer"><span>Built for thoughtful retrieval</span><span className="footer-divider" /><span>Quivr RAG <b>·</b> MemPalace memory</span></footer>
    </div>
  );
}

function StatCard({ icon, label, value, detail, accent }) {
  return <motion.div className={`stat-card ${accent}`} whileHover={{ y: -3 }} transition={{ type: "spring", stiffness: 300, damping: 20 }}><div className="stat-icon">{icon}</div><div className="stat-text"><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></div></motion.div>;
}

function Message({ prompt, answer }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [audioError, setAudioError] = useState("");
  const audioRef = useRef(null);
  const audioUrlRef = useRef("");

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, []);

  function releaseAudioUrl() {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = "";
    }
  }

  async function handlePlay() {
    if (isPlaying && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      return;
    }

    if (!answer) return;

    setIsLoadingAudio(true);
    setAudioError("");
    try {
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: answer, voice: "af_bella" }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Failed to generate audio.");
      }

      const blob = await response.blob();
      if (!blob.size || !audioRef.current) {
        throw new Error("The TTS service returned an empty audio file.");
      }

      releaseAudioUrl();
      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;
      audioRef.current.src = url;
      audioRef.current.onended = () => setIsPlaying(false);
      audioRef.current.onerror = () => {
        setIsPlaying(false);
        setAudioError("The browser could not play the generated audio.");
      };
      await audioRef.current.play();
      setIsPlaying(true);
    } catch (error) {
      console.error(error);
      setAudioError(error instanceof Error ? error.message : "Could not play audio.");
    } finally {
      setIsLoadingAudio(false);
    }
  }

  return (
    <motion.div className="message-pair" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <div className="user-message">
        <div className="message-avatar user">You</div>
        <div>
          <div className="message-role">Question</div>
          <div className="message-text">{prompt}</div>
        </div>
      </div>
      <div className="assistant-message">
        <div className="message-avatar assistant"><Sparkles size={14} /></div>
        <div className="message-content-wrapper">
          <div className="message-role-bar">
            <div className="message-role">Quivr + MemPalace</div>
            <button className="tts-button" type="button" onClick={handlePlay} disabled={isLoadingAudio} aria-label="Play response">
              {isLoadingAudio ? <LoaderCircle className="spin" size={13} /> : isPlaying ? <Square size={13} /> : <Volume2 size={13} />}
              {isPlaying ? " Stop" : isLoadingAudio ? " Loading..." : " Listen"}
            </button>
          </div>
          <div className="message-text answer-text">{answer}</div>
          {audioError && <div className="audio-error" role="status">{audioError}</div>}
        </div>
      </div>
      <audio ref={audioRef} style={{ display: "none" }} />
    </motion.div>
  );
}

createRoot(document.getElementById("root")).render(<StrictMode><App /></StrictMode>);
