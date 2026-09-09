import { lazy, StrictMode, Suspense, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion } from "motion/react";
import {
  Archive,
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowUp,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  ExternalLink,
  FileText,
  FolderOpen,
  Gauge,
  Layers3,
  LibraryBig,
  LoaderCircle,
  LogOut,
  MessageSquareText,
  Network,
  PanelLeft,
  Plus,
  Search,
  RefreshCw,
  ShieldCheck,
  Settings2,
  Sparkles,
  UploadCloud,
  Volume2,
  X,
  Zap,
  Square,
} from "lucide-react";
import "./styles.css";

const CUSTOM_MODEL = "__custom__";
const ACCEPTED_FILES = ".pdf,.docx,.xlsx,.html,.htm,.csv,.txt,.md,.markdown,.mdx";
const MarkdownAnswer = lazy(() => import("./MarkdownAnswer.jsx"));

function normalizeMessages(raw) {
  if (!Array.isArray(raw)) return [];
  const messages = [];
  raw.forEach((item) => {
    if (item && typeof item === "object" && !Array.isArray(item) && item.role && item.content) {
      messages.push({ role: item.role, content: String(item.content) });
      return;
    }
    if (Array.isArray(item) && item.length) {
      if (item[0]) messages.push({ role: "user", content: String(item[0]) });
      if (item[1]) messages.push({ role: "assistant", content: String(item[1]) });
    }
  });
  return messages;
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || "The API request failed.");
  }
  return payload;
}

function App() {
  const [view, setView] = useState(() => {
    return new URLSearchParams(window.location.search).get("view") === "observability"
      ? "observability"
      : "workspace";
  });
  const [modelCatalog, setModelCatalog] = useState({});
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [brainName, setBrainName] = useState("mempalace-quivr");
  const [wing, setWing] = useState("quivr-demo");
  const [memoryCount, setMemoryCount] = useState(5);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.42);
  const [files, setFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [memoryContext, setMemoryContext] = useState("");
  const [memoryMatches, setMemoryMatches] = useState([]);
  const [sources, setSources] = useState([]);
  const [policyInfo, setPolicyInfo] = useState(null);
  const [activeModel, setActiveModel] = useState("");
  const [helpOpen, setHelpOpen] = useState(false);
  const [indexStatus, setIndexStatus] = useState("No documents indexed yet.");
  const [status, setStatus] = useState({ tone: "neutral", text: "Ready when you are." });
  const [isIndexing, setIsIndexing] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [isRecalling, setIsRecalling] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  function openObservability() {
    window.location.href = "/admin/login";
  }

  function openWorkspace() {
    window.history.pushState({}, "", "/");
    setView("workspace");
  }

  useEffect(() => {
    let active = true;
    apiRequest("/api/models")
      .then((payload) => {
        if (active) setModelCatalog(payload.providers || {});
      })
      .catch(() => {
        if (active) {
          setStatus({ tone: "error", text: "Model catalog is unavailable." });
        }
      });
    apiRequest("/api/workspace")
      .then((payload) => {
        if (!active) return;
        setMessages(normalizeMessages(payload.messages));
        setIndexStatus(payload.index_status || "No documents indexed yet.");
        setActiveModel(payload.model_name || "");
        if (payload.provider) setProvider(payload.provider);
        if (payload.model_name) setModel(payload.model_name);
        if (typeof payload.similarity_threshold === "number") {
          setSimilarityThreshold(payload.similarity_threshold);
        }
        if (payload.brain_loaded) {
          setStatus({
            tone: "success",
            text: "Workspace restored after refresh. Index and memory are still on the server.",
          });
        }
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const providers = Object.keys(modelCatalog);
    if (!providers.length) return;
    setProvider((current) => (providers.includes(current) ? current : providers[0]));
  }, [modelCatalog]);

  useEffect(() => {
    if (!provider || model === CUSTOM_MODEL) return;
    const firstModel = modelCatalog[provider]?.models?.[0]?.id;
    if (!firstModel) return;
    const isKnownModel = modelCatalog[provider].models.some(({ id }) => id === model);
    if (!isKnownModel) setModel(firstModel);
  }, [modelCatalog, provider, model]);

  const providerConfig = modelCatalog[provider] || { description: "", models: [] };
  const selectedModel = model === CUSTOM_MODEL ? customModel.trim() : model;

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
    if (!provider || !selectedModel) {
      setIndexStatus("Choose a provider and model first.");
      return;
    }
    setIsIndexing(true);
    setStatus({ tone: "working", text: "Reading documents and building embeddings…" });
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    body.append("provider", provider);
    body.append("model_name", selectedModel);
    body.append("brain_name", brainName);
    try {
      const payload = await apiRequest("/api/index", { method: "POST", body });
      setIndexStatus(payload.message || "Documents indexed.");
      setActiveModel(payload.model_name || selectedModel);
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
    const asked = question.trim();
    setIsAsking(true);
    setStatus({ tone: "working", text: `Searching documents with ${selectedModel || "the selected model"}…` });
    setMessages((current) => [...current, { role: "user", content: asked }]);
    try {
      const payload = await apiRequest("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: asked,
          history: messages,
          provider,
          model_name: selectedModel,
          n_results: memoryCount,
          similarity_threshold: similarityThreshold,
        }),
      });
      const restored = normalizeMessages(payload.history);
      if (restored.length) {
        setMessages(restored);
      } else if (payload.answer) {
        setMessages((current) => [...current, { role: "assistant", content: payload.answer }]);
      }
      setMemoryContext(payload.context || "");
      setMemoryMatches(payload.matches || []);
      setSources(payload.sources || []);
      setPolicyInfo(payload.policy || null);
      setActiveModel(payload.model?.id || selectedModel);
      setQuestion("");
      setStatus({
        tone: payload.ok && payload.answer ? "success" : "error",
        text: payload.status || (payload.answer ? "Complete." : "The model returned an empty answer."),
      });
    } catch (error) {
      setStatus({ tone: "error", text: `Request failed: ${error.message}` });
    } finally {
      setIsAsking(false);
    }
  }

  async function handleRecall() {
    if (!question.trim()) {
      setMemoryContext("Enter a question to search conversation memory.");
      return;
    }
    setIsRecalling(true);
    try {
      const payload = await apiRequest("/api/recall", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          n_results: memoryCount,
          similarity_threshold: similarityThreshold,
        }),
      });
      setMemoryContext(payload.context || "No memories found.");
      setMemoryMatches(payload.matches || []);
      setPolicyInfo(payload.policy || null);
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
    setMemoryMatches([]);
    setSources([]);
    setQuestion("");
    setStatus({ tone: "neutral", text: "Conversation view cleared. Server memory is unchanged until you clear memory." });
  }

  async function clearIndex() {
    try {
      const payload = await apiRequest("/api/index/clear", { method: "POST" });
      setIndexStatus(payload.message || "Index cleared.");
      setSources([]);
      setStatus({ tone: "success", text: "Document vectors were removed. Memory is still available." });
    } catch (error) {
      setStatus({ tone: "error", text: error.message });
    }
  }

  async function clearMemory() {
    try {
      const payload = await apiRequest("/api/memory/clear", { method: "POST" });
      setMessages([]);
      setMemoryContext("");
      setMemoryMatches([]);
      setStatus({ tone: "success", text: payload.message || "Memory cleared." });
    } catch (error) {
      setStatus({ tone: "error", text: error.message });
    }
  }

  async function resetWorkspace() {
    try {
      const payload = await apiRequest("/api/workspace/reset", { method: "POST" });
      setMessages([]);
      setMemoryContext("");
      setMemoryMatches([]);
      setSources([]);
      setFiles([]);
      setIndexStatus("No documents indexed yet.");
      setActiveModel("");
      setStatus({ tone: "success", text: payload.message || "Workspace reset." });
    } catch (error) {
      setStatus({ tone: "error", text: error.message });
    }
  }

  if (view === "observability") {
    return <AdminConsole onBack={openWorkspace} />;
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
          <button className="admin-nav-button" type="button" onClick={openObservability}><ShieldCheck size={15} /> Admin observability</button>
          <div className="live-pill"><span className="live-dot" /> Local workspace</div>
          <button className="icon-button" type="button" aria-label="Help" onClick={() => setHelpOpen(true)}><CircleHelp size={18} /></button>
          <button className="avatar" type="button" aria-label="Single-user workspace">1</button>
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
          <StatCard icon={<Zap size={18} />} label="Active model" value={activeModel || selectedModel || "Loading"} detail={provider || "Fetching model catalog…"} accent="amber" />
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
                <select id="provider" value={provider} onChange={(event) => setProvider(event.target.value)} disabled={!Object.keys(modelCatalog).length}>
                  {!Object.keys(modelCatalog).length && <option value="">Loading providers…</option>}
                  {Object.entries(modelCatalog).map(([name]) => <option key={name} value={name}>{name}</option>)}
                </select>
                <ChevronDown size={15} />
              </div>
              {providerConfig.description && <div className="field-hint">{providerConfig.description}</div>}
            </div>
            <div className="form-section">
              <label htmlFor="model">Model</label>
              <div className="select-wrap">
                <select id="model" value={model} onChange={(event) => setModel(event.target.value)} disabled={!providerConfig.models.length}>
                  {providerConfig.models.map((option) => <option key={option.id} value={option.id}>{option.label} · {option.details}</option>)}
                  <option value={CUSTOM_MODEL}>Custom deployment/model ID…</option>
                </select>
                <ChevronDown size={15} />
              </div>
              {model === CUSTOM_MODEL && <input id="custom-model" className="custom-model-input" placeholder="Enter your deployment/model ID" value={customModel} onChange={(event) => setCustomModel(event.target.value)} />}
              {provider === "Microsoft Foundry" && <div className="field-hint">Use the deployment name configured in Microsoft Foundry.</div>}
              {activeModel && activeModel !== selectedModel && <div className="field-hint">Last answer used {activeModel}. The next question will use {selectedModel}.</div>}
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
              <label htmlFor="wing">Memory collection</label>
              <input id="wing" value={wing} onChange={(event) => setWing(event.target.value)} />
            </div>
            <div className="form-section range-section">
              <div className="range-label"><label htmlFor="memory-count">Memories to recall</label><span>{memoryCount}</span></div>
              <input id="memory-count" type="range" min="1" max="10" value={memoryCount} onChange={(event) => setMemoryCount(Number(event.target.value))} />
            </div>
            <div className="form-section range-section">
              <div className="range-label"><label htmlFor="similarity-threshold">Recall similarity threshold</label><span>{similarityThreshold.toFixed(2)}</span></div>
              <input id="similarity-threshold" type="range" min="0" max="1" step="0.01" value={similarityThreshold} onChange={(event) => setSimilarityThreshold(Number(event.target.value))} />
              <div className="field-hint">Memories below this cosine score are dropped unless regex overlap is strong.</div>
            </div>
            <div className="workspace-actions">
              <button className="secondary-button small" type="button" onClick={clearIndex}>Clear index</button>
              <button className="secondary-button small" type="button" onClick={clearMemory}>Clear memory</button>
              <button className="secondary-button small" type="button" onClick={resetWorkspace}>Reset workspace</button>
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
                <div className="dropzone-subtitle">PDF, DOCX, XLSX, HTML, CSV, TXT, or Markdown · up to your imagination</div>
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
                    {messages.map((item, index) => (
                      item.role === "user"
                        ? <UserMessage key={`user-${index}`} prompt={item.content} />
                        : <Message key={`assistant-${index}`} prompt="" answer={item.content} sources={item.role === "assistant" && index === messages.length - 1 ? sources : []} />
                    ))}
                  </div>
                )}
              </div>
              <form className="question-composer" onSubmit={handleAsk}>
                <div className="composer-label"><Search size={14} /> Ask about your sources</div>
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      handleAsk(event);
                    }
                  }}
                  placeholder="What would you like to understand?"
                  rows="2"
                />
                <div className="composer-footer"><span>Press Enter to send · Shift + Enter for a new line</span><div className="composer-actions"><button className="secondary-button small" type="button" onClick={handleRecall} disabled={isRecalling}>{isRecalling ? <LoaderCircle className="spin" size={15} /> : <Archive size={15} />} Recall memory</button><motion.button className="primary-button send-button" type="submit" disabled={isAsking || !question.trim()} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>{isAsking ? <LoaderCircle className="spin" size={17} /> : <ArrowUp size={17} />} {isAsking ? "Thinking…" : "Ask"}</motion.button></div></div>
              </form>
            </div>

            <div className="memory-panel panel">
              <div className="panel-heading compact"><div><div className="section-kicker"><Archive size={14} /> Retrieval trace</div><h3>Memory matches</h3></div><div className="context-badge">{memoryMatches.length ? `${memoryMatches.length} kept` : "Waiting for recall"}</div></div>
              {policyInfo && <div className="policy-strip">Budget {policyInfo.budget?.memory_budget || "—"} tokens · used {policyInfo.token_usage?.memory || 0}{policyInfo.summarized ? " · older turns summarized" : ""}</div>}
              {memoryMatches.length > 0 && (
                <div className="match-list">
                  {memoryMatches.map((match) => (
                    <div className="match-card" key={match.id}>
                      <div className="match-meta"><strong>{match.matched_field}</strong><span>score {match.score}</span></div>
                      <div className="match-explain">{match.explanation}</div>
                    </div>
                  ))}
                </div>
              )}
              <div className={`context-box ${memoryContext ? "has-context" : ""}`}>{memoryContext || "Memory retrieved for your next question will appear here. Refresh keeps this store; use Clear memory to wipe it."}</div>
            </div>
          </section>
        </div>
      </main>

      <footer className="footer"><span>Built for thoughtful retrieval</span><span className="footer-divider" /><span>Quivr RAG <b>·</b> DuckDB memory</span></footer>
      {helpOpen && (
        <div className="help-overlay" onClick={() => setHelpOpen(false)}>
          <div className="help-card" onClick={(event) => event.stopPropagation()}>
            <h3>Single-user workspace</h3>
            <p>Refresh restores the index and conversation from DuckDB. Use <b>Clear index</b> to drop vectors, <b>Clear memory</b> to drop Q&A turns, or <b>Reset workspace</b> for both.</p>
            <p>Changing the model applies on the next question without re-indexing. Memory is packed to that model's context window and summarized when it would overflow.</p>
            <button className="primary-button" type="button" onClick={() => setHelpOpen(false)}>Got it</button>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ icon, label, value, detail, accent }) {
  return <motion.div className={`stat-card ${accent}`} whileHover={{ y: -3 }} transition={{ type: "spring", stiffness: 300, damping: 20 }}><div className="stat-icon">{icon}</div><div className="stat-text"><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></div></motion.div>;
}

function AdminConsole({ onBack }) {
  const [dashboard, setDashboard] = useState(null);
  const [selectedTraceId, setSelectedTraceId] = useState(null);
  const [error, setError] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [governanceInput, setGovernanceInput] = useState({ collection: "quivr-demo", text: "" });
  const [governanceResult, setGovernanceResult] = useState(null);
  const [isEvaluatingGovernance, setIsEvaluatingGovernance] = useState(false);

  async function loadDashboard() {
    setIsRefreshing(true);
    try {
      const meResponse = await fetch("/api/admin/me");
      const me = await meResponse.json().catch(() => ({}));
      if (meResponse.status === 401 || !me.authenticated) {
        window.location.href = "/admin/login";
        return;
      }
      const payload = await apiRequest("/api/admin/observability");
      setDashboard(payload);
      setError("");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load observability data.");
    } finally {
      setIsRefreshing(false);
    }
  }

  async function evaluateGovernance(event) {
    event.preventDefault();
    setIsEvaluatingGovernance(true);
    try {
      const result = await apiRequest("/api/admin/governance/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(governanceInput),
      });
      setGovernanceResult(result);
    } catch (requestError) {
      setGovernanceResult({ decision: "error", reasons: [requestError.message] });
    } finally {
      setIsEvaluatingGovernance(false);
    }
  }

  useEffect(() => {
    loadDashboard();
    const interval = window.setInterval(loadDashboard, 10000);
    return () => window.clearInterval(interval);
  }, []);

  const summary = dashboard?.summary || {};
  const traces = dashboard?.traces || [];
  const selectedTrace = traces.find((trace) => trace.trace_id === selectedTraceId) || traces[0];
  const integrations = dashboard?.integrations || {};
  const governance = dashboard?.governance || {};
  const governancePolicy = governance.policy || {};
  const auditEvents = governance.audit_events || [];

  return (
    <div className="app-shell admin-shell">
      <div className="noise" />
      <div className="aurora aurora-one" />
      <div className="aurora aurora-two" />
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><Activity size={21} /></div>
          <div>
            <div className="brand-name">mempalace <span>×</span> quivr</div>
            <div className="brand-caption">Admin observability console</div>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="admin-identity"><ShieldCheck size={14} /> {dashboard?.admin?.name || "Admin"}</div>
          <button className="icon-button" type="button" onClick={loadDashboard} aria-label="Refresh traces"><RefreshCw className={isRefreshing ? "spin" : ""} size={17} /></button>
          <button className="icon-button" type="button" onClick={() => { window.location.href = "/admin/logout"; }} aria-label="Sign out"><LogOut size={17} /></button>
        </div>
      </header>

      <main className="page-container">
        <section className="hero-section admin-hero">
          <button className="back-button" type="button" onClick={onBack}><ArrowLeft size={15} /> Back to workspace</button>
          <div className="eyebrow"><Activity size={14} /> Open-source RAG observability</div>
          <h1>See every <span>decision.</span></h1>
          <p className="hero-copy">Trace retrieval, memory, model generation, and TTS as one operational story. Content capture is off by default; Logfire receives safe metadata only when enabled.</p>
        </section>

        {error && <div className="status-banner error"><AlertTriangle size={14} /> {error}</div>}
        <section className="stats-grid" aria-label="Observability overview">
          <StatCard icon={<Activity size={18} />} label="Traces in memory" value={summary.total_traces ?? "—"} detail="Most recent 100" accent="violet" />
          <StatCard icon={<Clock3 size={18} />} label="P95 latency" value={formatDuration(summary.p95_latency_ms)} detail={`Average ${formatDuration(summary.avg_latency_ms)}`} accent="cyan" />
          <StatCard icon={<AlertTriangle size={18} />} label="Error rate" value={formatPercent(summary.error_rate)} detail={`${summary.error_traces || 0} failed traces`} accent="amber" />
          <StatCard icon={<ShieldCheck size={18} />} label="Trace export" value={integrations.logfire_configured && integrations.logfire_send_to_logfire ? "Logfire live" : "Local only"} detail={integrations.content_capture ? "Content capture on" : "Metadata only"} accent="green" />
        </section>

        <div className="admin-grid">
          <section className="panel trace-panel">
            <div className="panel-heading">
              <div><div className="section-kicker"><Activity size={14} /> Recent traces</div><h2>What the system is doing</h2></div>
              <div className="source-count">Auto-refresh 10s</div>
            </div>
            {traces.length === 0 ? (
              <div className="admin-empty"><Activity size={28} /><h3>Waiting for traffic</h3><p>Ask a question, recall a memory, index a document, or play TTS to create the first trace.</p></div>
            ) : (
              <div className="trace-list">
                {traces.map((trace) => (
                  <button className={`trace-row ${selectedTrace?.trace_id === trace.trace_id ? "selected" : ""}`} type="button" key={trace.trace_id} onClick={() => setSelectedTraceId(trace.trace_id)}>
                    <span className={`trace-status ${trace.status}`} />
                    <span className="trace-main"><strong>{trace.name}</strong><small>{formatDate(trace.started_at)} · {trace.spans.length} span{trace.spans.length === 1 ? "" : "s"}</small></span>
                    <span className="trace-duration">{formatDuration(trace.duration_ms)}</span>
                  </button>
                ))}
              </div>
            )}
          </section>

          <section className="panel detail-panel">
            <div className="panel-heading">
              <div><div className="section-kicker"><ShieldCheck size={14} /> Trace detail</div><h2>{selectedTrace?.name || "No trace selected"}</h2></div>
              {integrations.logfire_project_url && <button className="text-button" type="button" onClick={() => window.open(integrations.logfire_project_url, "_blank", "noopener,noreferrer")}><ExternalLink size={13} /> Logfire</button>}
            </div>
            {selectedTrace ? (
              <>
                <div className="trace-meta"><span>{selectedTrace.status}</span><span>{formatDate(selectedTrace.started_at)}</span><span>{formatDuration(selectedTrace.duration_ms)}</span></div>
                <div className="span-list">
                  {selectedTrace.spans.map((span) => (
                    <div className="span-row" key={span.span_id}>
                      <span className={`trace-status ${span.status}`} />
                      <span className="span-main"><strong>{span.name}</strong><small>{span.kind} · {Object.entries(span.attributes || {}).map(([key, value]) => `${key}: ${value}`).join(" · ") || "no attributes"}</small></span>
                      <span className="trace-duration">{formatDuration(span.duration_ms)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : <div className="admin-empty compact"><ShieldCheck size={25} /><p>Trace details will appear here.</p></div>}
          </section>
        </div>

        <section className="panel operations-panel">
          <div className="panel-heading compact"><div><div className="section-kicker"><Gauge size={14} /> Instrumentation map</div><h3>Every important path is measurable</h3></div></div>
          <div className="operation-grid">
            {Object.entries(summary.operations || {}).map(([name, count]) => <div className="operation-card" key={name}><span>{name}</span><strong>{count}</strong></div>)}
            {Object.keys(summary.operations || {}).length === 0 && <div className="operation-card muted-operation">Operations appear after the first request.</div>}
          </div>
        </section>

        <section className="panel governance-panel">
          <div className="panel-heading compact">
            <div><div className="section-kicker"><ShieldCheck size={14} /> Agent governance</div><h3>Policy controls for every retrieval</h3></div>
            <div className={`governance-status ${governance.enabled ? "active" : "inactive"}`}><span />{governance.enabled ? "AGT active" : "AGT unavailable"}</div>
          </div>
          {governance.error && <div className="status-banner error"><AlertTriangle size={14} /> {governance.error}</div>}
          <div className="governance-summary">
            <div className="governance-card"><span>Agent identity</span><strong>{governance.agent_id || "—"}</strong></div>
            <div className="governance-card"><span>Collection</span><strong>{governance.collection || "—"}</strong></div>
            <div className="governance-card"><span>Rate limit</span><strong>{governancePolicy.max_retrievals_per_minute || 0} / min</strong></div>
            <div className="governance-card"><span>AGT version</span><strong>{governance.core_version || governance.rag_version || "—"}</strong></div>
          </div>
          <div className="governance-rules"><span><b>Allowed:</b> {formatPolicyValues(governancePolicy.allowed_collections)}</span><span><b>Denied:</b> {formatPolicyValues(governancePolicy.denied_collections)}</span><span><b>Scanners:</b> {formatPolicyValues(governancePolicy.content_policies)}</span></div>
          <div className="governance-lower-grid">
            <div>
              <div className="governance-label">Enabled capabilities</div>
              <div className="capability-list">
                {(governance.capabilities || []).map((capability) => <span className="capability-chip" key={capability}><Check size={12} /> {capability.replaceAll("_", " ")}</span>)}
                {!governance.capabilities?.length && <span className="muted-operation">No governance capabilities reported.</span>}
              </div>
              <div className="governance-label audit-label">Recent audit events</div>
              {auditEvents.length ? <div className="audit-list">{auditEvents.map((event, index) => <div className="audit-row" key={`${event.timestamp || "event"}-${index}`}><span className={`audit-decision ${event.decision === "allowed" ? "allow" : "deny"}`}>{event.decision || "event"}</span><span className="audit-main"><strong>{event.collection || "unknown collection"}</strong><small>{formatDate(event.timestamp)} · {event.num_chunks_retrieved ?? 0} chunks · {event.query_hash ? `${event.query_hash.slice(0, 12)}…` : "no query hash"}</small></span></div>)}</div> : <div className="admin-empty compact"><Activity size={22} /><p>Governed retrieval events will appear after the first question.</p></div>}
            </div>
            <form className="governance-evaluator" onSubmit={evaluateGovernance}>
              <div className="governance-label">Dry-run a retrieval decision</div>
              <p>Check collection access and content policies without calling the model or writing an audit event.</p>
              <label htmlFor="governance-collection">Collection</label>
              <input id="governance-collection" value={governanceInput.collection} onChange={(event) => setGovernanceInput((current) => ({ ...current, collection: event.target.value }))} />
              <label htmlFor="governance-text">Query text</label>
              <textarea id="governance-text" rows="4" value={governanceInput.text} onChange={(event) => setGovernanceInput((current) => ({ ...current, text: event.target.value }))} placeholder="Try a normal query or a policy-sensitive string…" />
              <button className="secondary-button" type="submit" disabled={isEvaluatingGovernance}>{isEvaluatingGovernance ? <LoaderCircle className="spin" size={15} /> : <ShieldCheck size={15} />} {isEvaluatingGovernance ? "Checking…" : "Evaluate retrieval"}</button>
              {governanceResult && <div className={`governance-result ${governanceResult.decision}`}><strong>{governanceResult.decision}</strong>{governanceResult.reasons?.length ? <span>{governanceResult.reasons.join(" ")}</span> : governanceResult.warnings?.length ? <span>{governanceResult.warnings.join(" ")}</span> : <span>No policy violations detected.</span>}</div>}
            </form>
          </div>
        </section>
      </main>
      <footer className="footer"><span>Microsoft Entra protected</span><span className="footer-divider" /><span>Logfire traces <b>·</b> safe metadata by default</span></footer>
    </div>
  );
}

function formatDuration(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "—";
  const duration = Number(value);
  return duration < 1000 ? `${Math.round(duration)} ms` : `${(duration / 1000).toFixed(2)} s`;
}

function formatPercent(value) {
  if (value === undefined || value === null) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatPolicyValues(values) {
  return values?.length ? values.join(", ") : "none configured";
}

function formatDate(value) {
  if (!value) return "unknown time";
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function UserMessage({ prompt }) {
  return (
    <div className="user-message">
      <div className="message-avatar user">You</div>
      <div>
        <div className="message-role">Question</div>
        <div className="message-text">{prompt}</div>
      </div>
    </div>
  );
}

function Message({ prompt, answer, sources = [] }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [audioError, setAudioError] = useState("");
  const [audioReady, setAudioReady] = useState(false);
  const audioRef = useRef(null);
  const audioUrlRef = useRef("");

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      window.speechSynthesis?.cancel();
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
      window.speechSynthesis?.cancel();
      setIsPlaying(false);
      return;
    }

    if (!answer) return;

    setIsLoadingAudio(true);
    setAudioError("");
    setAudioReady(false);
    releaseAudioUrl();
    try {
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: answer, voice: "af_bella" }),
      });
      if (!response.ok) {
        const body = await response.text().catch(() => "");
        let detail = body.trim();
        try {
          const payload = JSON.parse(body);
          detail = payload.detail || detail;
        } catch {
          // Render/proxy errors are often plain text or HTML rather than JSON.
        }
        throw new Error(
          detail || `TTS request failed with HTTP ${response.status}.`,
        );
      }

      const blob = await response.blob();
      if (!blob.size || !audioRef.current) {
        throw new Error("The TTS service returned an empty audio file.");
      }

      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;
      audioRef.current.src = url;
      audioRef.current.load();
      setAudioReady(true);
      audioRef.current.onended = () => setIsPlaying(false);
      audioRef.current.onerror = () => {
        setIsPlaying(false);
        setAudioError("The browser could not play the generated audio.");
      };
      try {
        await audioRef.current.play();
        setIsPlaying(true);
      } catch (playError) {
        if (playError?.name === "NotAllowedError") {
          setAudioError("Audio generated. Press play in the audio controls below.");
        } else {
          throw playError;
        }
      }
    } catch (error) {
      console.error(error);
      const serverError = error instanceof Error ? error.message : "Could not generate server audio.";
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(answer);
        utterance.onend = () => setIsPlaying(false);
        utterance.onerror = (event) => {
          setIsPlaying(false);
          setAudioError(`Browser speech failed${event.error ? `: ${event.error}` : "."}`);
        };
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
        setIsPlaying(true);
        setAudioError(`Server audio unavailable; using your browser voice. ${serverError}`);
      } else {
        setAudioError(serverError);
      }
    } finally {
      setIsLoadingAudio(false);
    }
  }

  return (
    <motion.div className="message-pair" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      {prompt ? (
        <div className="user-message">
          <div className="message-avatar user">You</div>
          <div>
            <div className="message-role">Question</div>
            <div className="message-text">{prompt}</div>
          </div>
        </div>
      ) : null}
      <div className="assistant-message">
        <div className="message-avatar assistant"><Sparkles size={14} /></div>
        <div className="message-content-wrapper">
          <div className="message-role-bar">
            <div className="assistant-title-group">
              <div className="message-role">Quivr + MemPalace</div>
              <span className="answer-badge"><Check size={11} /> Grounded response</span>
            </div>
            <div className="answer-actions">
              <button className="tts-button" type="button" onClick={handlePlay} disabled={isLoadingAudio} aria-label="Play response">
                {isLoadingAudio ? <LoaderCircle className="spin" size={13} /> : isPlaying ? <Square size={13} /> : <Volume2 size={13} />}
                {isPlaying ? " Stop" : isLoadingAudio ? " Loading..." : " Listen"}
              </button>
            </div>
          </div>
          <div className="message-text answer-text markdown-content" aria-label="Assistant answer">
            <Suspense fallback={<div className="markdown-loading">Formatting answer…</div>}>
              {answer ? <MarkdownAnswer answer={answer} /> : <p>The model did not return visible text.</p>}
            </Suspense>
          </div>
          {sources?.length > 0 && (
            <div className="source-list">
              {sources.map((source, index) => (
                <div className="source-chip" key={`${source.filename}-${index}`}>
                  <strong>{source.filename || "chunk"}</strong>
                  <span>{source.snippet}</span>
                </div>
              ))}
            </div>
          )}
          {audioError && <div className="audio-error" role="status">{audioError}</div>}
        </div>
      </div>
      <audio
        ref={audioRef}
        className="tts-audio"
        controls={audioReady}
        preload="metadata"
        aria-label="Generated response audio"
      />
    </motion.div>
  );
}

createRoot(document.getElementById("root")).render(<StrictMode><App /></StrictMode>);
