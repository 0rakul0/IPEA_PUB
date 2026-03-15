const elements = {
  apiBase: document.getElementById("apiBase"),
  chatForm: document.getElementById("chatForm"),
  documentsList: document.getElementById("documentsList"),
  messages: document.getElementById("messages"),
  resultLimit: document.getElementById("resultLimit"),
  sendButton: document.getElementById("sendButton"),
  sourcesSummary: document.getElementById("sourcesSummary"),
  statusBadge: document.getElementById("statusBadge"),
  userInput: document.getElementById("userInput"),
};

const STORAGE_KEY = "ipea_pub_api_base";

initialize();

function initialize() {
  const savedApiBase = (window.localStorage.getItem(STORAGE_KEY) || "").trim();
  const defaultApiBase = savedApiBase || getDefaultApiBase();
  elements.apiBase.value = defaultApiBase;

  elements.chatForm.addEventListener("submit", sendMessage);
  elements.userInput.addEventListener("keydown", handleComposerKeydown);
  elements.apiBase.addEventListener("change", persistApiBase);

  document.querySelectorAll(".prompt-chip").forEach((button) => {
    button.addEventListener("click", () => {
      elements.userInput.value = button.dataset.prompt || "";
      elements.userInput.focus();
    });
  });

  addMessage(
    "Pergunte sobre um tema, autor ou conjunto de publicacoes. Eu vou responder e listar as fontes usadas.",
    "assistant",
  );
}

function getDefaultApiBase() {
  const { hostname, port, protocol } = window.location;

  if (hostname === "localhost" || hostname === "127.0.0.1") {
    if (port === "8080") {
      return "";
    }

    return `${protocol}//${hostname}:8080`;
  }

  if (window.location.protocol.startsWith("http")) {
    return window.location.origin;
  }

  return "http://localhost:8080";
}

function persistApiBase() {
  window.localStorage.setItem(STORAGE_KEY, elements.apiBase.value.trim());
}

function handleComposerKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
}

function getApiUrl(path) {
  const base = elements.apiBase.value.trim().replace(/\/+$/, "");
  return base ? `${base}${path}` : path;
}

async function sendMessage(event) {
  event.preventDefault();

  const message = elements.userInput.value.trim();
  const limit = Number(elements.resultLimit.value) || 3;

  if (!message) {
    elements.userInput.focus();
    return;
  }

  addMessage(message, "user");
  elements.userInput.value = "";
  setLoadingState(true);
  updateSourcesSummary("Buscando resposta e fontes...");

  try {
    const response = await fetch(getApiUrl("/rag"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: message,
        limit,
      }),
    });

    if (!response.ok) {
      throw new Error(`Falha na API (${response.status})`);
    }

    const data = await response.json();
    addMessage(data.answer || "Nao foi possivel gerar uma resposta.", "assistant");
    updateDocuments(data.metadata || []);
    updateSourcesSummary(buildSourcesSummary(data.metadata || []));
    setStatus("Resposta pronta");
  } catch (error) {
    console.error("Erro ao consultar a API:", error);
    addMessage(
      `Nao consegui consultar a API agora. Verifique a URL configurada (${getApiUrl("") || "mesma origem"}) e se o backend esta ativo.`,
      "assistant error",
    );
    updateDocuments([]);
    updateSourcesSummary("Nenhuma fonte disponivel por causa de um erro de comunicacao.");
    setStatus("Falha na consulta");
  } finally {
    setLoadingState(false);
    elements.userInput.focus();
  }
}

function setLoadingState(isLoading) {
  elements.sendButton.disabled = isLoading;
  elements.userInput.disabled = isLoading;
  elements.resultLimit.disabled = isLoading;

  if (isLoading) {
    elements.sendButton.textContent = "Consultando...";
    setStatus("Consultando");
    return;
  }

  elements.sendButton.textContent = "Enviar pergunta";
}

function setStatus(text) {
  elements.statusBadge.textContent = text;
}

function addMessage(text, sender) {
  const article = document.createElement("article");
  article.className = `message ${sender}`;

  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = sender.includes("user") ? "Voce" : "Assistente";

  const body = document.createElement("p");
  body.className = "message-body";
  body.textContent = text;

  article.append(label, body);
  elements.messages.appendChild(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function updateDocuments(metadata) {
  elements.documentsList.replaceChildren();

  const documents = dedupeDocuments(metadata);
  if (!documents.length) {
    const emptyState = document.createElement("div");
    emptyState.className = "empty-state";
    emptyState.textContent = "Nenhum documento listado para esta resposta.";
    elements.documentsList.appendChild(emptyState);
    return;
  }

  documents.forEach((doc, index) => {
    elements.documentsList.appendChild(createDocumentCard(doc, index));
  });
}

function dedupeDocuments(metadata) {
  const seen = new Set();

  return metadata.filter((doc) => {
    const key = [
      doc.document_id || "",
      doc.titulo || "",
      doc.link_download || "",
      doc.link_pdf || "",
    ].join("|");

    if (seen.has(key)) {
      return false;
    }

    seen.add(key);
    return true;
  });
}

function createDocumentCard(doc, index) {
  const article = document.createElement("article");
  article.className = "document-card";

  const order = document.createElement("span");
  order.className = "document-rank";
  order.textContent = String(index + 1).padStart(2, "0");

  const title = document.createElement("h3");
  title.className = "document-title";
  title.textContent = doc.titulo || "Documento sem titulo";

  const meta = document.createElement("p");
  meta.className = "document-meta";
  meta.textContent = buildDocumentMeta(doc);

  const footer = document.createElement("div");
  footer.className = "document-footer";

  const score = document.createElement("span");
  score.className = "document-score";
  const rawScore = Number(doc.score);
  score.textContent = Number.isFinite(rawScore)
    ? `Score ${(rawScore * 100).toFixed(1)}%`
    : "Score indisponivel";

  footer.appendChild(score);

  const link = buildDocumentLink(doc);
  if (link) {
    const anchor = document.createElement("a");
    anchor.className = "document-link";
    anchor.href = link;
    anchor.target = "_blank";
    anchor.rel = "noreferrer";
    anchor.textContent = "Abrir documento";
    footer.appendChild(anchor);
  }

  article.append(order, title, meta, footer);
  return article;
}

function buildDocumentMeta(doc) {
  const parts = [];

  if (doc.autores) {
    parts.push(doc.autores);
  }

  if (doc.ano) {
    parts.push(String(doc.ano));
  }

  if (doc.tipo_conteudo) {
    parts.push(doc.tipo_conteudo);
  }

  return parts.length ? parts.join(" • ") : "Metadados indisponiveis";
}

function buildDocumentLink(doc) {
  return doc.link_download || doc.link_pdf || null;
}

function buildSourcesSummary(metadata) {
  if (!metadata.length) {
    return "A resposta nao trouxe documentos de apoio.";
  }

  const uniqueDocuments = dedupeDocuments(metadata).length;
  return `Resposta apoiada em ${uniqueDocuments} documento(s) recuperado(s).`;
}

function updateSourcesSummary(text) {
  elements.sourcesSummary.textContent = text;
}
