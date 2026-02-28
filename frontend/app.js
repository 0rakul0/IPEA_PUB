async function sendMessage() {
  const input = document.getElementById("userInput");
  const message = input.value;

  if (!message) return;

  addMessage(message, "user");
  input.value = "";

  try {
    const response = await fetch("http://localhost:8080/rag", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: message,
        limit: 3
      })
    });

    const data = await response.json();

    addMessage(data.answer, "bot");
    updateDocuments(data.metadata);

  } catch (error) {
    console.error("Erro:", error);
  }
}

function addMessage(text, sender) {
  const messagesDiv = document.getElementById("messages");

  const div = document.createElement("div");
  div.classList.add("message", sender);
  div.innerText = text;

  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function updateDocuments(metadata) {
  const docList = document.getElementById("documentsList");
  docList.innerHTML = "";

  metadata.forEach(doc => {
    const div = document.createElement("div");
    div.style.marginBottom = "15px";

    div.innerHTML = `
      <strong>${doc.titulo || "Sem título"}</strong><br>
      Autor(es): ${doc.autores || "Não informado"}<br>
      Score: ${(doc.score * 100).toFixed(1)}%
      <hr>
    `;

    docList.appendChild(div);
  });
}