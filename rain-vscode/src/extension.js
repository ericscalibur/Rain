// Rain ⛈️ VSCode Extension
// Connects VSCode to a locally-running Rain server at http://localhost:7734
// No cloud. No API keys. No tracking.

const vscode = require("vscode");

// ── Config helpers ────────────────────────────────────────────────────

function getServerUrl() {
  return (
    vscode.workspace.getConfiguration("rain").get("serverUrl") ||
    "http://localhost:7734"
  );
}

function getWebSearch() {
  return vscode.workspace.getConfiguration("rain").get("webSearch") || false;
}

function getSandbox() {
  return vscode.workspace.getConfiguration("rain").get("sandbox") || false;
}

// ── HTTP helpers (no external deps — Node built-ins only) ─────────────

function httpPost(url, body, timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    const http = url.startsWith("https") ? require("https") : require("http");
    const parsed = new URL(url);
    const data = JSON.stringify(body);

    const req = http.request(
      {
        hostname: parsed.hostname,
        port: parsed.port || (url.startsWith("https") ? 443 : 80),
        path: parsed.pathname + (parsed.search || ""),
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(data),
        },
      },
      (res) => {
        let raw = "";
        res.on("data", (chunk) => (raw += chunk));
        res.on("end", () => {
          try {
            resolve({ status: res.statusCode, body: JSON.parse(raw) });
          } catch {
            resolve({ status: res.statusCode, body: raw });
          }
        });
      }
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new Error(`Request timed out after ${timeoutMs}ms`));
    });
    req.on("error", reject);
    req.write(data);
    req.end();
  });
}

function httpGet(url, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const http = url.startsWith("https") ? require("https") : require("http");
    const req = http.get(url, { timeout: timeoutMs }, (res) => {
      let raw = "";
      res.on("data", (chunk) => (raw += chunk));
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(raw) });
        } catch {
          resolve({ status: res.statusCode, body: raw });
        }
      });
    });
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timed out"));
    });
  });
}

// SSE streaming over HTTP for the /api/chat endpoint
function streamChat(url, body, onEvent, onDone, onError) {
  const http = url.startsWith("https") ? require("https") : require("http");
  const parsed = new URL(url);
  const data = JSON.stringify(body);

  const req = http.request(
    {
      hostname: parsed.hostname,
      port: parsed.port || 80,
      path: parsed.pathname,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(data),
        Accept: "text/event-stream",
      },
    },
    (res) => {
      let buf = "";
      res.on("data", (chunk) => {
        buf += chunk.toString();
        const lines = buf.split("\n");
        buf = lines.pop(); // keep incomplete line in buffer
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const raw = line.slice(6).trim();
            if (!raw) continue;
            try {
              const evt = JSON.parse(raw);
              onEvent(evt);
              if (evt.type === "done" || evt.type === "error") {
                onDone(evt);
                return;
              }
            } catch {
              /* ignore malformed event */
            }
          }
        }
      });
      res.on("end", () => onDone(null));
      res.on("error", onError);
    }
  );
  req.on("error", onError);
  req.write(data);
  req.end();
  return req; // caller can call req.destroy() to cancel
}

// ── Check if Rain server is reachable ─────────────────────────────────

async function checkServer() {
  try {
    const res = await httpGet(`${getServerUrl()}/api/health`, 5000);
    return res.status === 200 && res.body && res.body.ollama === true;
  } catch {
    return false;
  }
}

// ── Ask Rain and stream into the output channel ───────────────────────

async function askRain(prompt, context = {}) {
  const base = getServerUrl();
  const alive = await checkServer();
  if (!alive) {
    vscode.window.showErrorMessage(
      `Rain server not reachable at ${base}. ` +
        "Make sure Rain is running: ./rain-web"
    );
    return null;
  }

  const body = {
    message: prompt,
    web_search: context.webSearch ?? getWebSearch(),
    sandbox: context.sandbox ?? getSandbox(),
    project_path: context.projectPath ?? null,
  };

  return new Promise((resolve, reject) => {
    let finalContent = null;
    let progressMessage = null;

    vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "⛈️ Rain thinking...",
        cancellable: true,
      },
      (progress, token) => {
        return new Promise((progressResolve) => {
          const req = streamChat(
            `${base}/api/chat`,
            body,
            (evt) => {
              if (token.isCancellationRequested) {
                req.destroy();
                progressResolve();
                resolve(null);
                return;
              }
              if (evt.type === "routing") {
                progress.report({ message: `🤖 ${evt.agent}` });
              } else if (evt.type === "progress") {
                progress.report({ message: evt.message });
              } else if (evt.type === "done") {
                finalContent = evt.content;
              } else if (evt.type === "error") {
                vscode.window.showErrorMessage(`Rain error: ${evt.message}`);
              }
            },
            (_lastEvt) => {
              progressResolve();
              resolve(finalContent);
            },
            (err) => {
              progressResolve();
              reject(err);
            }
          );

          token.onCancellationRequested(() => {
            req.destroy();
            progressResolve();
            resolve(null);
          });
        });
      }
    );
  });
}

// ── Build context from active editor ─────────────────────────────────

function getEditorContext(editor, mode = "selection") {
  if (!editor) return { text: "", filename: "", language: "" };

  const doc = editor.document;
  const filename = doc.fileName.split(/[\\/]/).pop();
  const language = doc.languageId;

  let text = "";
  if (mode === "selection" && !editor.selection.isEmpty) {
    text = doc.getText(editor.selection);
  } else if (mode === "file") {
    text = doc.getText();
  }

  return { text, filename, language, uri: doc.uri };
}

function buildCodePrompt(instruction, { text, filename, language }) {
  if (!text) return instruction;
  return (
    `${instruction}\n\n` +
    `File: ${filename}\n` +
    `Language: ${language}\n\n` +
    "```" +
    language +
    "\n" +
    text +
    "\n```"
  );
}

// ── Webview panel (chat UI) ───────────────────────────────────────────

let chatPanel = null;

function openChatPanel(context) {
  if (chatPanel) {
    chatPanel.reveal(vscode.ViewColumn.Beside);
    return;
  }

  chatPanel = vscode.window.createWebviewPanel(
    "rainChat",
    "Rain ⛈️",
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    }
  );

  chatPanel.webview.html = getChatPanelHtml(getServerUrl());

  // Forward messages from the webview to the Rain server
  chatPanel.webview.onDidReceiveMessage(
    async (msg) => {
      if (msg.type === "send") {
        const content = await askRain(msg.text, {
          webSearch: msg.webSearch,
          sandbox: msg.sandbox,
          projectPath: msg.projectPath || null,
        });
        if (content !== null) {
          chatPanel.webview.postMessage({ type: "response", content });
        }
      } else if (msg.type === "getServerUrl") {
        chatPanel.webview.postMessage({
          type: "serverUrl",
          url: getServerUrl(),
        });
      } else if (msg.type === "getWorkspacePath") {
        const ws =
          vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath || null;
        chatPanel.webview.postMessage({ type: "workspacePath", path: ws });
      }
    },
    undefined,
    context.subscriptions
  );

  chatPanel.onDidDispose(() => {
    chatPanel = null;
  });
}

function getChatPanelHtml(serverUrl) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Rain ⛈️</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:      var(--vscode-editor-background);
      --fg:      var(--vscode-editor-foreground);
      --border:  var(--vscode-panel-border, #333);
      --input-bg: var(--vscode-input-background);
      --btn-bg:  var(--vscode-button-background);
      --btn-fg:  var(--vscode-button-foreground);
      --accent:  var(--vscode-focusBorder, #4a9eff);
      --code-bg: var(--vscode-textCodeBlock-background, #1a1a1a);
    }
    html, body { height: 100%; background: var(--bg); color: var(--fg);
                 font-family: var(--vscode-font-family, system-ui); font-size: 13px; }
    #app { display: flex; flex-direction: column; height: 100vh; }
    #messages { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
    .msg { max-width: 100%; line-height: 1.55; }
    .msg.user   { align-self: flex-end; background: var(--vscode-button-background, #2a5db0);
                  color: var(--vscode-button-foreground, #fff); padding: 8px 12px;
                  border-radius: 12px 12px 2px 12px; max-width: 88%; }
    .msg.rain   { align-self: flex-start; background: var(--vscode-editor-inactiveSelectionBackground, #2a2a2a);
                  padding: 10px 14px; border-radius: 2px 12px 12px 12px;
                  max-width: 96%; white-space: pre-wrap; word-break: break-word; }
    .msg.system { align-self: center; font-size: 11px; color: var(--vscode-descriptionForeground, #888);
                  font-style: italic; }
    pre { background: var(--code-bg); border-radius: 6px; padding: 10px; overflow-x: auto;
          font-size: 12px; margin: 6px 0; }
    code { font-family: var(--vscode-editor-font-family, monospace); }
    #toolbar { padding: 8px 12px; border-top: 1px solid var(--border);
               display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .toggle-row { display: flex; align-items: center; gap: 5px; font-size: 11px;
                  color: var(--vscode-descriptionForeground, #888); }
    input[type=checkbox] { cursor: pointer; }
    #input-row { display: flex; gap: 6px; padding: 8px 12px;
                 border-top: 1px solid var(--border); }
    #chat-input { flex: 1; background: var(--input-bg); color: var(--fg);
                  border: 1px solid var(--border); border-radius: 6px;
                  padding: 8px 10px; font-size: 13px; font-family: inherit;
                  resize: none; min-height: 38px; max-height: 140px; outline: none; }
    #chat-input:focus { border-color: var(--accent); }
    #send-btn { background: var(--btn-bg); color: var(--btn-fg); border: none;
                border-radius: 6px; padding: 8px 16px; cursor: pointer;
                font-size: 13px; font-family: inherit; white-space: nowrap; }
    #send-btn:hover { opacity: 0.85; }
    #send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .spinner { display: inline-block; width: 12px; height: 12px;
               border: 2px solid transparent; border-top-color: var(--accent);
               border-radius: 50%; animation: spin 0.7s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .badge { font-size: 10px; opacity: 0.65; margin-top: 4px; }
  </style>
</head>
<body>
<div id="app">
  <div id="messages">
    <div class="msg system">⛈️ Rain — sovereign local AI. Type a message to start.</div>
  </div>
  <div id="toolbar">
    <label class="toggle-row">
      <input type="checkbox" id="web-search-toggle"> 🌐 Web search
    </label>
    <label class="toggle-row">
      <input type="checkbox" id="sandbox-toggle"> 🔬 Sandbox
    </label>
    <div class="toggle-row" id="project-label" style="margin-left:auto; font-size:10px;"></div>
  </div>
  <div id="input-row">
    <textarea id="chat-input" placeholder="Ask Rain anything..." rows="1"></textarea>
    <button id="send-btn">Send</button>
  </div>
</div>

<script>
  const vscode = acquireVsCodeApi();
  const messages   = document.getElementById('messages');
  const input      = document.getElementById('chat-input');
  const sendBtn    = document.getElementById('send-btn');
  const wsToggle   = document.getElementById('web-search-toggle');
  const sbToggle   = document.getElementById('sandbox-toggle');
  const projLabel  = document.getElementById('project-label');

  let workspacePath = null;
  let thinking = false;

  // Ask host for workspace path
  vscode.postMessage({ type: 'getWorkspacePath' });

  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (msg.type === 'response') {
      setThinking(false);
      appendMsg('rain', msg.content);
    } else if (msg.type === 'workspacePath') {
      workspacePath = msg.path;
      if (workspacePath) {
        const name = workspacePath.split(/[\\/]/).pop();
        projLabel.textContent = '📂 ' + name;
        projLabel.title = workspacePath;
      }
    } else if (msg.type === 'serverUrl') {
      // available if needed
    }
  });

  function appendMsg(role, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;

    // Simple code block rendering
    const escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    div.innerHTML = escaped.replace(
      /```(\w*)\n([\s\S]*?)```/g,
      (_, lang, code) => '<pre><code>' + code + '</code></pre>'
    );

    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  function appendSystem(text) {
    const div = document.createElement('div');
    div.className = 'msg system';
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  function setThinking(val) {
    thinking = val;
    sendBtn.disabled = val;
    sendBtn.innerHTML = val
      ? '<span class="spinner"></span>'
      : 'Send';
    input.disabled = val;
  }

  function send() {
    const text = input.value.trim();
    if (!text || thinking) return;
    appendMsg('user', text);
    input.value = '';
    input.style.height = 'auto';
    setThinking(true);

    vscode.postMessage({
      type: 'send',
      text,
      webSearch: wsToggle.checked,
      sandbox: sbToggle.checked,
      projectPath: workspacePath,
    });
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  // Auto-grow textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  });
</script>
</body>
</html>`;
}

// ── Output channel for non-panel results ─────────────────────────────

let outputChannel = null;

function getOutputChannel() {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel("Rain ⛈️");
  }
  return outputChannel;
}

function showInOutput(content) {
  const ch = getOutputChannel();
  ch.clear();
  ch.appendLine(content);
  ch.show(true);
}

// ── Commands ──────────────────────────────────────────────────────────

async function cmdAskAboutSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Rain: Select some code first.");
    return;
  }
  const question = await vscode.window.showInputBox({
    prompt: "What do you want to ask Rain about this code?",
    placeHolder: "e.g. What does this do? Are there any bugs?",
  });
  if (!question) return;

  const ctx = getEditorContext(editor, "selection");
  const prompt = buildCodePrompt(question, ctx);
  const result = await askRain(prompt);
  if (result) showInOutput(result);
}

async function cmdExplainSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Rain: Select some code first.");
    return;
  }
  const ctx = getEditorContext(editor, "selection");
  const prompt = buildCodePrompt(
    "Explain what this code does in plain English. Be concise and precise.",
    ctx
  );
  const result = await askRain(prompt);
  if (result) showInOutput(result);
}

async function cmdRefactorSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Rain: Select some code first.");
    return;
  }
  const instructions = await vscode.window.showInputBox({
    prompt: "How should Rain refactor this? (leave blank for general cleanup)",
    placeHolder: "e.g. Make it more readable, extract a helper function...",
  });

  const ctx = getEditorContext(editor, "selection");
  const baseInstruction = instructions
    ? `Refactor this code: ${instructions}`
    : "Refactor this code for clarity, correctness, and best practices. Preserve the original behaviour exactly.";
  const prompt = buildCodePrompt(baseInstruction, ctx);
  const result = await askRain(prompt, { sandbox: getSandbox() });
  if (result) showInOutput(result);
}

async function cmdFindBugs() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Rain: Select some code first.");
    return;
  }
  const ctx = getEditorContext(editor, "selection");
  const prompt = buildCodePrompt(
    "Find all bugs, edge cases, security issues, and logic errors in this code. " +
      "For each problem, explain what it is and how to fix it.",
    ctx
  );
  const result = await askRain(prompt);
  if (result) showInOutput(result);
}

async function cmdWriteTests() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Rain: Select some code first.");
    return;
  }
  const ctx = getEditorContext(editor, "selection");
  const prompt = buildCodePrompt(
    `Write thorough unit tests for this ${ctx.language} code. ` +
      "Cover happy paths, edge cases, and error conditions. " +
      "Use the standard testing framework for this language.",
    ctx
  );
  const result = await askRain(prompt, { sandbox: getSandbox() });
  if (result) showInOutput(result);
}

async function cmdAskAboutFile() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("Rain: Open a file first.");
    return;
  }
  const question = await vscode.window.showInputBox({
    prompt: "What do you want to ask Rain about this file?",
    placeHolder: "e.g. Summarise this file, are there any security issues?",
  });
  if (!question) return;

  const ctx = getEditorContext(editor, "file");

  // Warn if file is very large
  if (ctx.text.length > 80000) {
    const go = await vscode.window.showWarningMessage(
      `This file is large (${Math.round(ctx.text.length / 1024)}KB). Rain will try but it may be slow or truncated.`,
      "Continue",
      "Cancel"
    );
    if (go !== "Continue") return;
  }

  const prompt = buildCodePrompt(question, ctx);
  const result = await askRain(prompt);
  if (result) showInOutput(result);
}

async function cmdIndexProject(extContext) {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    vscode.window.showWarningMessage(
      "Rain: No workspace folder open. Open a project first."
    );
    return;
  }

  const choices = workspaceFolders.map((f) => ({
    label: f.name,
    description: f.uri.fsPath,
    path: f.uri.fsPath,
  }));

  const picked =
    choices.length === 1
      ? choices[0]
      : await vscode.window.showQuickPick(choices, {
          placeHolder: "Which workspace folder should Rain index?",
        });

  if (!picked) return;

  const alive = await checkServer();
  if (!alive) {
    vscode.window.showErrorMessage(
      `Rain server not reachable at ${getServerUrl()}. Make sure Rain is running: ./rain-web`
    );
    return;
  }

  vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `⛈️ Rain indexing ${picked.label}...`,
      cancellable: false,
    },
    async (progress) => {
      progress.report({
        message: "Embedding files with nomic-embed-text (this may take a while)",
      });
      try {
        const res = await httpPost(
          `${getServerUrl()}/api/index-project`,
          { project_path: picked.path, force: false },
          300000 // 5-min timeout for large projects
        );
        if (res.status === 200) {
          const s = res.body;
          vscode.window.showInformationMessage(
            `⛈️ Rain indexed ${picked.label}: ` +
              `${s.files_indexed} file(s), ${s.chunks_total} chunks in ${s.duration_s}s`
          );
        } else {
          vscode.window.showErrorMessage(
            `Rain index failed: ${res.body?.detail || res.status}`
          );
        }
      } catch (err) {
        vscode.window.showErrorMessage(`Rain index error: ${err.message}`);
      }
    }
  );
}

async function cmdSetServerUrl() {
  const current = getServerUrl();
  const value = await vscode.window.showInputBox({
    prompt: "Enter the Rain server URL",
    value: current,
    placeHolder: "http://localhost:7734",
  });
  if (value && value !== current) {
    await vscode.workspace
      .getConfiguration("rain")
      .update("serverUrl", value, vscode.ConfigurationTarget.Global);
    vscode.window.showInformationMessage(`Rain server URL set to: ${value}`);
  }
}

// ── Auto-index on workspace open ──────────────────────────────────────

async function maybeAutoIndex() {
  const autoIndex = vscode.workspace
    .getConfiguration("rain")
    .get("autoIndexOnOpen");
  if (!autoIndex) return;

  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) return;

  const alive = await checkServer();
  if (!alive) return;

  for (const folder of folders) {
    try {
      await httpPost(
        `${getServerUrl()}/api/index-project`,
        { project_path: folder.uri.fsPath, force: false },
        600000
      );
    } catch {
      /* silent — auto-index should never nag the user */
    }
  }
}

// ── Status bar ────────────────────────────────────────────────────────

let statusBarItem = null;

async function updateStatusBar() {
  if (!statusBarItem) return;
  const alive = await checkServer();
  if (alive) {
    statusBarItem.text = "$(cloud) Rain ⛈️";
    statusBarItem.tooltip = `Rain server running at ${getServerUrl()}`;
    statusBarItem.backgroundColor = undefined;
  } else {
    statusBarItem.text = "$(cloud-offline) Rain offline";
    statusBarItem.tooltip = `Rain server not reachable at ${getServerUrl()}. Run ./rain-web to start it.`;
    statusBarItem.backgroundColor = new vscode.ThemeColor(
      "statusBarItem.warningBackground"
    );
  }
}

// ── Activate / Deactivate ─────────────────────────────────────────────

function activate(context) {
  // Status bar
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBarItem.command = "rain.openChat";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  updateStatusBar();
  const statusInterval = setInterval(updateStatusBar, 30000);
  context.subscriptions.push({ dispose: () => clearInterval(statusInterval) });

  // Register commands
  const cmds = [
    vscode.commands.registerCommand("rain.askAboutSelection", cmdAskAboutSelection),
    vscode.commands.registerCommand("rain.explainSelection",  cmdExplainSelection),
    vscode.commands.registerCommand("rain.refactorSelection", cmdRefactorSelection),
    vscode.commands.registerCommand("rain.findBugs",          cmdFindBugs),
    vscode.commands.registerCommand("rain.writeTests",        cmdWriteTests),
    vscode.commands.registerCommand("rain.askAboutFile",      cmdAskAboutFile),
    vscode.commands.registerCommand("rain.openChat",         () => openChatPanel(context)),
    vscode.commands.registerCommand("rain.setServerUrl",      cmdSetServerUrl),
    vscode.commands.registerCommand("rain.indexProject",     () => cmdIndexProject(context)),
  ];
  cmds.forEach((c) => context.subscriptions.push(c));

  // Auto-index on startup (if configured)
  maybeAutoIndex();

  console.log("Rain ⛈️ extension activated");
}

function deactivate() {
  if (chatPanel) {
    chatPanel.dispose();
    chatPanel = null;
  }
}

module.exports = { activate, deactivate };
