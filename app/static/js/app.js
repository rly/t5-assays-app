// AG Grid instance
let gridApi = null;

// Load data and initialize grid
function loadData() {
    const gridDiv = document.getElementById("data-grid");
    const rowCount = document.getElementById("row-count");
    if (gridDiv) {
        gridDiv.innerHTML = '<div class="grid-loading"><div class="grid-loading-spinner"></div>Loading data...</div>';
    }
    if (rowCount) rowCount.textContent = "";

    fetch("/api/data")
        .then(r => r.json())
        .then(data => {
            if (!gridDiv) return;

            const columnDefs = data.columns.map(col => ({
                field: col,
                sortable: true,
                filter: true,
                resizable: true,
                pinned: col === "Name" ? "left" : null,
                width: 120,
                tooltipField: col,
            }));

            const myTheme = agGrid.themeQuartz.withParams({
                fontSize: 12,
                headerFontSize: 12,
                rowHeight: 28,
                headerHeight: 32,
                cellHorizontalPadding: 6,
            });

            const gridOptions = {
                theme: myTheme,
                columnDefs: columnDefs,
                rowData: data.rows,
                domLayout: "normal",
                defaultColDef: {
                    flex: 0,
                    minWidth: 60,
                    filter: "agTextColumnFilter",
                },
                autoSizeStrategy: {
                    type: "fitCellContents",
                    skipHeader: false,
                },
                tooltipShowDelay: 300,
                enableCellTextSelection: true,
            };

            // Clear existing grid
            gridDiv.innerHTML = "";
            gridApi = agGrid.createGrid(gridDiv, gridOptions);

            // Update row count
            const rowCount = document.getElementById("row-count");
            if (rowCount) {
                rowCount.textContent = `${data.rows.length} rows x ${data.columns.length} cols`;
            }

        })
        .catch(err => {
            console.error("Error loading data:", err);
            const gridDiv = document.getElementById("data-grid");
            if (gridDiv) gridDiv.innerHTML = "<p>Error loading data. Check console.</p>";
        });
}

function updateContextSize(rows, cols) {
    const el = document.getElementById("context-size");
    if (el) {
        const estimatedTokens = Math.round(rows * cols * 10);
        el.textContent = `Context: ${rows} rows x ${cols} cols (~${estimatedTokens.toLocaleString()} tokens)`;
    }
}

// Reload plots (wait for Plotly to be available)
function loadPlots() {
    if (typeof Plotly === "undefined") {
        setTimeout(loadPlots, 200);
        return;
    }
    htmx.ajax("GET", "/plots", {target: "#plots-container", swap: "innerHTML"});
}

// Full reload of data + plots
function reloadAll() {
    loadData();
    loadPlots();
}

// Listen for HTMX events from sidebar actions that should trigger reload
document.addEventListener("htmx:afterRequest", function(evt) {
    const path = evt.detail.pathInfo?.requestPath || "";
    if (path === "/data/source" || path === "/data/filters") {
        // Reload the sidebar too for filter visibility changes
        if (path === "/data/source") {
            fetch("/partials/sidebar")
                .then(r => r.text())
                .then(html => {
                    document.getElementById("sidebar").innerHTML = html;
                    htmx.process(document.getElementById("sidebar"));
                });
        }
        reloadAll();
    }
});

// Chat functions
function onChatSend(event) {
    const input = document.getElementById("chat-input");
    const btn = document.getElementById("chat-submit-btn");
    let msg = "";

    if (event.target.id === "chat-form") {
        msg = input ? input.value.trim() : "";
        if (!msg) {
            event.preventDefault();
            return;
        }
    } else {
        // Suggestion chip or other button — extract message from hx-vals
        try {
            const vals = JSON.parse(event.target.getAttribute("hx-vals") || "{}");
            msg = vals.message || "";
        } catch(e) {}
    }

    // Show user message immediately
    if (msg) {
        appendChatMessage("user", msg);
    }

    // Show loading indicator with elapsed timer
    const loading = document.createElement("div");
    loading.className = "chat-loading";
    loading.id = "chat-loading";
    loading.innerHTML = '<span class="chat-loading-text">Thinking...</span><span class="chat-loading-timer"></span>';
    document.getElementById("chat-messages").appendChild(loading);
    scrollChat();

    // Update timer and cycle status messages
    const startTime = Date.now();
    const statuses = ["Thinking...", "Analyzing data...", "Running computations...", "Generating response..."];
    let statusIdx = 0;
    window._chatTimer = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        const timer = loading.querySelector(".chat-loading-timer");
        const text = loading.querySelector(".chat-loading-text");
        if (timer) timer.textContent = `${elapsed}s`;
        // Cycle status every 5 seconds
        if (elapsed % 5 === 0 && elapsed > 0 && statusIdx < statuses.length - 1) {
            statusIdx++;
            if (text) text.textContent = statuses[statusIdx];
        }
    }, 1000);

    // Disable input
    if (btn) btn.disabled = true;
    if (input) {
        input.disabled = true;
        setTimeout(() => { input.value = ""; }, 0);
    }
}

function onChatDone(event) {
    // Clear timer and remove loading indicator
    if (window._chatTimer) {
        clearInterval(window._chatTimer);
        window._chatTimer = null;
    }
    const loading = document.getElementById("chat-loading");
    if (loading) loading.remove();

    // Re-enable input
    const input = document.getElementById("chat-input");
    const btn = document.getElementById("chat-submit-btn");
    if (btn) btn.disabled = false;
    if (input) {
        input.disabled = false;
        input.value = "";
        input.focus();
    }

    // Render markdown in new messages, then scroll after DOM settles
    renderMarkdown();
    setTimeout(scrollChat, 100);
}

function appendChatMessage(role, content) {
    const container = document.getElementById("chat-messages");
    const div = document.createElement("div");
    div.className = `chat-msg chat-${role}`;
    div.innerHTML = `
        <div class="chat-msg-header">${role === "user" ? "You" : "AI"}</div>
        <div class="chat-msg-body">${role === "assistant" ? '<div class="markdown-content">' + content + '</div>' : escapeHtml(content)}</div>
    `;
    container.appendChild(div);
    scrollChat();
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function scrollChat() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function renderMarkdown() {
    document.querySelectorAll(".markdown-content").forEach(el => {
        if (el.dataset.rendered) return;
        const raw = el.textContent;
        if (typeof marked !== "undefined") {
            el.innerHTML = marked.parse(raw);
            // Wrap python code blocks in collapsible <details>
            el.querySelectorAll("pre").forEach(pre => {
                const code = pre.querySelector("code");
                // Skip result/output blocks — only collapse code
                const text = pre.textContent.trim();
                if (text.startsWith("import ") || text.startsWith("# ") || text.startsWith("df") ||
                    text.includes("print(") || text.includes("= df") || text.includes(".mean(")) {
                    const details = document.createElement("details");
                    const summary = document.createElement("summary");
                    summary.textContent = "Show code";
                    details.appendChild(summary);
                    pre.parentNode.insertBefore(details, pre);
                    details.appendChild(pre);
                }
            });
        }
        el.dataset.rendered = "true";
    });
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", function() {
    loadData();
    loadPlots();
    renderMarkdown();
});
