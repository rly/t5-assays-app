// AG Grid instance
let gridApi = null;

// Load data for a specific dataset and initialize grid
function loadData(datasetKey) {
    if (!datasetKey) return;

    const gridDiv = document.getElementById("data-grid");
    const rowCount = document.getElementById("row-count");
    if (gridDiv) {
        gridDiv.innerHTML = '<div class="grid-loading"><div class="grid-loading-spinner"></div>Loading data...</div>';
    }
    if (rowCount) rowCount.textContent = "";

    fetch(`/api/data?dataset_key=${encodeURIComponent(datasetKey)}`)
        .then(r => r.json())
        .then(data => {
            if (!gridDiv) return;

            const colDescs = data.column_descriptions || {};
            const columnDefs = data.columns.map(col => {
                const desc = colDescs[col];
                const def = {
                    field: col,
                    sortable: true,
                    filter: true,
                    resizable: true,
                    pinned: col === "Name" ? "left" : null,
                    width: 120,
                    tooltipField: col,
                    headerTooltip: desc ? `${col}: ${desc}` : col,
                };
                // Make Structure column clickable to show 2D molecule
                if (col === "Structure") {
                    def.cellRenderer = function(params) {
                        if (!params.value) return "";
                        const link = document.createElement("a");
                        link.href = "#";
                        link.textContent = params.value.substring(0, 20) + (params.value.length > 20 ? "..." : "");
                        link.title = "Click to view structure";
                        link.style.color = "var(--pico-primary)";
                        link.style.cursor = "pointer";
                        link.onclick = function(e) {
                            e.preventDefault();
                            showStructure(params.value, params.data.Name || "");
                        };
                        return link;
                    };
                }
                return def;
            });

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

            gridDiv.innerHTML = "";
            gridApi = agGrid.createGrid(gridDiv, gridOptions);

            if (rowCount) {
                rowCount.textContent = `${data.rows.length} rows x ${data.columns.length} cols`;
            }
        })
        .catch(err => {
            console.error("Error loading data:", err);
            if (gridDiv) gridDiv.innerHTML = "<p>Error loading data.</p>";
        });
}

// Load plots for a specific dataset (wait for Plotly)
function loadPlots(datasetKey) {
    if (!datasetKey) return;
    if (typeof Plotly === "undefined") {
        setTimeout(() => loadPlots(datasetKey), 200);
        return;
    }
    const container = document.getElementById("plots-container");
    if (container) {
        htmx.ajax("GET", `/plots?dataset_key=${encodeURIComponent(datasetKey)}`, {
            target: "#plots-container", swap: "innerHTML"
        });
    }
}

// Called after clicking "View" on a dataset
function onDatasetView(datasetKey) {
    // The server response's <script> sets window.APP_CONFIG.viewingDataset.
    // Wait a tick for that to execute, then update button states.
    setTimeout(function() {
        var viewing = window.APP_CONFIG.viewingDataset;
        document.querySelectorAll(".ds-view-btn").forEach(function(btn) {
            var btnKey = btn.getAttribute("hx-post").replace("/datasets/", "").replace("/view", "");
            if (btnKey === viewing) {
                btn.textContent = "Viewing";
                btn.classList.remove("outline");
                btn.classList.add("ds-view-active");
                btn.closest("tr").classList.add("ds-viewing");
            } else {
                btn.textContent = "View";
                btn.classList.add("outline");
                btn.classList.remove("ds-view-active");
                btn.closest("tr").classList.remove("ds-viewing");
            }
        });
    }, 50);
}

// Submit chat on Enter (Shift+Enter for newline)
document.addEventListener("DOMContentLoaded", function() {
    const ta = document.getElementById("chat-input");
    if (ta) {
        ta.addEventListener("keydown", function(e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                document.getElementById("chat-form").requestSubmit();
            }
        });
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
        // Suggestion chip — extract message from hx-vals
        try {
            const vals = JSON.parse(event.target.getAttribute("hx-vals") || "{}");
            msg = vals.message || "";
        } catch(e) {}
    }

    if (msg) {
        appendChatMessage("user", msg);
    }

    // Show loading indicator with timer
    const loading = document.createElement("div");
    loading.className = "chat-loading";
    loading.id = "chat-loading";
    loading.innerHTML = '<span class="chat-loading-text">Thinking...</span><span class="chat-loading-timer"></span>';
    document.getElementById("chat-messages").appendChild(loading);
    scrollChat();

    const startTime = Date.now();
    const statuses = ["Thinking...", "Analyzing data...", "Running computations...", "Generating response..."];
    let statusIdx = 0;
    window._chatTimer = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        const timer = loading.querySelector(".chat-loading-timer");
        const text = loading.querySelector(".chat-loading-text");
        if (timer) timer.textContent = `${elapsed}s`;
        if (elapsed % 5 === 0 && elapsed > 0 && statusIdx < statuses.length - 1) {
            statusIdx++;
            if (text) text.textContent = statuses[statusIdx];
        }
    }, 1000);

    if (btn) btn.disabled = true;
    if (input) {
        input.disabled = true;
        setTimeout(() => { input.value = ""; }, 0);
    }
}

function onChatDone(event) {
    if (window._chatTimer) {
        clearInterval(window._chatTimer);
        window._chatTimer = null;
    }
    const loading = document.getElementById("chat-loading");
    if (loading) loading.remove();

    const input = document.getElementById("chat-input");
    const btn = document.getElementById("chat-submit-btn");
    if (btn) btn.disabled = false;
    if (input) {
        input.disabled = false;
        input.value = "";
        input.focus();
    }

    renderMarkdown();
    renderPlots();
    setTimeout(scrollChat, 100);
}

function appendChatMessage(role, content) {
    const container = document.getElementById("chat-messages");
    // Remove empty state if present
    const empty = container.querySelector(".chat-empty-state");
    if (empty) empty.remove();

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

function renderPlots() {
    if (typeof Plotly === 'undefined') return;
    document.querySelectorAll('.plotly-output[data-plotly]:not([data-rendered])').forEach(el => {
        try {
            const fig = JSON.parse(el.getAttribute('data-plotly'));
            const layout = Object.assign({ margin: { t: 40, b: 60, l: 60, r: 20 } }, fig.layout || {});
            Plotly.react(el, fig.data || [], layout, { responsive: true });
            el.setAttribute('data-rendered', 'true');
        } catch (e) {
            console.error('Plotly render error:', e);
        }
    });
}

function renderMarkdown() {
    document.querySelectorAll(".markdown-content").forEach(el => {
        if (el.dataset.rendered) return;
        const raw = el.textContent;
        if (typeof marked !== "undefined") {
            el.innerHTML = marked.parse(raw);
            // Wrap python code blocks in collapsible <details>
            el.querySelectorAll("pre").forEach(pre => {
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

// Structure viewer
let smilesDrawer = null;

function showStructure(smiles, compoundName) {
    if (!smiles) return;
    const popup = document.getElementById("structure-popup");
    const title = document.getElementById("structure-popup-title");

    title.textContent = compoundName || smiles.substring(0, 30);
    popup.style.display = "block";

    // Initialize SmilesDrawer v1 if needed
    if (!smilesDrawer && typeof SmilesDrawer !== "undefined") {
        smilesDrawer = new SmilesDrawer.Drawer({width: 400, height: 300});
    }

    if (smilesDrawer) {
        SmilesDrawer.parse(smiles, function(tree) {
            smilesDrawer.draw(tree, "structure-canvas", "light", false);
        }, function(err) {
            console.error("SMILES parse error:", err);
        });
    }
}

function hideStructure() {
    document.getElementById("structure-popup").style.display = "none";
}

// Close popup on outside click
document.addEventListener("click", function(e) {
    const popup = document.getElementById("structure-popup");
    if (popup && popup.style.display === "block" && !popup.contains(e.target)) {
        // Don't close if clicking a structure link
        if (!e.target.closest("[onclick*='showStructure']") && !e.target.onclick) {
            hideStructure();
        }
    }
});

// Initialize on page load
document.addEventListener("DOMContentLoaded", function() {
    const key = window.APP_CONFIG.viewingDataset;
    if (key) {
        loadData(key);
        loadPlots(key);
    }
    renderMarkdown();
});
