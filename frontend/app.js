const STRATEGIES = {
    direct_llm: {
        label: 'Direct LLM',
        baselineSession: 'test-direct-llm',
    },
    sliding_window: {
        label: 'Sliding Window',
        baselineSession: 'test-sliding-window',
    },
    hierarchical: {
        label: 'Hierarchical',
        baselineSession: 'test-hierarchical',
    },
    entity_graph: {
        label: 'Entity Graph',
        baselineSession: 'test-entity-graph',
    },
};

const BASELINE_TURN_COUNT = 60;
const ALLOWED_TURN_DRIFT = 3;
const BASELINE_LOAD_COMMAND = 'python scripts/load_test_data.py --force';
const SAMPLE_QUESTIONS = [
    "What's the tech stack?",
    "Can you give me a summary of our project team and their responsibilities?",
    "What's the total budget including the additional ML cluster approval?",
    "Which integrations do we need to support for e-commerce platforms?",
    "What was Mary Doe's model accuracy for churn prediction?",
    "What's Tom Doe's estimated monthly AWS cost?",
    "Who are our pilot customers for the soft launch?",
    "What's our uptime SLA requirement?",
    "When is our first milestone deadline?",
];

let currentStrategy = 'sliding_window';
let tokenHistory = [];
let loadSessionsRequestId = 0;

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const questionSelect = document.getElementById('question-select');
const loadQuestionBtn = document.getElementById('load-question-btn');
const clearQuestionBtn = document.getElementById('clear-question-btn');
const sessionSelect = document.getElementById('session-select');
const strategyBtns = document.querySelectorAll('.strategy-btn');
const newSessionBtn = document.getElementById('new-session-btn');
const viewMemoryBtn = document.getElementById('view-memory-btn');
const deleteSessionBtn = document.getElementById('delete-session-btn');
const deleteAllBtn = document.getElementById('delete-all-btn');
const resetBaselineBtn = document.getElementById('reset-baseline-btn');
const appStatus = document.getElementById('app-status');
const baselineNote = document.getElementById('baseline-note');
const memoryPanel = document.getElementById('memory-panel');
const memoryContent = document.getElementById('memory-content');
const closeMemoryBtn = document.getElementById('close-memory');
const copyMemoryBtn = document.getElementById('copy-memory');

function currentStrategyConfig() {
    return STRATEGIES[currentStrategy];
}

function selectedSessionStorageKey() {
    return `selected_session_${currentStrategy}`;
}

function getSessionId() {
    return sessionSelect.value || null;
}

function newSessionId() {
    return `session-${Date.now().toString(36)}`;
}

function setStatus(message, level = '') {
    appStatus.textContent = message || '';
    appStatus.className = 'app-status';
    if (level) appStatus.classList.add(level);
}

function setBaselineNote(message, level = 'warning') {
    if (!message) {
        baselineNote.hidden = true;
        baselineNote.textContent = '';
        baselineNote.className = 'inline-note';
        return;
    }
    baselineNote.hidden = false;
    baselineNote.textContent = message;
    baselineNote.className = 'inline-note';
    baselineNote.classList.add(level);
}

function normalizeTurnCount(value) {
    const turnCount = Number(value);
    if (!Number.isFinite(turnCount) || turnCount < 0) return 0;
    return Math.floor(turnCount);
}

function formatDriftDelta(delta) {
    return delta > 0 ? `+${delta}` : String(delta);
}

function getSeededSessionState(sessionId, turnCount) {
    const normalizedTurnCount = normalizeTurnCount(turnCount);
    const driftDelta = normalizedTurnCount - BASELINE_TURN_COUNT;

    if (normalizedTurnCount < BASELINE_TURN_COUNT) {
        return {
            label: `${normalizedTurnCount}/${BASELINE_TURN_COUNT} incomplete`,
            level: 'warning',
            message: `${sessionId} is incomplete at ${normalizedTurnCount}/${BASELINE_TURN_COUNT} user turns. The seeded demo profile is expected to stay at exactly ${BASELINE_TURN_COUNT}, so recall results may be wrong until you reset to ${BASELINE_TURN_COUNT}.`,
        };
    }

    if (driftDelta === 0) {
        return {
            label: 'clean baseline',
            level: '',
            message: '',
        };
    }

    if (driftDelta > ALLOWED_TURN_DRIFT) {
        return {
            label: `${formatDriftDelta(driftDelta)} hard drift`,
            level: 'danger',
            message: `Seeded dataset drifted: ${sessionId} is at ${normalizedTurnCount}/${BASELINE_TURN_COUNT} user turns (${formatDriftDelta(driftDelta)}). Chat is still enabled, but recall results may be wrong until you reset to ${BASELINE_TURN_COUNT}.`,
        };
    }

    return {
        label: `${formatDriftDelta(driftDelta)} soft drift`,
        level: 'warning',
        message: `${sessionId} is at ${normalizedTurnCount}/${BASELINE_TURN_COUNT} user turns (${formatDriftDelta(driftDelta)}). Chat is still enabled, but results may start to skew because the seeded comparison point has drifted. Reset to ${BASELINE_TURN_COUNT} for clean comparisons.`,
    };
}

function formatSessionOptionLabel(session, baselineSessionId) {
    const turnCount = normalizeTurnCount(session.turn_count);
    if (session.session_id !== baselineSessionId) {
        return `${session.session_id} (${turnCount} turns)`;
    }

    const seededState = getSeededSessionState(session.session_id, turnCount);
    return `${session.session_id} (${turnCount} turns - ${seededState.label})`;
}

function resetLocalPanels() {
    tokenHistory = [];
    updateTokenChart();
    resetMetrics();
    memoryPanel.style.display = 'none';
}

function renderWelcomeMessage() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <p>Select a memory strategy and a session.</p>
            <p>Tip: click <strong>Reset to 60</strong> to seed the canonical baseline conversation, then load a sample question or type your own.</p>
        </div>
    `;
}

function populateQuestionSelect() {
    if (!questionSelect) return;

    questionSelect.innerHTML = '<option value="">-- Select a recall question --</option>';
    SAMPLE_QUESTIONS.forEach((question) => {
        const option = document.createElement('option');
        option.value = question;
        option.textContent = question;
        questionSelect.appendChild(option);
    });
}

function loadSelectedQuestion() {
    const question = questionSelect?.value?.trim() || '';
    if (!question) {
        setStatus('Select a question to load first.', 'error');
        return;
    }

    chatInput.value = question;
    chatInput.focus();
    chatInput.setSelectionRange(question.length, question.length);
    setStatus('Loaded question into the input.', 'success');
}

function clearQuestionComposer() {
    if (questionSelect) questionSelect.value = '';
    chatInput.value = '';
    chatInput.focus();
    setStatus('Cleared the question input.', 'success');
}

function updateBaselineWarning(sessions) {
    const cfg = currentStrategyConfig();
    const baseline = sessions.find((session) => session.session_id === cfg.baselineSession);

    if (sessions.length === 0) {
        setBaselineNote(
            `No sessions found for ${cfg.label}. Load the 60-turn baseline with "${BASELINE_LOAD_COMMAND}", or click New to start an empty session.`,
        );
        return;
    }

    if (!baseline) {
        setBaselineNote(
            `The canonical baseline session for ${cfg.label} is missing. Run "${BASELINE_LOAD_COMMAND}" if you want a shared 60-turn comparison point.`,
            'info',
        );
        return;
    }

    const seededState = getSeededSessionState(cfg.baselineSession, baseline.turn_count);
    if (seededState.message) {
        setBaselineNote(seededState.message, seededState.level || 'warning');
        return;
    }

    setBaselineNote('');
}

async function loadSessions(preferredSessionId = '') {
    const requestId = ++loadSessionsRequestId;
    const previousSelection = preferredSessionId || getSessionId() || localStorage.getItem(selectedSessionStorageKey()) || '';
    sessionSelect.innerHTML = '<option value="">Loading sessions...</option>';

    try {
        const resp = await fetch(`/api/sessions/${currentStrategy}`, { cache: 'no-store' });
        if (!resp.ok) {
            let detail = '';
            try {
                const body = await resp.json();
                detail = String(body?.detail || '');
            } catch (_error) {
                detail = '';
            }
            throw new Error(detail || `Failed to load sessions (${resp.status})`);
        }

        const sessionsRaw = await resp.json();
        if (requestId !== loadSessionsRequestId) return;

        const sessions = Array.isArray(sessionsRaw) ? sessionsRaw : [];
        const cfg = currentStrategyConfig();
        sessionSelect.innerHTML = '<option value="">-- Select Session --</option>';

        sessions.forEach((session) => {
            const option = document.createElement('option');
            option.value = session.session_id;
            option.textContent = formatSessionOptionLabel(session, cfg.baselineSession);
            sessionSelect.appendChild(option);
        });
        const hasPreferred = previousSelection && sessions.some((session) => session.session_id === previousSelection);
        const hasBaseline = sessions.some((session) => session.session_id === cfg.baselineSession);

        if (hasPreferred) {
            sessionSelect.value = previousSelection;
        } else if (hasBaseline) {
            sessionSelect.value = cfg.baselineSession;
        } else if (sessions.length > 0) {
            sessionSelect.value = sessions[0].session_id;
        }

        if (sessionSelect.value) {
            localStorage.setItem(selectedSessionStorageKey(), sessionSelect.value);
            await loadMemoryState();
        } else {
            localStorage.removeItem(selectedSessionStorageKey());
            memoryPanel.style.display = 'none';
        }

        updateBaselineWarning(sessions);
        setStatus(`Loaded ${sessions.length} session${sessions.length === 1 ? '' : 's'} for ${cfg.label}.`, 'success');
    } catch (error) {
        console.error('Failed to load sessions:', error);
        sessionSelect.innerHTML = '<option value="">⚠️ Error loading sessions</option>';
        setBaselineNote(
            `Sessions could not be loaded. Make sure the server is running and Cosmos is reachable. You can seed the baseline with "${BASELINE_LOAD_COMMAND}" after startup.`,
        );
        setStatus(String(error?.message || 'Failed to load sessions.'), 'error');
    }
}

async function loadMemoryState() {
    const sessionId = getSessionId();
    if (!sessionId) {
        memoryPanel.style.display = 'none';
        return;
    }

    try {
        const resp = await fetch(`/api/memory/${currentStrategy}/${sessionId}`, { cache: 'no-store' });
        if (!resp.ok) {
            throw new Error(`Failed to load memory (${resp.status})`);
        }

        const data = await resp.json();
        data.strategy = currentStrategy;
        data.session_id = sessionId;

        const hasData = data.turn_count > 0
            || (Array.isArray(data.entities) && data.entities.length > 0)
            || (Array.isArray(data.recent_turns) && data.recent_turns.length > 0)
            || (Array.isArray(data.tier1) && data.tier1.length > 0);

        if (!hasData) {
            memoryPanel.style.display = 'none';
            return;
        }

        memoryContent.textContent = JSON.stringify(data, null, 2);
        memoryPanel.style.display = 'block';
    } catch (_error) {
        memoryPanel.style.display = 'none';
    }
}

function addMessage(text, className) {
    const div = document.createElement('div');
    div.className = `message ${className}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function clearChat() {
    renderWelcomeMessage();
    resetMetrics();
}

function updateMetrics(metrics) {
    document.getElementById('metric-prompt-tokens').textContent = Number(metrics.prompt_tokens || 0).toLocaleString();
    document.getElementById('metric-completion-tokens').textContent = Number(metrics.completion_tokens || 0).toLocaleString();
    document.getElementById('metric-total-tokens').textContent = Number(metrics.total_tokens || 0).toLocaleString();
    document.getElementById('metric-latency').textContent = `${Number(metrics.latency_ms || 0).toFixed(0)}ms`;
    document.getElementById('metric-turns-stored').textContent = Number(metrics.memory_turns_stored || 0).toLocaleString();
    document.getElementById('metric-context-sent').textContent = Number(metrics.context_turns_sent || 0).toLocaleString();
}

function resetMetrics() {
    ['prompt-tokens', 'completion-tokens', 'total-tokens', 'latency', 'turns-stored', 'context-sent'].forEach((id) => {
        document.getElementById(`metric-${id}`).textContent = '-';
    });
}

function updateTokenChart() {
    const container = document.getElementById('token-chart-bars');
    container.innerHTML = '';

    if (!tokenHistory.length) return;

    const max = Math.max(...tokenHistory);
    tokenHistory.forEach((value) => {
        const bar = document.createElement('div');
        bar.className = 'chart-bar';
        bar.style.height = `${Math.max((value / max) * 70, 4)}px`;
        bar.dataset.value = Number(value).toLocaleString();
        container.appendChild(bar);
    });
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    const sessionId = getSessionId();
    if (!sessionId) {
        alert('Please select or create a session first.');
        return;
    }

    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    addMessage(message, 'user');
    chatInput.value = '';
    sendBtn.disabled = true;
    setStatus(`Sending message to ${currentStrategyConfig().label}...`, 'loading');

    const loadingEl = addMessage('Thinking...', 'assistant loading');

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                message,
                strategy: currentStrategy,
            }),
        });

        let body = {};
        try {
            body = await resp.json();
        } catch (_error) {
            body = {};
        }

        if (!resp.ok) {
            throw new Error(body?.detail || `Request failed (${resp.status})`);
        }

        loadingEl.remove();
        addMessage(String(body.reply || ''), 'assistant');

        const metrics = body.metrics || {};
        updateMetrics(metrics);
        tokenHistory.push(Number(metrics.total_tokens || 0));
        if (tokenHistory.length > 20) tokenHistory.shift();
        updateTokenChart();

        await loadMemoryState();
        await loadSessions(sessionId);
        setStatus(`Reply received from ${currentStrategyConfig().label}.`, 'success');
    } catch (error) {
        loadingEl.remove();
        addMessage(`Error: ${String(error?.message || 'Request failed.')}`, 'assistant');
        setStatus(String(error?.message || 'Request failed.'), 'error');
    } finally {
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

async function resetBaseline() {
    const warning = [
        `Reload the canonical ${BASELINE_TURN_COUNT}-turn baseline sessions?`,
        '',
        '- This reseeds all four strategies.',
        '- This can take several minutes.',
        '- This uses live OpenAI calls unless MOCK_OPENAI=true.',
    ].join('\n');

    if (!confirm(warning)) return;

    const originalText = resetBaselineBtn.textContent;
    resetBaselineBtn.textContent = 'Resetting...';
    resetBaselineBtn.disabled = true;
    setStatus(`Resetting the canonical ${BASELINE_TURN_COUNT}-turn baseline...`, 'loading');

    try {
        const startResp = await fetch('/api/baseline/reset', {
            method: 'POST',
            cache: 'no-store',
        });
        const startBody = await startResp.json();
        if (!startResp.ok) {
            throw new Error(startBody?.detail || `Reset failed (${startResp.status})`);
        }

        const jobId = String(startBody?.job_id || '');
        if (!jobId) {
            throw new Error('Reset did not return a job id.');
        }

        const pollStart = Date.now();
        const timeoutMs = 45 * 60 * 1000;
        while (true) {
            if (Date.now() - pollStart > timeoutMs) {
                throw new Error('Baseline reset timed out while waiting for completion.');
            }

            const pollResp = await fetch(`/api/baseline/reset/${encodeURIComponent(jobId)}`, {
                cache: 'no-store',
            });
            const pollBody = await pollResp.json();
            if (!pollResp.ok) {
                throw new Error(pollBody?.detail || `Reset status check failed (${pollResp.status})`);
            }

            const status = String(pollBody?.status || '').toLowerCase();
            if (status === 'completed') break;
            if (status === 'failed') {
                throw new Error(pollBody?.message || 'Baseline reset job failed.');
            }

            setStatus('Baseline reset running... this can take several minutes.', 'loading');
            await new Promise((resolve) => setTimeout(resolve, 3000));
        }

        clearChat();
        resetLocalPanels();
        await loadSessions(currentStrategyConfig().baselineSession);
        setStatus(`Canonical ${BASELINE_TURN_COUNT}-turn baseline is ready.`, 'success');
    } catch (error) {
        const message = String(error?.message || 'Unknown error during reset.');
        setStatus(message, 'error');
        alert(`Baseline reset failed:\n${message}`);
    } finally {
        resetBaselineBtn.textContent = originalText;
        resetBaselineBtn.disabled = false;
    }
}

async function copyMemoryToClipboard() {
    const text = memoryContent.textContent || '';
    if (!text) {
        setStatus('No memory state is available to copy.', 'error');
        return;
    }

    try {
        await navigator.clipboard.writeText(text);
        copyMemoryBtn.classList.add('copied');
        setStatus('Copied memory state to the clipboard.', 'success');
        window.setTimeout(() => copyMemoryBtn.classList.remove('copied'), 1200);
    } catch (_error) {
        setStatus('Clipboard copy failed in this browser context.', 'error');
    }
}

strategyBtns.forEach((button) => {
    button.addEventListener('click', async () => {
        strategyBtns.forEach((candidate) => candidate.classList.remove('active'));
        button.classList.add('active');
        currentStrategy = button.dataset.strategy;
        clearChat();
        resetLocalPanels();
        setStatus(`Switched to ${currentStrategyConfig().label}.`, 'success');
        await loadSessions();
    });
});

sessionSelect.addEventListener('change', async () => {
    const sessionId = getSessionId();
    if (!sessionId) {
        localStorage.removeItem(selectedSessionStorageKey());
        clearChat();
        resetLocalPanels();
        return;
    }

    localStorage.setItem(selectedSessionStorageKey(), sessionId);
    clearChat();
    resetLocalPanels();
    await loadMemoryState();
    setStatus(`Selected session ${sessionId}.`, 'success');
});

newSessionBtn.addEventListener('click', () => {
    const sessionId = newSessionId();
    const option = document.createElement('option');
    option.value = sessionId;
    option.textContent = `${sessionId} (0 turns)`;
    sessionSelect.appendChild(option);
    sessionSelect.value = sessionId;
    localStorage.setItem(selectedSessionStorageKey(), sessionId);
    clearChat();
    resetLocalPanels();
    memoryPanel.style.display = 'none';
    setStatus(`Created session ${sessionId}.`, 'success');
});

viewMemoryBtn.addEventListener('click', loadMemoryState);

deleteSessionBtn.addEventListener('click', async () => {
    const sessionId = getSessionId();
    if (!sessionId) {
        alert('No session selected.');
        return;
    }

    if (!confirm(`Delete session "${sessionId}"? This permanently removes all stored data for this session.`)) {
        return;
    }

    try {
        const resp = await fetch(`/api/sessions/${currentStrategy}/${encodeURIComponent(sessionId)}`, {
            method: 'DELETE',
        });
        if (!resp.ok) {
            const body = await resp.json();
            throw new Error(body?.detail || `Delete failed (${resp.status})`);
        }

        localStorage.removeItem(selectedSessionStorageKey());
        clearChat();
        resetLocalPanels();
        await loadSessions();
        setStatus(`Deleted session ${sessionId}.`, 'success');
    } catch (error) {
        const message = String(error?.message || 'Delete failed.');
        setStatus(message, 'error');
        alert(`Error deleting session:\n${message}`);
    }
});

deleteAllBtn.addEventListener('click', async () => {
    const strategyLabel = currentStrategyConfig().label;
    if (!confirm(`Delete all ${strategyLabel} sessions? This removes every session for the current strategy.`)) {
        return;
    }
    if (!confirm(`Delete all ${strategyLabel} sessions permanently? This cannot be undone.`)) {
        return;
    }

    try {
        const resp = await fetch(`/api/sessions/${currentStrategy}`, { cache: 'no-store' });
        if (!resp.ok) {
            const body = await resp.json();
            throw new Error(body?.detail || `Load failed (${resp.status})`);
        }

        const sessions = await resp.json();
        for (const session of (Array.isArray(sessions) ? sessions : [])) {
            await fetch(`/api/sessions/${currentStrategy}/${encodeURIComponent(session.session_id)}`, {
                method: 'DELETE',
            });
        }

        localStorage.removeItem(selectedSessionStorageKey());
        clearChat();
        resetLocalPanels();
        await loadSessions();
        setStatus(`Deleted ${Array.isArray(sessions) ? sessions.length : 0} ${strategyLabel} session(s).`, 'success');
    } catch (error) {
        const message = String(error?.message || 'Delete failed.');
        setStatus(message, 'error');
        alert(`Error deleting sessions:\n${message}`);
    }
});

resetBaselineBtn.addEventListener('click', resetBaseline);
sendBtn.addEventListener('click', sendMessage);
loadQuestionBtn.addEventListener('click', loadSelectedQuestion);
clearQuestionBtn.addEventListener('click', clearQuestionComposer);
closeMemoryBtn.addEventListener('click', () => {
    memoryPanel.style.display = 'none';
});
copyMemoryBtn.addEventListener('click', copyMemoryToClipboard);
chatInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

populateQuestionSelect();
clearChat();
loadSessions();
