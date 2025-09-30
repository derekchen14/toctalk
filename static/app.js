let currentSessionId = null;
let currentSession = null;
let isStreaming = false;

const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const newSessionBtn = document.getElementById('newSessionBtn');
const quirkbotBtn = document.getElementById('quirkbotBtn');
const sessionInfo = document.getElementById('sessionInfo');
const sessionsList = document.getElementById('sessionsList');
const loading = document.getElementById('loading');
let biographyFiles = [];

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
}

async function loadSessions() {
    try {
        const response = await fetch('/api/sessions');
        const sessions = await response.json();

        sessionsList.innerHTML = '';

        if (sessions.length === 0) {
            sessionsList.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No sessions yet</div>';
            return;
        }

        sessions.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));

        sessions.forEach(session => {
            const sessionItem = document.createElement('div');
            sessionItem.className = 'session-item';
            if (session.session_id === currentSessionId) {
                sessionItem.classList.add('active');
            }

            const title = session.description || `Session ${session.session_id.substring(0, 8)}`;
            const messageText = session.message_count === 1 ? '1 message' : `${session.message_count} messages`;

            sessionItem.innerHTML = `
                <div class="session-item-title">${title}</div>
                <div class="session-item-meta">${messageText} • ${formatDate(session.updated_at)}</div>
            `;

            sessionItem.onclick = () => loadSession(session.session_id);
            sessionsList.appendChild(sessionItem);
        });
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

async function loadSession(sessionId) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}`);
        if (!response.ok) {
            throw new Error('Session not found');
        }

        currentSession = await response.json();
        currentSessionId = sessionId;

        displaySession();
        updateSessionInfo();
        enableChat();

        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.remove('active');
        });

        const activeItem = Array.from(document.querySelectorAll('.session-item')).find(
            item => item.onclick.toString().includes(sessionId)
        );
        if (activeItem) {
            activeItem.classList.add('active');
        }
    } catch (error) {
        console.error('Error loading session:', error);
        showError('Failed to load session');
    }
}

function displaySession() {
    if (!currentSession) return;

    messagesDiv.innerHTML = '';

    if (currentSession.messages.length === 0) {
        messagesDiv.innerHTML = '<div style="text-align: center; color: #7f8c8d; padding: 50px;">Type a message to start chatting.</div>';
        return;
    }

    currentSession.messages.forEach(msg => {
        addMessageToUI(msg.content, msg.role);
    });

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addMessageToUI(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const label = role === 'user' ? 'You' : 'Assistant';

    messageDiv.innerHTML = `
        <div>
            <div class="message-label">${label}</div>
            <div class="message-content">${escapeHtml(content)}</div>
        </div>
    `;

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateSessionInfo() {
    if (currentSessionId) {
        if (currentSession && currentSession.metadata && currentSession.metadata.is_quirkbot) {
            const name = currentSession.metadata.persona_name || 'Unknown';
            const age = currentSession.metadata.persona_age || 'Unknown';
            sessionInfo.innerHTML = `<strong>Quirkbot:</strong> ${name}, ${age}`;
            sessionInfo.style.color = '#9b59b6';
        } else {
            const shortId = currentSessionId.substring(0, 8);
            sessionInfo.textContent = `Session: ${shortId}`;
            sessionInfo.style.color = '';
        }
    } else {
        sessionInfo.textContent = 'New chat';
        sessionInfo.style.color = '';
    }
}

function enableChat() {
    messageInput.disabled = false;
    sendBtn.disabled = false;
    messageInput.focus();
}

function disableChat() {
    messageInput.disabled = true;
    sendBtn.disabled = true;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isStreaming) return;

    // Create a new session if we don't have one
    if (!currentSessionId) {
        await createNewSession(true);
        if (!currentSessionId) return;
    }

    isStreaming = true;
    messageInput.value = '';
    addMessageToUI(message, 'user');

    loading.classList.add('active');
    disableChat();

    const assistantMessageDiv = document.createElement('div');
    assistantMessageDiv.className = 'message assistant';
    assistantMessageDiv.innerHTML = `
        <div>
            <div class="message-label">Assistant</div>
            <div class="message-content"></div>
        </div>
    `;
    messagesDiv.appendChild(assistantMessageDiv);
    const contentDiv = assistantMessageDiv.querySelector('.message-content');

    try {
        const response = await fetch(`/api/sessions/${currentSessionId}/messages/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                message: message
            })
        });

        if (!response.ok) {
            throw new Error('Failed to send message');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedContent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        console.log('Received SSE data:', data);
                        if (data.error) {
                            console.error('Server error:', data.error);
                            contentDiv.textContent = 'Error: ' + data.error;
                        }
                        if (data.chunk) {
                            accumulatedContent += data.chunk;
                            contentDiv.textContent = accumulatedContent;
                            messagesDiv.scrollTop = messagesDiv.scrollHeight;
                        }
                        if (data.done) {
                            if (accumulatedContent === '') {
                                contentDiv.textContent = '(No response received)';
                            }
                            await loadSessions();
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e, 'Line:', line);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error sending message:', error);
        assistantMessageDiv.remove();
        showError('Failed to send message. Please try again.');
    } finally {
        loading.classList.remove('active');
        enableChat();
        isStreaming = false;
        messageInput.focus();
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    messagesDiv.appendChild(errorDiv);
    setTimeout(() => errorDiv.remove(), 5000);
}

function showModal() {
    document.getElementById('newSessionModal').classList.add('active');
    document.getElementById('sessionDescription').focus();
}

function closeModal() {
    document.getElementById('newSessionModal').classList.remove('active');
    document.getElementById('sessionDescription').value = '';
    document.getElementById('sessionTags').value = '';
}

async function createNewSession(autoCreate = false) {
    let description = '';
    let tags = [];

    if (!autoCreate) {
        description = document.getElementById('sessionDescription').value.trim();
        const tagsInput = document.getElementById('sessionTags').value.trim();
        tags = tagsInput ? tagsInput.split(',').map(t => t.trim()).filter(t => t) : [];
    }

    try {
        const response = await fetch('/api/sessions/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description, tags })
        });

        if (!response.ok) {
            throw new Error('Failed to create session');
        }

        const data = await response.json();
        if (!autoCreate) {
            closeModal();
        }

        currentSessionId = data.session_id;
        currentSession = {
            session_id: data.session_id,
            description: description,
            tags: tags,
            messages: [],
            created_at: data.created_at,
            updated_at: data.created_at
        };

        displaySession();
        updateSessionInfo();
        enableChat();
        await loadSessions();

    } catch (error) {
        console.error('Error creating session:', error);
        showError('Failed to create new session');
    }
}

async function loadBiographyFiles() {
    try {
        const response = await fetch('/api/biography-files');
        if (!response.ok) {
            throw new Error('Failed to load biography files');
        }

        biographyFiles = await response.json();

        const select = document.getElementById('biographyFile');
        select.innerHTML = '';

        if (biographyFiles.length === 0) {
            select.innerHTML = '<option value="">No biography files found</option>';
            return;
        }

        // Group files by directory
        const benchmarkFiles = biographyFiles.filter(f => f.directory === 'benchmarks');
        const biographyDirFiles = biographyFiles.filter(f => f.directory === 'biographies');

        if (benchmarkFiles.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Benchmarks';
            benchmarkFiles.forEach(file => {
                const option = document.createElement('option');
                option.value = file.path;
                option.textContent = file.filename;
                optgroup.appendChild(option);
            });
            select.appendChild(optgroup);
        }

        if (biographyDirFiles.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Biographies';
            biographyDirFiles.forEach(file => {
                const option = document.createElement('option');
                option.value = file.path;
                option.textContent = file.filename;
                optgroup.appendChild(option);
            });
            select.appendChild(optgroup);
        }

        // Select first file by default
        if (biographyFiles.length > 0) {
            select.value = biographyFiles[0].path;
            updateFileInfo();
        }

    } catch (error) {
        console.error('Error loading biography files:', error);
        const select = document.getElementById('biographyFile');
        select.innerHTML = '<option value="">Error loading files</option>';
    }
}

function updateFileInfo() {
    const select = document.getElementById('biographyFile');
    const fileInfo = document.getElementById('fileInfo');

    if (select.value) {
        const file = biographyFiles.find(f => f.path === select.value);
        if (file) {
            fileInfo.innerHTML = `<strong>Directory:</strong> ${file.directory}<br><strong>Path:</strong> ${file.path}`;
        }
    } else {
        fileInfo.innerHTML = '';
    }
}

function showQuirkbotModal() {
    const modal = document.getElementById('quirkbotModal');
    modal.classList.add('active');
    loadBiographyFiles();
}

function closeQuirkbotModal() {
    const modal = document.getElementById('quirkbotModal');
    modal.classList.remove('active');
}

async function startQuirkbotSession() {
    const select = document.getElementById('biographyFile');
    const biographyFile = select.value;

    if (!biographyFile) {
        showError('Please select a biography file');
        return;
    }

    try {
        const response = await fetch('/api/quirkbot/random', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ biography_file: biographyFile })
        });

        if (!response.ok) {
            throw new Error('Failed to create quirkbot session');
        }

        const data = await response.json();

        closeQuirkbotModal();
        currentSessionId = data.session_id;

        // Load the full session to get all metadata
        await loadSession(currentSessionId);
        await loadSessions(); // Refresh sessions list

    } catch (error) {
        console.error('Error creating quirkbot session:', error);
        showError('Failed to create quirkbot session');
    }
}

newSessionBtn.onclick = showModal;
quirkbotBtn.onclick = showQuirkbotModal;
sendBtn.onclick = sendMessage;

// Add event listener for biography file selection
if (document.getElementById('biographyFile')) {
    document.getElementById('biographyFile').onchange = updateFileInfo;
}

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

window.onclick = (event) => {
    const modal = document.getElementById('newSessionModal');
    if (event.target === modal) {
        closeModal();
    }
};

// Initialize the app
loadSessions();
updateSessionInfo();
messageInput.focus();