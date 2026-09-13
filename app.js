/* =========================================================
   MASTER AI — client-only build
   Real: camera (getUserMedia), speech-to-text & text-to-speech
   (Web Speech API), live-mode interrupt, image attachments,
   cross-chat memory, multi-provider API calls straight from
   the browser using YOUR OWN key (stored only in this browser's
   localStorage — there is no server, so nothing leaves your
   device except the direct call to the AI provider you choose).
   ========================================================= */

const store = {
  get(key, fallback){ try{ const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }catch(e){ return fallback; } },
  set(key, val){ localStorage.setItem(key, JSON.stringify(val)); }
};

const KEYS = { apis:'masterai_apis', convos:'masterai_convos', active:'masterai_active_convo', settings:'masterai_settings' };

let apis = store.get(KEYS.apis, []);
let convos = store.get(KEYS.convos, []);
let activeConvoId = store.get(KEYS.active, null);
let settings = store.get(KEYS.settings, { override:false, focusNudge:false });

function saveApis(){ store.set(KEYS.apis, apis); }
function saveConvos(){ store.set(KEYS.convos, convos); }
function saveSettings(){ store.set(KEYS.settings, settings); }

function getActiveApi(){ return apis.find(a => a.active); }

function ensureConvo(){
  if (!activeConvoId || !convos.find(c => c.id === activeConvoId)) {
    const c = { id: 'c' + Date.now(), title: 'New chat', messages: [], updatedAt: Date.now() };
    convos.unshift(c);
    activeConvoId = c.id;
    saveConvos(); store.set(KEYS.active, activeConvoId);
  }
  return convos.find(c => c.id === activeConvoId);
}

/* =================== Provider calls =================== */
const PROVIDER_DEFAULTS = {
  gemini:    { model:'gemini-2.5-flash' },
  openai:    { model:'gpt-4o-mini' },
  anthropic: { model:'claude-3-5-sonnet-20241022' },
  custom:    { model:'' }
};

// messages: [{role:'user'|'assistant', text, image(dataURL)?}]
async function callProvider(api, messages, systemPrompt, signal){
  const provider = api.provider;
  if (provider === 'gemini') return callGemini(api, messages, systemPrompt, signal);
  if (provider === 'openai') return callOpenAiCompatible(api, messages, systemPrompt, signal, 'https://api.openai.com/v1');
  if (provider === 'anthropic') return callAnthropic(api, messages, systemPrompt, signal);
  if (provider === 'custom') return callOpenAiCompatible(api, messages, systemPrompt, signal, api.endpoint);
  throw new Error('Unknown provider');
}

async function callGemini(api, messages, systemPrompt, signal){
  const model = api.model || PROVIDER_DEFAULTS.gemini.model;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(api.apiKey)}`;
  const contents = messages.map(m => {
    const parts = [];
    if (m.text) parts.push({ text: m.text });
    if (m.image) parts.push({ inlineData: { mimeType: 'image/jpeg', data: m.image.split(',')[1] } });
    return { role: m.role === 'assistant' ? 'model' : 'user', parts };
  });
  const body = { contents };
  if (systemPrompt) body.systemInstruction = { parts: [{ text: systemPrompt }] };
  const res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body), signal });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `Gemini error ${res.status}`);
  const text = data?.candidates?.[0]?.content?.parts?.map(p => p.text).filter(Boolean).join('\n');
  if (!text) throw new Error('Gemini returned no text (it may have blocked the response).');
  return text;
}

async function callOpenAiCompatible(api, messages, systemPrompt, signal, baseUrl){
  if (!baseUrl) throw new Error('No endpoint configured for this Custom API.');
  const model = api.model || PROVIDER_DEFAULTS.openai.model;
  const url = baseUrl.replace(/\/$/, '') + '/chat/completions';
  const msgs = [];
  if (systemPrompt) msgs.push({ role:'system', content: systemPrompt });
  messages.forEach(m => {
    if (m.image) {
      msgs.push({ role: m.role, content: [
        ...(m.text ? [{ type:'text', text: m.text }] : []),
        { type:'image_url', image_url: { url: m.image } }
      ]});
    } else {
      msgs.push({ role: m.role, content: m.text });
    }
  });
  const res = await fetch(url, {
    method:'POST',
    headers:{ 'Content-Type':'application/json', 'Authorization': `Bearer ${api.apiKey}` },
    body: JSON.stringify({ model, messages: msgs }),
    signal
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `API error ${res.status} (if this is a CORS error, the provider may be blocking direct browser calls — try a Custom OpenAI-compatible proxy instead)`);
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error('No text returned.');
  return text;
}

async function callAnthropic(api, messages, systemPrompt, signal){
  const model = api.model || PROVIDER_DEFAULTS.anthropic.model;
  const msgs = messages.map(m => {
    if (m.image) {
      return { role: m.role, content: [
        ...(m.text ? [{ type:'text', text: m.text }] : []),
        { type:'image', source: { type:'base64', media_type:'image/jpeg', data: m.image.split(',')[1] } }
      ]};
    }
    return { role: m.role, content: m.text };
  });
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method:'POST',
    headers:{
      'Content-Type':'application/json',
      'x-api-key': api.apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true'
    },
    body: JSON.stringify({ model, max_tokens: 1024, system: systemPrompt || undefined, messages: msgs }),
    signal
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `Anthropic error ${res.status}`);
  const text = data?.content?.map(c => c.text).filter(Boolean).join('\n');
  if (!text) throw new Error('No text returned.');
  return text;
}

/* small cross-conversation memory: last few messages from the other
   most-recently-updated conversation, folded into the system prompt */
function buildMemoryPreamble(){
  const others = convos.filter(c => c.id !== activeConvoId && c.messages.length)
                        .sort((a,b) => b.updatedAt - a.updatedAt)
                        .slice(0, 1);
  if (!others.length) return '';
  const recent = others[0].messages.slice(-6)
    .map(m => `${m.role === 'user' ? 'User' : 'You'}: ${m.text || '[image]'}`).join('\n');
  return `For light context, here is a snippet from an earlier conversation with this user:\n${recent}\n\nOnly use this if relevant; otherwise ignore it.`;
}

/* =================== UI: tabs / drawer =================== */
const screens = document.querySelectorAll('.screen');
const navItems = document.querySelectorAll('.nav-item');
function showTab(name){
  screens.forEach(s => s.classList.toggle('active', s.id === 'screen-' + name));
  navItems.forEach(n => n.classList.toggle('active', n.dataset.tab === name));
  closeDrawer();
}
navItems.forEach(btn => btn.addEventListener('click', () => showTab(btn.dataset.tab)));

const drawer = document.getElementById('drawer');
const scrim = document.getElementById('scrim');
function openDrawer(){ renderHistory(); drawer.classList.add('open'); scrim.classList.add('show'); }
function closeDrawer(){ drawer.classList.remove('open'); scrim.classList.remove('show'); }
document.getElementById('menuBtn').addEventListener('click', openDrawer);
document.getElementById('drawerClose').addEventListener('click', closeDrawer);
scrim.addEventListener('click', closeDrawer);

const appRoot = document.getElementById('appRoot');
const drawerOverride = document.getElementById('drawerOverride');
const overrideToggle = document.getElementById('overrideToggle');
function applyOverride(v){ appRoot.classList.toggle('amaterasu', v); drawerOverride.checked = v; overrideToggle.checked = v; settings.override = v; saveSettings(); }
drawerOverride.addEventListener('change', () => applyOverride(drawerOverride.checked));
overrideToggle.addEventListener('change', () => applyOverride(overrideToggle.checked));
applyOverride(!!settings.override);

const focusNudgeToggle = document.getElementById('focusNudgeToggle');
focusNudgeToggle.checked = !!settings.focusNudge;
focusNudgeToggle.addEventListener('change', () => { settings.focusNudge = focusNudgeToggle.checked; saveSettings(); });

let hiddenAt = null;
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { hiddenAt = Date.now(); }
  else if (hiddenAt && settings.focusNudge) {
    const away = Date.now() - hiddenAt;
    if (away > 15000) addMessage('ai', "Welcome back. Was that a break, or are you supposed to be somewhere else right now?");
    hiddenAt = null;
  }
});

document.getElementById('newSessionBtn').addEventListener('click', () => {
  const c = { id:'c'+Date.now(), title:'New chat', messages:[], updatedAt: Date.now() };
  convos.unshift(c); activeConvoId = c.id;
  saveConvos(); store.set(KEYS.active, activeConvoId);
  renderChatLog(); renderHistory(); showTab('home');
});

function renderHistory(){
  const list = document.getElementById('historyList');
  list.innerHTML = '';
  convos.sort((a,b) => b.updatedAt - a.updatedAt).forEach(c => {
    const btn = document.createElement('button');
    btn.className = 'history-item' + (c.id === activeConvoId ? ' active' : '');
    const sub = c.messages.length ? (c.messages[c.messages.length-1].text || '[image]') : 'Empty session';
    btn.innerHTML = `<span class="h-title">${escapeHtml(c.title)}</span><span class="h-sub">${escapeHtml(sub)}</span>`;
    btn.addEventListener('click', () => { activeConvoId = c.id; store.set(KEYS.active, activeConvoId); renderChatLog(); showTab('home'); });
    list.appendChild(btn);
  });
}

function escapeHtml(s){ const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

/* =================== Chat =================== */
const chatLog = document.getElementById('chatLog');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const attachBtn = document.getElementById('attachBtn');
const fileInput = document.getElementById('fileInput');
const attachPreview = document.getElementById('attachPreview');
const attachThumb = document.getElementById('attachThumb');
const removeAttach = document.getElementById('removeAttach');
const statusPill = document.getElementById('statusPill');
const statusText = document.getElementById('statusText');

let pendingImage = null;

function updateStatusPill(){
  const api = getActiveApi();
  statusPill.querySelector('.status-dot').classList.toggle('on', !!api);
  statusText.textContent = api ? `${api.name.toUpperCase()} · ${(api.model || '').toUpperCase()}` : 'NO API CONFIGURED';
}

function renderChatLog(){
  const convo = ensureConvo();
  chatLog.innerHTML = '';
  convo.messages.forEach(m => addMessageDom(m.role, m.text, m.image));
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addMessageDom(role, text, image){
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'ai');
  if (image) { const img = document.createElement('img'); img.src = image; div.appendChild(img); }
  if (text) { const p = document.createElement('div'); p.textContent = text; div.appendChild(p); }
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function addMessage(role, text, image){
  const convo = ensureConvo();
  convo.messages.push({ role: role === 'user' ? 'user' : 'assistant', text, image: image || null });
  convo.updatedAt = Date.now();
  if (convo.title === 'New chat' && role === 'user' && text) convo.title = text.slice(0, 40);
  saveConvos();
  return addMessageDom(role, text, image);
}

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => { pendingImage = reader.result; attachThumb.src = pendingImage; attachPreview.classList.remove('hidden'); };
  reader.readAsDataURL(file);
});
removeAttach.addEventListener('click', () => { pendingImage = null; attachPreview.classList.add('hidden'); fileInput.value = ''; });

async function sendMessage(){
  const text = chatInput.value.trim();
  if (!text && !pendingImage) return;
  const api = getActiveApi();
  chatInput.value = '';
  const image = pendingImage;
  pendingImage = null; attachPreview.classList.add('hidden'); fileInput.value = '';

  addMessage('user', text, image);

  if (!api) {
    addMessage('ai', 'No active AI API configured yet. Go to Settings → AI APIs, add one, test it, and switch it on.');
    return;
  }

  const thinkingDom = addMessageDom('ai', 'Thinking…');
  thinkingDom.classList.add('pending');

  try {
    const convo = ensureConvo();
    const history = convo.messages.slice(0, -1).slice(-10); // exclude the just-added user msg, cap length
    const messagesForApi = [...history, { role:'user', text, image }];
    const memory = buildMemoryPreamble();
    const systemPrompt = [api.systemPrompt, memory].filter(Boolean).join('\n\n');
    const reply = await callProvider(api, messagesForApi, systemPrompt);
    thinkingDom.classList.remove('pending');
    thinkingDom.querySelector('div') ? (thinkingDom.querySelector('div').textContent = reply) : (thinkingDom.textContent = reply);
    addMessage('ai', reply); // persist (dom already shows it via thinkingDom, so just persist without re-render)
    // remove the duplicate freshly-appended dom node since thinkingDom already displays it
    chatLog.removeChild(chatLog.lastChild);
  } catch (e) {
    thinkingDom.classList.remove('pending');
    thinkingDom.classList.add('error');
    const msg = 'Error: ' + e.message;
    thinkingDom.querySelector('div') ? (thinkingDom.querySelector('div').textContent = msg) : (thinkingDom.textContent = msg);
  }
}
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

/* dictation (one-shot speech to text into the input box) */
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const dictateBtn = document.getElementById('dictateBtn');
if (SpeechRecognition) {
  dictateBtn.addEventListener('click', () => {
    const rec = new SpeechRecognition();
    rec.lang = 'en-IN'; rec.interimResults = false; rec.maxAlternatives = 1;
    dictateBtn.classList.add('recording');
    rec.onresult = (e) => { chatInput.value = e.results[0][0].transcript; };
    rec.onend = () => dictateBtn.classList.remove('recording');
    rec.onerror = () => dictateBtn.classList.remove('recording');
    rec.start();
  });
} else {
  dictateBtn.addEventListener('click', () => alert('Speech recognition is not supported in this browser.'));
}

/* =================== Vision / camera =================== */
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraOff = document.getElementById('cameraOff');
const startCamBtn = document.getElementById('startCamBtn');
const flipCamBtn = document.getElementById('flipCamBtn');
const stopCamBtn = document.getElementById('stopCamBtn');
const askVisionBtn = document.getElementById('askVisionBtn');
const visionQuestion = document.getElementById('visionQuestion');
const visionAnswer = document.getElementById('visionAnswer');

let camStream = null;
let facingMode = 'user';

async function startCamera(){
  try {
    if (camStream) camStream.getTracks().forEach(t => t.stop());
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
    cameraVideo.srcObject = camStream;
    cameraOff.style.display = 'none';
    flipCamBtn.disabled = false; stopCamBtn.disabled = false; askVisionBtn.disabled = false;
  } catch (e) {
    visionAnswer.textContent = 'Could not access the camera: ' + e.message;
  }
}
function stopCamera(){
  if (camStream) camStream.getTracks().forEach(t => t.stop());
  camStream = null; cameraVideo.srcObject = null;
  cameraOff.style.display = 'flex';
  flipCamBtn.disabled = true; stopCamBtn.disabled = true; askVisionBtn.disabled = true;
}
startCamBtn.addEventListener('click', startCamera);
stopCamBtn.addEventListener('click', stopCamera);
flipCamBtn.addEventListener('click', () => { facingMode = facingMode === 'user' ? 'environment' : 'user'; startCamera(); });

askVisionBtn.addEventListener('click', async () => {
  const api = getActiveApi();
  if (!api) { visionAnswer.textContent = 'Add and switch on an AI API in Settings first.'; return; }
  if (!camStream) { visionAnswer.textContent = 'Start the camera first.'; return; }
  cameraCanvas.width = cameraVideo.videoWidth; cameraCanvas.height = cameraVideo.videoHeight;
  cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0);
  const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.8);
  const question = visionQuestion.value.trim() || 'What do you see in this image?';
  visionAnswer.textContent = 'Looking…';
  try {
    const reply = await callProvider(api, [{ role:'user', text: question, image: dataUrl }], api.systemPrompt);
    visionAnswer.textContent = reply;
  } catch (e) {
    visionAnswer.textContent = 'Error: ' + e.message;
  }
});

/* =================== Master Live (continuous voice) =================== */
const liveOverlay = document.getElementById('liveOverlay');
const liveOrb = document.getElementById('liveOrb');
const liveHint = document.getElementById('liveHint');
const liveTranscript = document.getElementById('liveTranscript');
const micBig = document.getElementById('micBig');
const stopInterruptBtn = document.getElementById('stopInterruptBtn');

let liveRecognition = null;
let liveState = 'idle'; // idle | listening | thinking | speaking
let liveAbortController = null;
let liveOpen = false;

function setLiveState(state){
  liveState = state;
  liveOrb.classList.remove('listening', 'thinking', 'speaking');
  if (state !== 'idle') liveOrb.classList.add(state);
  stopInterruptBtn.classList.toggle('hidden', state === 'idle' || state === 'listening');
  liveHint.textContent = { idle:'TAP MIC TO START', listening:'LISTENING…', thinking:'THINKING…', speaking:'SPEAKING…' }[state];
  micBig.classList.toggle('on', state !== 'idle');
}

document.getElementById('liveBtn').addEventListener('click', openLive);
document.getElementById('liveClose').addEventListener('click', closeLive);

function openLive(){
  liveOverlay.classList.add('show');
  liveOpen = true;
  liveTranscript.textContent = '';
  setLiveState('idle');
}
function closeLive(){
  liveOpen = false;
  stopLiveListening();
  window.speechSynthesis.cancel();
  if (liveAbortController) liveAbortController.abort();
  liveOverlay.classList.remove('show');
  setLiveState('idle');
}

function startLiveListening(){
  if (!SpeechRecognition) { liveHint.textContent = 'Speech recognition not supported in this browser.'; return; }
  liveRecognition = new SpeechRecognition();
  liveRecognition.lang = 'en-IN';
  liveRecognition.continuous = true;
  liveRecognition.interimResults = true;

  liveRecognition.onresult = (e) => {
    let finalText = '', interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) finalText += r[0].transcript; else interim += r[0].transcript;
    }
    liveTranscript.textContent = finalText || interim;
    if (finalText.trim()) handleLiveUtterance(finalText.trim());
  };
  liveRecognition.onerror = () => { if (liveOpen && liveState === 'listening') { try{ liveRecognition.start(); }catch(e){} } };
  liveRecognition.onend = () => { if (liveOpen && liveState === 'listening') { try{ liveRecognition.start(); }catch(e){} } };

  try { liveRecognition.start(); } catch(e){}
  setLiveState('listening');
}
function stopLiveListening(){
  if (liveRecognition) { liveRecognition.onend = null; liveRecognition.onerror = null; liveRecognition.stop(); liveRecognition = null; }
}

micBig.addEventListener('click', () => {
  if (liveState === 'idle') startLiveListening();
  else { stopLiveListening(); window.speechSynthesis.cancel(); if (liveAbortController) liveAbortController.abort(); setLiveState('idle'); }
});

stopInterruptBtn.addEventListener('click', () => {
  // interrupt whatever Master AI is doing (thinking or speaking) and go straight back to listening
  window.speechSynthesis.cancel();
  if (liveAbortController) liveAbortController.abort();
  liveTranscript.textContent = '';
  startLiveListening();
});

async function handleLiveUtterance(text){
  stopLiveListening();
  setLiveState('thinking');
  const api = getActiveApi();
  if (!api) {
    speakAndResume("I don't have an active AI API yet. Add one in Settings.");
    return;
  }
  liveAbortController = new AbortController();
  try {
    const convo = ensureConvo();
    addMessage('user', text);
    const history = convo.messages.slice(-10);
    const memory = buildMemoryPreamble();
    const systemPrompt = [api.systemPrompt, memory, 'Keep spoken replies short and conversational.'].filter(Boolean).join('\n\n');
    const reply = await callProvider(api, history, systemPrompt, liveAbortController.signal);
    addMessage('ai', reply);
    speakAndResume(reply);
  } catch (e) {
    if (e.name === 'AbortError') return; // user interrupted — startLiveListening already called by stop button
    speakAndResume('Sorry, something went wrong: ' + e.message);
  }
}

function speakAndResume(text){
  if (!liveOpen) return;
  setLiveState('speaking');
  liveTranscript.textContent = text;
  const utter = new SpeechSynthesisUtterance(text);
  utter.onend = () => { if (liveOpen) startLiveListening(); };
  utter.onerror = () => { if (liveOpen) startLiveListening(); };
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

/* =================== AI API settings =================== */
const addApiBtn = document.getElementById('addApiBtn');
const newApiForm = document.getElementById('newApiForm');
const closeApiForm = document.getElementById('closeApiForm');
const saveApiBtn = document.getElementById('saveApiBtn');
const testApiBtn = document.getElementById('testApiBtn');
const testResult = document.getElementById('testResult');
const apiList = document.getElementById('apiList');
const apiEmpty = document.getElementById('apiEmpty');
const apiProviderSel = document.getElementById('apiProvider');
const apiEndpointInput = document.getElementById('apiEndpoint');
const apiModelInput = document.getElementById('apiModel');

function updateEndpointVisibility(){
  apiEndpointInput.style.display = apiProviderSel.value === 'custom' ? 'block' : 'none';
  if (!apiModelInput.value) apiModelInput.placeholder = 'Model (e.g. ' + (PROVIDER_DEFAULTS[apiProviderSel.value].model || 'model-name') + ')';
}
apiProviderSel.addEventListener('change', updateEndpointVisibility);
updateEndpointVisibility();

addApiBtn.addEventListener('click', () => { newApiForm.classList.remove('hidden'); testResult.textContent=''; testResult.className='test-result'; });
closeApiForm.addEventListener('click', () => newApiForm.classList.add('hidden'));

function readForm(){
  return {
    name: document.getElementById('apiName').value.trim() || 'Untitled API',
    provider: apiProviderSel.value,
    apiKey: document.getElementById('apiKey').value.trim(),
    endpoint: apiEndpointInput.value.trim(),
    model: apiModelInput.value.trim() || PROVIDER_DEFAULTS[apiProviderSel.value].model,
    purpose: document.getElementById('apiCategory').value.trim() || 'general',
    systemPrompt: document.getElementById('apiPrompt').value.trim()
  };
}

testApiBtn.addEventListener('click', async () => {
  const draft = readForm();
  if (!draft.apiKey) { testResult.textContent = 'Enter an API key first.'; testResult.className = 'test-result bad'; return; }
  testApiBtn.textContent = '⚡ Testing…';
  testResult.textContent = ''; testResult.className = 'test-result';
  try {
    const reply = await callProvider(draft, [{ role:'user', text:"Reply with just the word 'ok'." }], '');
    testResult.textContent = 'Working — response: ' + reply.slice(0, 80);
    testResult.className = 'test-result ok';
  } catch (e) {
    testResult.textContent = e.message;
    testResult.className = 'test-result bad';
  }
  testApiBtn.textContent = '⚡ Test';
});

saveApiBtn.addEventListener('click', () => {
  const draft = readForm();
  if (!draft.apiKey && draft.provider !== 'custom') { testResult.textContent = 'Enter an API key first.'; testResult.className = 'test-result bad'; return; }
  draft.id = 'api' + Date.now();
  draft.active = apis.length === 0; // first one added becomes active automatically
  apis.push(draft);
  saveApis();
  renderApiList();
  newApiForm.classList.add('hidden');
  ['apiName','apiKey','apiEndpoint','apiModel','apiPrompt'].forEach(id => document.getElementById(id).value = '');
  document.getElementById('apiCategory').value = 'general';
  updateStatusPill();
});

function renderApiList(){
  apiList.innerHTML = '';
  apiEmpty.style.display = apis.length ? 'none' : 'block';
  apis.forEach(a => {
    const row = document.createElement('div');
    row.className = 'api-row';
    row.innerHTML = `
      <div class="api-meta"><strong>${escapeHtml(a.name)}</strong> · ${escapeHtml(a.provider)} · ${escapeHtml(a.model||'')}
        ${a.active ? '<span class="api-badge">ACTIVE</span>' : ''}</div>
      <div class="api-row-actions">
        <button class="zap" title="Test">&#9889;</button>
        <label class="switch" style="width:38px;height:22px;">
          <input type="checkbox" ${a.active ? 'checked' : ''} />
          <span class="slider"></span>
        </label>
        <button class="del" title="Delete">&times;</button>
      </div>`;
    row.querySelector('.zap').addEventListener('click', async () => {
      row.querySelector('.zap').textContent = '…';
      try { await callProvider(a, [{role:'user', text:"Reply with just the word 'ok'."}], ''); alert(a.name + ': working ✓'); }
      catch(e){ alert(a.name + ': ' + e.message); }
      row.querySelector('.zap').innerHTML = '&#9889;';
    });
    row.querySelector('input[type=checkbox]').addEventListener('change', (e) => {
      if (e.target.checked) apis.forEach(x => x.active = (x.id === a.id));
      else a.active = false;
      saveApis(); renderApiList(); updateStatusPill();
    });
    row.querySelector('.del').addEventListener('click', () => {
      apis = apis.filter(x => x.id !== a.id);
      saveApis(); renderApiList(); updateStatusPill();
    });
    apiList.appendChild(row);
  });
}

/* =================== init =================== */
renderApiList();
renderChatLog();
updateStatusPill();
