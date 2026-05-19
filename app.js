import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache = true;

const MOBILENET_MODEL = 'Xenova/mobilenet-v2';
const SMOLVLM_MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct';

const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;

const STAGES = [
    { id: 'unripe',   emoji: '🟢', title: 'Unripe — Not Yet',     desc: 'This banana needs more time. Leave it at room temperature for 3–5 days until it turns yellow.', pos: 7 },
    { id: 'nearly',   emoji: '🟡', title: 'Almost Ready',          desc: "Getting close! Give it 1–2 more days and it'll be perfect.", pos: 28 },
    { id: 'perfect',  emoji: '⭐',       title: 'Perfect to Eat!',        desc: 'This is the sweet spot — bright yellow, firm, and at peak sweetness. Eat it now!', pos: 50 },
    { id: 'ripe',     emoji: '✅',       title: 'Ripe & Very Sweet',       desc: 'The brown spots mean extra sugar. Great for eating fresh, smoothies, or freezing.', pos: 72 },
    { id: 'overripe', emoji: '🍞', title: 'Overripe — Bake It!', desc: "Too soft to eat fresh, but perfect for banana bread, muffins, or pancakes. Don't throw it out!", pos: 92 }
];

let vlmPipe = null;
let classifier = null;
let modelLoading = false;
let useVLM = false;
let cameraStream = null;

const el = id => document.getElementById(id);
const show = (...ids) => ids.forEach(id => el(id).classList.remove('hidden'));
const hide = (...ids) => ids.forEach(id => el(id).classList.add('hidden'));
const delay = ms => new Promise(r => setTimeout(r, ms));

function setModeIndicator(text) {
    const ind = el('mode-indicator');
    if (ind) { ind.textContent = `🤖 ${text}`; ind.classList.remove('hidden'); }
}

async function loadMobileNet() {
    el('loading-text').textContent = 'Downloading model (first time only)...';
    el('progress-fill').style.width = '5%';
    try {
        classifier = await pipeline('image-classification', MOBILENET_MODEL, {
            progress_callback: ({ status, progress }) => {
                if (status === 'progress' && progress != null) {
                    el('progress-fill').style.width = `${5 + progress * 0.88}%`;
                    el('loading-text').textContent = `Loading model: ${Math.round(progress)}%`;
                }
            }
        });
        el('progress-fill').style.width = '100%';
        el('loading-text').textContent = 'Model ready!';
        setModeIndicator('MobileNet-v2 + color analysis');
        await delay(400);
    } catch (err) {
        console.warn('MobileNet failed:', err);
        el('loading-text').textContent = 'Using color analysis (model unavailable).';
        setModeIndicator('Color analysis only');
        await delay(1200);
    }
}

async function loadModel() {
    if (vlmPipe || classifier || modelLoading) return;
    modelLoading = true;
    show('loading');
    hide('start-screen');

    if (hasWebGPU) {
        el('loading-text').textContent = 'Loading SmolVLM-256M (~200 MB, cached after this)...';
        el('progress-fill').style.width = '3%';
        try {
            vlmPipe = await pipeline('image-text-to-text', SMOLVLM_MODEL, {
                device: 'webgpu',
                dtype: {
                    embed_tokens: 'q8',
                    vision_encoder: 'q8',
                    decoder_model_merged: 'q4',
                },
                progress_callback: ({ status, progress }) => {
                    if (status === 'progress' && progress != null) {
                        el('progress-fill').style.width = `${3 + progress * 0.9}%`;
                        el('loading-text').textContent = `Loading SmolVLM: ${Math.round(progress)}%`;
                    }
                }
            });
            useVLM = true;
            el('progress-fill').style.width = '100%';
            el('loading-text').textContent = 'SmolVLM ready!';
            setModeIndicator('SmolVLM-256M · WebGPU');
            await delay(400);
        } catch (err) {
            console.warn('SmolVLM failed, falling back to MobileNet:', err);
            vlmPipe = null;
            useVLM = false;
            await loadMobileNet();
        }
    } else {
        await loadMobileNet();
    }

    hide('loading');
    modelLoading = false;
}

async function startCamera() {
    hide('btn-start', 'start-screen', 'result');
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } }
        });
        el('video').srcObject = cameraStream;
        show('camera-view', 'btn-capture');
        loadModel();
    } catch {
        alert('Camera access was denied. Please allow camera access and reload the page.');
        show('btn-start', 'start-screen');
    }
}

function captureFrame() {
    const video = el('video');
    const canvas = el('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.92);
}

function stopCamera() {
    if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
}

// ---- VLM path ----

async function analyzeWithVLM(dataUrl) {
    const messages = [{
        role: 'user',
        content: [
            { type: 'image', url: dataUrl },
            { type: 'text', text: 'Is there a banana in this image? If yes, assess its ripeness as exactly one of: unripe, nearly-ripe, perfect, ripe, overripe. Reply with the ripeness word first, then one sentence describing what you see. If there is no banana, reply: not-a-banana.' }
        ]
    }];
    const output = await vlmPipe(messages, { max_new_tokens: 80 });
    const last = output[0].generated_text.at(-1);
    const content = last.content;
    return (typeof content === 'string' ? content
        : Array.isArray(content) ? (content.find(c => c.type === 'text')?.text ?? '')
        : String(content)).trim();
}

function parseVLMResponse(text) {
    const lower = text.toLowerCase();
    if (/not.a.banana|no banana|cannot see|don.t see|i see no/i.test(lower)) {
        return { stage: null, explanation: text };
    }
    let stage;
    if (/over.?ripe/i.test(lower))                   stage = STAGES[4];
    else if (/nearly.?ripe|almost ripe/i.test(lower)) stage = STAGES[1];
    else if (/\bperfect/i.test(lower))                stage = STAGES[2];
    else if (/unripe|un.ripe|not.?ripe/i.test(lower)) stage = STAGES[0];
    else if (/\bripe\b/i.test(lower))                 stage = STAGES[3];
    else                                               stage = STAGES[2];

    const explanation = text
        .replace(/^(overripe|over-ripe|nearly-ripe|nearly ripe|perfect|unripe|ripe)[,\.\s]*/i, '')
        .trim();
    return { stage, explanation: explanation.length > 10 ? explanation : stage.desc };
}

// ---- MobileNet + color analysis fallback ----

function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const l = (max + min) / 2;
    if (max === min) return [0, 0, l];
    const d = max - min;
    const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    let h;
    switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
        case g: h = ((b - r) / d + 2) / 6; break;
        default: h = ((r - g) / d + 4) / 6;
    }
    return [h * 360, s, l];
}

function analyzeRipeness(ctx, w, h) {
    const { data } = ctx.getImageData(0, 0, w, h);
    const counts = [0, 0, 0, 0, 0];
    let relevant = 0;
    for (let i = 0; i < data.length; i += 4) {
        const [hue, s, l] = rgbToHsl(data[i], data[i + 1], data[i + 2]);
        if (s < 0.18 || l < 0.22 || l > 0.88) continue;
        relevant++;
        if (hue >= 85 && hue <= 155)      counts[0]++;
        else if (hue >= 60 && hue < 85)   counts[1]++;
        else if (hue >= 44 && hue < 60)   counts[2]++;
        else if (hue >= 25 && hue < 44)   counts[3]++;
        else if (hue < 25 || hue > 330)   counts[4]++;
    }
    if (relevant < 150) return null;
    const total = counts.reduce((a, b) => a + b, 0) || 1;
    return counts[0]/total*0.05 + counts[1]/total*0.28 + counts[2]/total*0.52 + counts[3]/total*0.76 + counts[4]/total*0.95;
}

function stageFromScore(score) {
    if (score < 0.18) return STAGES[0];
    if (score < 0.38) return STAGES[1];
    if (score < 0.58) return STAGES[2];
    if (score < 0.78) return STAGES[3];
    return STAGES[4];
}

async function analyzeFallback(dataUrl) {
    show('loading');
    el('loading-text').textContent = 'Analyzing...';
    el('progress-fill').style.width = '20%';

    const canvas = el('canvas');
    const ctx = canvas.getContext('2d');
    const colorScore = analyzeRipeness(ctx, canvas.width, canvas.height);
    el('progress-fill').style.width = '55%';

    let isBanana = true;
    let modelInfo = 'Color analysis only';

    if (classifier) {
        try {
            el('loading-text').textContent = 'Running vision model...';
            const results = await classifier(dataUrl, { topk: 5 });
            el('progress-fill').style.width = '90%';
            const hit = results.find(r => r.label.toLowerCase().includes('banana'));
            if (hit) {
                modelInfo = `MobileNet-v2: banana ${(hit.score * 100).toFixed(0)}% confident`;
            } else {
                isBanana = false;
                modelInfo = `MobileNet-v2 top: ${results[0].label} (${(results[0].score * 100).toFixed(0)}%)`;
            }
        } catch {
            modelInfo = 'Vision model error — color analysis only';
        }
    }

    el('progress-fill').style.width = '100%';
    await delay(250);
    hide('loading');

    if (!isBanana) {
        return showResult('🤔', "That Doesn't Look Like a Banana",
            "Try pointing the camera directly at a banana. Make sure it's well-lit and fills the frame.",
            50, modelInfo);
    }
    if (colorScore === null) {
        return showResult('❓', 'Hmm, Try Again',
            'The image was too dark or unclear. Move to a brighter spot and try again.',
            50, modelInfo);
    }
    const stage = stageFromScore(colorScore);
    showResult(stage.emoji, stage.title, stage.desc, stage.pos, modelInfo);
}

// ---- Main dispatcher ----

async function analyze(dataUrl) {
    if (useVLM && vlmPipe) {
        show('loading');
        hide('result');
        el('loading-text').textContent = 'Thinking with SmolVLM…';
        el('progress-fill').style.width = '35%';
        try {
            const text = await analyzeWithVLM(dataUrl);
            el('progress-fill').style.width = '100%';
            await delay(200);
            hide('loading');
            const { stage, explanation } = parseVLMResponse(text);
            if (!stage) {
                return showResult('🤔', "That Doesn't Look Like a Banana",
                    'Try pointing the camera directly at a banana and try again.',
                    50, 'SmolVLM-256M (WebGPU)');
            }
            showResult(stage.emoji, stage.title, explanation, stage.pos, 'SmolVLM-256M (WebGPU)');
        } catch (err) {
            console.warn('VLM inference failed, using fallback:', err);
            hide('loading');
            await analyzeFallback(dataUrl);
        }
    } else {
        await analyzeFallback(dataUrl);
    }
}

function showResult(emoji, title, desc, pos, info) {
    el('result-emoji').textContent = emoji;
    el('result-title').textContent = title;
    el('result-desc').textContent = desc;
    el('ripeness-indicator').style.left = `${pos}%`;
    el('model-info').textContent = info;
    show('result');
    el('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function handleCapture() {
    const dataUrl = captureFrame();
    el('preview-img').src = dataUrl;
    hide('camera-view', 'btn-capture');
    show('preview-view', 'btn-retake');
    stopCamera();
    await analyze(dataUrl);
}

function handleRetake() {
    hide('preview-view', 'result', 'btn-retake', 'loading');
    startCamera();
}

el('btn-start').addEventListener('click', startCamera);
el('btn-capture').addEventListener('click', handleCapture);
el('btn-retake').addEventListener('click', handleRetake);

if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(err => console.warn('SW:', err));
}
