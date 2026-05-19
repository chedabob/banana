import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL = 'Xenova/mobilenet-v2';

const STAGES = [
    {
        id: 'unripe',
        emoji: '🟢',
        title: 'Unripe — Not Yet',
        desc: 'This banana needs more time. Leave it out at room temperature for 3–5 days until it turns yellow.',
        pos: 7
    },
    {
        id: 'nearly',
        emoji: '🟡',
        title: 'Almost Ready',
        desc: 'Getting close! Give it 1–2 more days and it\'ll be perfect.',
        pos: 28
    },
    {
        id: 'perfect',
        emoji: '⭐',
        title: 'Perfect to Eat!',
        desc: 'This is the sweet spot — bright yellow, firm, and at peak sweetness. Eat it now!',
        pos: 50
    },
    {
        id: 'ripe',
        emoji: '✅',
        title: 'Ripe & Very Sweet',
        desc: 'The brown spots mean extra sugar. Great for eating fresh, smoothies, or freezing.',
        pos: 72
    },
    {
        id: 'overripe',
        emoji: '🍞',
        title: 'Overripe — Bake It!',
        desc: 'Too soft to eat fresh, but perfect for banana bread, muffins, or pancakes. Don\'t throw it out!',
        pos: 92
    }
];

let classifier = null;
let modelLoading = false;
let cameraStream = null;

const el = id => document.getElementById(id);
const show = (...ids) => ids.forEach(id => el(id).classList.remove('hidden'));
const hide = (...ids) => ids.forEach(id => el(id).classList.add('hidden'));

async function loadModel() {
    if (classifier || modelLoading) return;
    modelLoading = true;
    show('loading');
    hide('start-screen');
    el('loading-text').textContent = 'Downloading vision model (cached after first use)...';
    el('progress-fill').style.width = '5%';

    try {
        classifier = await pipeline('image-classification', MODEL, {
            progress_callback: ({ status, progress }) => {
                if (status === 'progress' && progress != null) {
                    el('progress-fill').style.width = `${5 + progress * 0.88}%`;
                    el('loading-text').textContent = `Loading model: ${Math.round(progress)}%`;
                }
            }
        });
        el('progress-fill').style.width = '100%';
        el('loading-text').textContent = 'Model ready!';
        await delay(400);
    } catch (err) {
        console.warn('Model failed to load — falling back to color analysis only.', err);
        el('loading-text').textContent = 'Using color analysis (model unavailable).';
        await delay(1200);
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
        loadModel(); // background load
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
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
}

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
    const counts = [0, 0, 0, 0, 0]; // unripe, nearly, perfect, ripe, overripe
    let relevant = 0;

    for (let i = 0; i < data.length; i += 4) {
        const [h, s, l] = rgbToHsl(data[i], data[i+1], data[i+2]);
        if (s < 0.18 || l < 0.22 || l > 0.88) continue; // skip grey/white/black
        relevant++;
        if (h >= 85 && h <= 155)       counts[0]++; // green = unripe
        else if (h >= 60 && h < 85)    counts[1]++; // yellow-green = nearly
        else if (h >= 44 && h < 60)    counts[2]++; // pure yellow = perfect
        else if (h >= 25 && h < 44)    counts[3]++; // yellow-brown = ripe
        else if (h < 25 || h > 330)    counts[4]++; // brown/orange = overripe
    }

    if (relevant < 150) return null;

    const total = counts.reduce((a, b) => a + b, 0) || 1;
    // Weighted ripeness score 0–1
    const score =
        counts[0] / total * 0.05 +
        counts[1] / total * 0.28 +
        counts[2] / total * 0.52 +
        counts[3] / total * 0.76 +
        counts[4] / total * 0.95;

    return score;
}

function stageFromScore(score) {
    if (score < 0.18) return STAGES[0];
    if (score < 0.38) return STAGES[1];
    if (score < 0.58) return STAGES[2];
    if (score < 0.78) return STAGES[3];
    return STAGES[4];
}

async function analyze(dataUrl) {
    show('loading');
    hide('result');
    el('loading-text').textContent = 'Analyzing...';
    el('progress-fill').style.width = '20%';

    const canvas = el('canvas');
    const ctx = canvas.getContext('2d');
    const colorScore = analyzeRipeness(ctx, canvas.width, canvas.height);
    el('progress-fill').style.width = '55%';

    let isBanana = true;
    let modelInfo = '';

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
        } catch (e) {
            modelInfo = 'Vision model error — color analysis only';
        }
    } else {
        modelInfo = 'Color analysis only (model loading in background)';
    }

    el('progress-fill').style.width = '100%';
    await delay(250);
    hide('loading');

    if (!isBanana) {
        return showResult('🤔', "That Doesn't Look Like a Banana",
            'Try pointing the camera directly at a banana. Make sure it\'s well-lit and fills the frame.',
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

const delay = ms => new Promise(r => setTimeout(r, ms));

el('btn-start').addEventListener('click', startCamera);
el('btn-capture').addEventListener('click', handleCapture);
el('btn-retake').addEventListener('click', handleRetake);

if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(err => console.warn('SW:', err));
}
