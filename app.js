import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/+esm';

env.allowLocalModels = false;
env.useBrowserCache = true;

// Serve WASM from CDN; disable threading (avoids COOP/COEP header requirement)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
ort.env.wasm.numThreads = 1;

const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);

const STAGES = [
    { id:'unripe',   emoji:'🟢', title:'Unripe — Not Yet',       desc:'This banana needs more time. Leave it at room temperature for 3–5 days until it turns yellow.',            pos: 7  },
    { id:'nearly',   emoji:'🟡', title:'Almost Ready',             desc:"Getting close! Give it 1–2 more days and it'll be perfect.",                                               pos: 28 },
    { id:'perfect',  emoji:'⭐',  title:'Perfect to Eat!',           desc:'This is the sweet spot — bright yellow, firm, and at peak sweetness. Eat it now!',                      pos: 50 },
    { id:'ripe',     emoji:'✅',  title:'Ripe & Very Sweet',          desc:'The brown spots mean extra sugar. Great for eating fresh, in smoothies, or frozen.',                   pos: 72 },
    { id:'overripe', emoji:'🍞', title:'Overripe — Bake It!',    desc:"Too soft to eat fresh, but perfect for banana bread, muffins, or pancakes. Don't throw it out!",       pos: 92 }
];

let classifierSession = null;
let classifierClasses = null;
let detector     = null;
let cocoModel    = null;
let modelLoading = false;
let cameraStream = null;

const el    = id => document.getElementById(id);
const show  = (...ids) => ids.forEach(id => el(id)?.classList.remove('hidden'));
const hide  = (...ids) => ids.forEach(id => el(id)?.classList.add('hidden'));
const delay = ms => new Promise(r => setTimeout(r, ms));

// ---- Splash helpers ----

function setSplash(text, pct = null) {
    const s = el('splash-status'), f = el('splash-fill');
    if (s) s.textContent = text;
    if (f && pct != null) f.style.width = `${pct}%`;
}

function hideSplash() {
    const s = el('splash');
    if (!s) return;
    s.classList.add('splash-out');
    setTimeout(() => s.classList.add('hidden'), 450);
}

function setModeIndicator(text) {
    const ind = el('mode-indicator');
    if (ind) { ind.textContent = text; ind.classList.remove('hidden'); }
}

// ---- Classifier ----

async function loadClassifier() {
    const meta = await fetch('./models/metadata.json').then(r => r.json());
    classifierClasses = meta.classes;

    // Stream the model file so we can show download progress
    const url = `./models/${meta.onnx_file}`;
    const res = await fetch(url);
    const total = parseInt(res.headers.get('content-length') || '0');
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (total) setSplash(`Downloading banana brain… ${Math.round(received / total * 100)}%`, 5 + (received / total) * 75);
    }
    const buffer = new Uint8Array(received);
    let pos = 0;
    for (const chunk of chunks) { buffer.set(chunk, pos); pos += chunk.length; }

    setSplash('Initialising model…', 82);
    classifierSession = await ort.InferenceSession.create(buffer.buffer, {
        executionProviders: ['wasm']
    });
}

function preprocessForClassifier(canvas, size = 224) {
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    c.getContext('2d').drawImage(canvas, 0, 0, size, size);
    const { data } = c.getContext('2d').getImageData(0, 0, size, size);
    const n = size * size;
    const t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        t[i]         = data[i * 4]     / 255;
        t[n + i]     = data[i * 4 + 1] / 255;
        t[2 * n + i] = data[i * 4 + 2] / 255;
    }
    return new ort.Tensor('float32', t, [1, 3, size, size]);
}

async function classifyRipeness(cropCanvas) {
    if (!classifierSession || !classifierClasses) return null;
    try {
        const tensor = preprocessForClassifier(cropCanvas);
        const { output0 } = await classifierSession.run({ images: tensor });
        const logits = output0.data;
        const max = Math.max(...logits);
        const exps = Array.from(logits).map(x => Math.exp(x - max));
        const sum  = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(x => x / sum);
        const idx = probs.indexOf(Math.max(...probs));
        const stage = STAGES.find(s => s.id === classifierClasses[idx]) ?? STAGES[2];
        return { stage, confidence: probs[idx] };
    } catch (err) {
        console.warn('Classifier error:', err);
        return null;
    }
}

// ---- Detector (bounding boxes) ----

function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
        const s = document.createElement('script');
        s.src = src; s.onload = resolve; s.onerror = reject;
        document.head.appendChild(s);
    });
}

async function loadCocoSsd() {
    await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
    await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js');
    await window.tf.setBackend('webgl');
    cocoModel = await window.cocoSsd.load({ base: 'lite_mobilenet_v2' });
}

async function loadYolos() {
    detector = await pipeline('object-detection', 'Xenova/yolos-tiny', {
        device: 'wasm',
        progress_callback: ({ status, progress }) => {
            if (status === 'progress' && progress != null)
                setModeIndicator(`Banana-v1 · loading detector ${Math.round(progress)}%…`);
        }
    });
}

async function loadModels() {
    if (classifierSession || modelLoading) return;
    modelLoading = true;
    try {
        setSplash('Loading banana brain… (~6 MB, cached after this)', 5);
        await loadClassifier();
        setModeIndicator('Banana-v1 · 99.1% accuracy');
        setSplash('✓ Ready', 100);
        await delay(300);
        hideSplash();
        // Detector loads in background — adds bounding boxes once ready
        (isIOS ? loadCocoSsd() : loadYolos())
            .then(() => setModeIndicator('Banana-v1 · ' + (isIOS ? 'COCO-SSD' : 'YOLOS-tiny') + ' · 99.1%'))
            .catch(err => console.warn('Detector unavailable:', err));
    } catch (err) {
        console.error('Classifier load failed:', err);
        setModeIndicator('Colour analysis only');
        setSplash('⚠ Model failed — colour analysis only', 100);
        await delay(2500);
        hideSplash();
    } finally {
        modelLoading = false;
    }
}

// ---- Camera ----

async function startCamera() {
    hide('btn-start', 'start-screen', 'result', 'shelf-result');
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode:'environment', width:{ ideal:1280 }, height:{ ideal:960 } }
        });
        el('video').srcObject = cameraStream;
        show('camera-view', 'btn-capture');
    } catch {
        alert('Camera access was denied. Please allow camera access and reload the page.');
        show('btn-start', 'start-screen');
    }
}

function captureFrame() {
    const video = el('video'), canvas = el('canvas');
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas;
}

function stopCamera() {
    cameraStream?.getTracks().forEach(t => t.stop());
    cameraStream = null;
}

// ---- Detection ----

async function runDetection(canvas) {
    if (cocoModel) {
        const preds = await cocoModel.detect(canvas);
        return preds
            .filter(p => p.class === 'banana' && p.score > 0.45)
            .map(p => ({
                label: p.class, score: p.score,
                box: { x: Math.round(p.bbox[0]), y: Math.round(p.bbox[1]),
                       w: Math.round(p.bbox[2]), h: Math.round(p.bbox[3]) }
            }));
    }
    if (detector) {
        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
        const preds = await detector(dataUrl, { threshold: 0.45 });
        return preds
            .filter(p => p.label === 'banana')
            .map(p => ({
                label: p.label, score: p.score,
                box: { x: Math.round(p.box.xmin), y: Math.round(p.box.ymin),
                       w: Math.round(p.box.xmax - p.box.xmin), h: Math.round(p.box.ymax - p.box.ymin) }
            }));
    }
    return null;
}

function cropCanvas(src, box) {
    const c = document.createElement('canvas');
    c.width = box.w; c.height = box.h;
    c.getContext('2d').drawImage(src, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
    return c;
}

// ---- Colour analysis (fallback when classifier unavailable) ----

function rgbToHsl(r,g,b) {
    r/=255; g/=255; b/=255;
    const max=Math.max(r,g,b), min=Math.min(r,g,b), l=(max+min)/2;
    if (max===min) return [0,0,l];
    const d=max-min, s=l>0.5?d/(2-max-min):d/(max+min);
    let h;
    switch(max){ case r: h=((g-b)/d+(g<b?6:0))/6; break; case g: h=((b-r)/d+2)/6; break; default: h=((r-g)/d+4)/6; }
    return [h*360,s,l];
}

function analyzeRipeness(ctx, w, h, pad = 0.2) {
    const { data } = ctx.getImageData(
        Math.floor(w*pad), Math.floor(h*pad),
        Math.floor(w*(1-pad*2)), Math.floor(h*(1-pad*2))
    );
    const counts=[0,0,0,0,0]; let relevant=0;
    for (let i=0; i<data.length; i+=4) {
        const [hue,s,l]=rgbToHsl(data[i],data[i+1],data[i+2]);
        if (s<0.18||l<0.22||l>0.88) continue;
        relevant++;
        if      (hue>=85&&hue<=155) counts[0]++;
        else if (hue>=60&&hue<85)   counts[1]++;
        else if (hue>=44&&hue<60)   counts[2]++;
        else if (hue>=25&&hue<44)   counts[3]++;
        else if (hue<25||hue>330)   counts[4]++;
    }
    if (relevant<80) return null;
    const t=counts.reduce((a,b)=>a+b,0)||1;
    return counts[0]/t*0.05+counts[1]/t*0.28+counts[2]/t*0.52+counts[3]/t*0.76+counts[4]/t*0.95;
}

function stageFromScore(s) {
    if (s<0.18) return STAGES[0]; if (s<0.38) return STAGES[1];
    if (s<0.58) return STAGES[2]; if (s<0.78) return STAGES[3];
    return STAGES[4];
}

function hslFallback(canvas, pad) {
    const score = analyzeRipeness(canvas.getContext('2d'), canvas.width, canvas.height, pad);
    return score !== null ? stageFromScore(score) : null;
}

// ---- Annotation ----

function drawBananaBoxes(canvas, ranked) {
    const ctx = canvas.getContext('2d');
    ctx.save();
    ranked.forEach((b, i) => {
        const isFirst = i === 0;
        const { x, y, w, h } = b.box;
        ctx.shadowColor = 'rgba(0,0,0,0.7)';
        ctx.shadowBlur = 5;
        ctx.strokeStyle = isFirst ? '#FFD700' : '#ffffff';
        ctx.lineWidth = isFirst ? 4 : 2.5;
        ctx.strokeRect(x, y, w, h);
        const label = `${i+1}. ${b.stage.emoji} ${b.stage.title.split(' —')[0].replace(/!/g,'')}`;
        const fontSize = Math.max(13, Math.min(20, w / 7));
        ctx.font = `bold ${fontSize}px -apple-system, sans-serif`;
        ctx.shadowBlur = 0;
        const tw = ctx.measureText(label).width + 12;
        const lh = fontSize + 8;
        const lx = Math.max(0, x);
        const ly = y > lh + 4 ? y - 4 : y + h + lh;
        ctx.fillStyle = 'rgba(0,0,0,0.72)';
        ctx.fillRect(lx, ly - fontSize - 2, tw, lh);
        ctx.fillStyle = isFirst ? '#FFD700' : '#ffffff';
        ctx.fillText(label, lx + 6, ly + 2);
    });
    ctx.restore();
}

function showShelfResult(ranked) {
    const div = el('shelf-result');
    if (!div) return;
    if (ranked.length === 0) { hide('shelf-result'); return; }
    div.innerHTML = ranked.map((b, i) =>
        `<div class="shelf-item${i===0?' shelf-best':''}">
            <span class="shelf-rank">${i+1}</span>
            <span class="shelf-emoji" aria-hidden="true">${b.stage.emoji}</span>
            <span class="shelf-label">${b.stage.title}</span>
        </div>`
    ).join('');
    show('shelf-result');
}

// ---- Main analyze ----

async function analyze(canvas) {
    hide('result', 'shelf-result');
    show('loading');
    el('loading-text').textContent = 'Detecting bananas…';
    el('progress-fill').style.width = '20%';

    let bananas = null;
    try { bananas = await runDetection(canvas); }
    catch (err) { console.warn('Detection error:', err); }
    el('progress-fill').style.width = '50%';

    if (bananas && bananas.length > 0) {
        el('loading-text').textContent = `Classifying ${bananas.length} banana${bananas.length>1?'s':''}…`;

        const analysed = await Promise.all(bananas.map(async b => {
            const crop = cropCanvas(canvas, b.box);
            const cls  = await classifyRipeness(crop);
            const stage = cls?.stage ?? hslFallback(crop, 0.08) ?? STAGES[2];
            return { ...b, stage };
        }));

        analysed.sort((a, b) => Math.abs(a.stage.pos - 50) - Math.abs(b.stage.pos - 50));

        const annotated = document.createElement('canvas');
        annotated.width = canvas.width; annotated.height = canvas.height;
        annotated.getContext('2d').drawImage(canvas, 0, 0);
        drawBananaBoxes(annotated, analysed);

        el('progress-fill').style.width = '100%';
        await delay(200);
        hide('loading');

        el('preview-img').src = annotated.toDataURL('image/jpeg', 0.92);
        showShelfResult(analysed);

        const best = analysed[0];
        const detLabel = isIOS ? 'COCO-SSD' : 'YOLOS-tiny';
        showResult(best.stage.emoji, best.stage.title, best.stage.desc, best.stage.pos,
            `Banana-v1 · ${detLabel} · ${analysed.length} banana${analysed.length>1?'s':''}`);

    } else {
        el('loading-text').textContent = 'Analysing…';

        if (bananas !== null && bananas.length === 0) {
            el('progress-fill').style.width = '100%';
            await delay(200);
            hide('loading');
            return showResult('🙅🍌', 'Not Banana',
                "That's definitely not a banana. Point the camera at a banana and make sure it's well-lit.",
                50, (isIOS ? 'COCO-SSD' : 'YOLOS-tiny') + ' · no banana detected');
        }

        // No detector loaded — classify whole frame
        const cls   = await classifyRipeness(canvas);
        const stage = cls?.stage ?? hslFallback(canvas, 0.2);
        el('progress-fill').style.width = '100%';
        await delay(200);
        hide('loading');

        if (!stage) {
            return showResult('❓', 'Try Again',
                'Make sure the banana is well-lit and fills the frame.', 50, 'Analysis failed');
        }
        showResult(stage.emoji, stage.title, stage.desc, stage.pos,
            cls ? 'Banana-v1 · whole frame' : 'Colour analysis');
    }
}

function showResult(emoji, title, desc, pos, info) {
    el('result-emoji').textContent = emoji;
    el('result-title').textContent = title;
    el('result-desc').textContent = desc;
    el('ripeness-indicator').style.left = `${pos}%`;
    el('model-info').textContent = info;
    show('result');
    el('result').scrollIntoView({ behavior:'smooth', block:'nearest' });
}

async function handleCapture() {
    const canvas = captureFrame();
    el('preview-img').src = canvas.toDataURL('image/jpeg', 0.92);
    hide('camera-view', 'btn-capture');
    show('preview-view', 'btn-retake');
    stopCamera();
    await analyze(canvas);
}

function handleRetake() {
    hide('preview-view', 'result', 'btn-retake', 'loading', 'shelf-result');
    startCamera();
}

// ---- Samples ----

function drawSample(stageIdx) {
    const c=document.createElement('canvas'); c.width=320; c.height=240;
    const ctx=c.getContext('2d');
    ctx.fillStyle='#d0d0d0'; ctx.fillRect(0,0,320,240);
    const body=['#5aab1a','#b0cc28','#ffd700','#e8a020','#7a4e1e'][stageIdx];
    const tip =['#2d6608','#7a9010','#b89000','#9a5800','#3e2008'][stageIdx];
    ctx.save(); ctx.translate(160,118); ctx.rotate(-0.12);
    ctx.beginPath();
    ctx.moveTo(-105,18); ctx.bezierCurveTo(-95,-58,95,-58,105,18); ctx.bezierCurveTo(92,32,-92,32,-105,18);
    ctx.fillStyle=body; ctx.fill(); ctx.strokeStyle=tip; ctx.lineWidth=1.5; ctx.stroke();
    if (stageIdx>=3) {
        ctx.fillStyle=tip+'cc';
        const n=stageIdx===3?5:11;
        for (let i=0;i<n;i++) {
            ctx.beginPath();
            ctx.ellipse(-70+Math.cos(i*2.44)*58,-4+Math.sin(i*1.3)*10,5+i%3,3.5,0.4+i*0.2,0,Math.PI*2);
            ctx.fill();
        }
    }
    ctx.beginPath(); ctx.moveTo(98,12); ctx.quadraticCurveTo(112,0,108,-12);
    ctx.strokeStyle=tip; ctx.lineWidth=5; ctx.lineCap='round'; ctx.stroke();
    ctx.restore();
    return c;
}

function initSamples() {
    const grid=el('samples-grid'); if (!grid) return;
    STAGES.forEach((stage, i) => {
        const btn=document.createElement('button');
        btn.className='sample-thumb'; btn.title=stage.title;
        btn.setAttribute('aria-label', stage.title);
        const sampleCanvas = drawSample(i);
        const img=document.createElement('img');
        img.src = sampleCanvas.toDataURL('image/jpeg', 0.92);
        img.alt=stage.title;
        const lbl=document.createElement('span');
        lbl.textContent=stage.emoji+' '+stage.title.split(' —')[0].replace('!','');
        btn.appendChild(img); btn.appendChild(lbl);
        btn.addEventListener('click', () => loadSample(sampleCanvas));
        grid.appendChild(btn);
    });
}

async function loadSample(sampleCanvas) {
    el('preview-img').src = sampleCanvas.toDataURL('image/jpeg', 0.92);
    hide('camera-view', 'btn-capture', 'start-screen', 'result', 'shelf-result');
    show('preview-view', 'btn-retake');
    stopCamera();
    await analyze(sampleCanvas);
}

// ---- Boot ----

el('btn-start').addEventListener('click', startCamera);
el('btn-capture').addEventListener('click', handleCapture);
el('btn-retake').addEventListener('click', handleRetake);

initSamples();
loadModels();

if ('serviceWorker' in navigator)
    navigator.serviceWorker.register('sw.js').catch(e => console.warn('SW:', e));
