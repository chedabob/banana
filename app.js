import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache = true;

const MOBILENET_MODEL = 'Xenova/mobilenet-v2';
const SMOLVLM_MODEL   = 'HuggingFaceTB/SmolVLM-256M-Instruct';
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);

async function checkWebGPUViable() {
    if (isIOS) return false;
    if (!('gpu' in navigator)) return false;
    try { return !!(await navigator.gpu.requestAdapter()); } catch { return false; }
}

const STAGES = [
    { id:'unripe',   emoji:'🟢', title:'Unripe — Not Yet',     desc:'This banana needs more time. Leave it at room temperature for 3–5 days until it turns yellow.',            pos: 7  },
    { id:'nearly',   emoji:'🟡', title:'Almost Ready',           desc:"Getting close! Give it 1–2 more days and it'll be perfect.",                                           pos: 28 },
    { id:'perfect',  emoji:'⭐',  title:'Perfect to Eat!',         desc:'This is the sweet spot — bright yellow, firm, and at peak sweetness. Eat it now!',                  pos: 50 },
    { id:'ripe',     emoji:'✅',  title:'Ripe & Very Sweet',        desc:'The brown spots mean extra sugar. Great for eating fresh, in smoothies, or frozen.',               pos: 72 },
    { id:'overripe', emoji:'🍞', title:'Overripe — Bake It!',  desc:"Too soft to eat fresh, but perfect for banana bread, muffins, or pancakes. Don't throw it out!", pos: 92 }
];

let vlmPipe = null, classifier = null, modelLoading = false, useVLM = false, cameraStream = null;

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

// ---- Model loading (runs on startup, blocks via splash) ----

async function loadMobileNet() {
    setSplash('⏳ Downloading MobileNet (~13 MB)…', 5);
    try {
        classifier = await pipeline('image-classification', MOBILENET_MODEL, {
            progress_callback: ({ status, progress }) => {
                if (status === 'progress' && progress != null) {
                    setSplash(`⏳ Downloading MobileNet… ${Math.round(progress)}%`, 5 + progress * 0.88);
                }
            }
        });
        setModeIndicator('MobileNet-v2 + colour analysis');
        setSplash('✓ MobileNet loaded', 100);
        await delay(400);
    } catch (err) {
        console.error('MobileNet load error:', err);
        setModeIndicator('Colour analysis only');
        setSplash(`! Model failed — will use colour analysis\n(${(err?.message ?? String(err)).slice(0,100)})`, 100);
        await delay(2500);
    }
}

async function loadSmolVLM() {
    setSplash('⏳ Loading SmolVLM-256M (~200 MB, cached after this)…', 3);
    try {
        vlmPipe = await pipeline('image-text-to-text', SMOLVLM_MODEL, {
            device: 'webgpu',
            dtype: { embed_tokens:'q8', vision_encoder:'q8', decoder_model_merged:'q4' },
            progress_callback: ({ status, progress }) => {
                if (status === 'progress' && progress != null) {
                    setSplash(`⏳ SmolVLM: ${Math.round(progress)}%`, 3 + progress * 0.9);
                }
            }
        });
        useVLM = true;
        setModeIndicator('SmolVLM-256M · WebGPU');
        setSplash('✓ SmolVLM ready', 100);
        await delay(400);
    } catch (err) {
        console.error('SmolVLM failed:', err);
        setSplash('! SmolVLM failed — loading MobileNet instead…', 50);
        vlmPipe = null; useVLM = false;
        await delay(800);
        await loadMobileNet();
    }
}

async function loadModel() {
    if (vlmPipe || classifier || modelLoading) return;
    modelLoading = true;
    try {
        if (isIOS) {
            setSplash('⏳ Setting up…', 40);
            await delay(150);
            setModeIndicator('Colour analysis · iOS');
            setSplash('✓ Ready', 100);
            await delay(350);
        } else {
            const gpuViable = await checkWebGPUViable();
            if (gpuViable) await loadSmolVLM(); else await loadMobileNet();
        }
    } finally {
        modelLoading = false;
        hideSplash();
    }
}

// ---- Camera ----

async function startCamera() {
    hide('btn-start', 'start-screen', 'result');
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
    return canvas.toDataURL('image/jpeg', 0.92);
}

function stopCamera() {
    cameraStream?.getTracks().forEach(t => t.stop());
    cameraStream = null;
}

// ---- VLM path ----

async function analyzeWithVLM(dataUrl) {
    const messages = [{ role:'user', content:[
        { type:'image', url: dataUrl },
        { type:'text',  text:'Is there a banana in this image? If yes, assess its ripeness as exactly one of: unripe, nearly-ripe, perfect, ripe, overripe. Reply with the ripeness word first, then one sentence describing what you see. If no banana: not-a-banana.' }
    ]}];
    const out = await vlmPipe(messages, { max_new_tokens:80 });
    const last = out[0].generated_text.at(-1);
    const c = last.content;
    return (typeof c==='string'?c:Array.isArray(c)?(c.find(x=>x.type==='text')?.text??''):String(c)).trim();
}

function parseVLMResponse(text) {
    const lo = text.toLowerCase();
    if (/not.a.banana|no banana|cannot see|don.t see|i see no/i.test(lo)) return { stage:null, explanation:text };
    let stage;
    if      (/over.?ripe/i.test(lo))                   stage = STAGES[4];
    else if (/nearly.?ripe|almost ripe/i.test(lo))     stage = STAGES[1];
    else if (/\bperfect/i.test(lo))                    stage = STAGES[2];
    else if (/unripe|un.ripe|not.?ripe/i.test(lo))     stage = STAGES[0];
    else if (/\bripe\b/i.test(lo))                     stage = STAGES[3];
    else                                                stage = STAGES[2];
    const explanation = text.replace(/^(overripe|over-ripe|nearly-ripe|nearly ripe|perfect|unripe|ripe)[,\.\s]*/i,'').trim();
    return { stage, explanation: explanation.length>10 ? explanation : stage.desc };
}

// ---- Colour analysis ----

function rgbToHsl(r,g,b) {
    r/=255; g/=255; b/=255;
    const max=Math.max(r,g,b), min=Math.min(r,g,b), l=(max+min)/2;
    if (max===min) return [0,0,l];
    const d=max-min, s=l>0.5?d/(2-max-min):d/(max+min);
    let h;
    switch(max){ case r: h=((g-b)/d+(g<b?6:0))/6; break; case g: h=((b-r)/d+2)/6; break; default: h=((r-g)/d+4)/6; }
    return [h*360,s,l];
}

function analyzeRipeness(ctx, w, h) {
    const pad=0.2;
    const { data } = ctx.getImageData(Math.floor(w*pad), Math.floor(h*pad), Math.floor(w*(1-pad*2)), Math.floor(h*(1-pad*2)));
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
    if (relevant<100) return null;
    const t=counts.reduce((a,b)=>a+b,0)||1;
    return counts[0]/t*0.05+counts[1]/t*0.28+counts[2]/t*0.52+counts[3]/t*0.76+counts[4]/t*0.95;
}

function stageFromScore(s) {
    if (s<0.18) return STAGES[0]; if (s<0.38) return STAGES[1];
    if (s<0.58) return STAGES[2]; if (s<0.78) return STAGES[3];
    return STAGES[4];
}

async function analyzeFallback(dataUrl) {
    show('loading'); hide('result');
    el('loading-text').textContent = 'Analysing colours…';
    el('progress-fill').style.width = '30%';

    const canvas=el('canvas'), ctx=canvas.getContext('2d');
    const colorScore=analyzeRipeness(ctx, canvas.width, canvas.height);
    el('progress-fill').style.width = '65%';

    let isBanana=true, modelInfo=isIOS?'Colour analysis · iOS':'Colour analysis';

    if (classifier) {
        try {
            el('loading-text').textContent='Running vision model…';
            const results=await classifier(dataUrl,{topk:5});
            el('progress-fill').style.width='90%';
            const hit=results.find(r=>r.label.toLowerCase().includes('banana'));
            if (hit) {
                modelInfo=`MobileNet-v2: banana ${(hit.score*100).toFixed(0)}% confident`;
            } else {
                isBanana=false;
                modelInfo=`MobileNet-v2: top result was "${results[0].label}"`;
            }
        } catch { modelInfo='Colour analysis (model error)'; }
    }

    el('progress-fill').style.width='100%';
    await delay(200); hide('loading');

    if (!isBanana) return showResult('❓',"That Doesn't Look Like a Banana",
        "Point the camera at a banana, make sure it's well-lit and inside the box.",50,modelInfo);
    if (colorScore===null) return showResult('❓','Try Again',
        'Move the banana inside the frame and make sure it\'s well-lit.',50,modelInfo);

    const stage=stageFromScore(colorScore);
    showResult(stage.emoji, stage.title, stage.desc, stage.pos, modelInfo);
}

async function analyze(dataUrl) {
    if (useVLM && vlmPipe) {
        show('loading'); hide('result');
        el('loading-text').textContent='Thinking with SmolVLM…';
        el('progress-fill').style.width='35%';
        try {
            const text=await analyzeWithVLM(dataUrl);
            el('progress-fill').style.width='100%'; await delay(200); hide('loading');
            const {stage,explanation}=parseVLMResponse(text);
            if (!stage) return showResult('❓',"That Doesn't Look Like a Banana",'Point the camera at a banana and try again.',50,'SmolVLM-256M (WebGPU)');
            showResult(stage.emoji, stage.title, explanation, stage.pos, 'SmolVLM-256M (WebGPU)');
        } catch(err) {
            console.warn('VLM inference failed:',err); hide('loading');
            await analyzeFallback(dataUrl);
        }
    } else {
        await analyzeFallback(dataUrl);
    }
}

function showResult(emoji,title,desc,pos,info) {
    el('result-emoji').textContent=emoji;
    el('result-title').textContent=title;
    el('result-desc').textContent=desc;
    el('ripeness-indicator').style.left=`${pos}%`;
    el('model-info').textContent=info;
    show('result');
    el('result').scrollIntoView({behavior:'smooth',block:'nearest'});
}

async function handleCapture() {
    const dataUrl=captureFrame();
    el('preview-img').src=dataUrl;
    hide('camera-view','btn-capture');
    show('preview-view','btn-retake');
    stopCamera();
    await analyze(dataUrl);
}

function handleRetake() {
    hide('preview-view','result','btn-retake','loading');
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
    return c.toDataURL('image/jpeg',0.92);
}

function initSamples() {
    const grid=el('samples-grid'); if (!grid) return;
    STAGES.forEach((stage,i)=>{
        const btn=document.createElement('button');
        btn.className='sample-thumb'; btn.title=stage.title;
        btn.setAttribute('aria-label', stage.title);
        const img=document.createElement('img'); img.src=drawSample(i); img.alt=stage.title;
        const lbl=document.createElement('span'); lbl.textContent=stage.emoji+' '+stage.title.split(' —')[0].replace('!','');
        btn.appendChild(img); btn.appendChild(lbl);
        btn.addEventListener('click',()=>loadSample(img.src));
        grid.appendChild(btn);
    });
}

async function loadSample(dataUrl) {
    const img=new Image();
    img.onload=async ()=>{
        const canvas=el('canvas'); canvas.width=img.width; canvas.height=img.height;
        canvas.getContext('2d').drawImage(img,0,0);
        el('preview-img').src=dataUrl;
        hide('camera-view','btn-capture','start-screen','result');
        show('preview-view','btn-retake');
        stopCamera();
        await analyze(dataUrl);
    };
    img.src=dataUrl;
}

// ---- Boot ----

el('btn-start').addEventListener('click', startCamera);
el('btn-capture').addEventListener('click', handleCapture);
el('btn-retake').addEventListener('click', handleRetake);

initSamples();
loadModel();

if ('serviceWorker' in navigator)
    navigator.serviceWorker.register('sw.js').catch(e=>console.warn('SW:',e));
