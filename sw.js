const CACHE = 'banana-detector-v7';
const PRECACHE = ['./', './index.html', './style.css', './app.js', './manifest.json', './icon.svg'];

const PRECACHE_MODELS = ['./models/banana_yolo11s-cls.onnx', './models/metadata.json'];

self.addEventListener('install', e => {
    e.waitUntil(
        caches.open(CACHE).then(async c => {
            await c.addAll(PRECACHE);
            // Model files cached separately — a fetch failure won't abort SW install
            await Promise.allSettled(PRECACHE_MODELS.map(url => c.add(url)));
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', e => {
    e.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', e => {
    if (e.request.url.includes('jsdelivr.net') || e.request.url.includes('huggingface.co')) return;
    e.respondWith(
        caches.match(e.request).then(cached => {
            if (cached) return cached;
            return fetch(e.request).then(res => {
                const clone = res.clone();
                caches.open(CACHE).then(c => c.put(e.request, clone));
                return res;
            });
        })
    );
});
