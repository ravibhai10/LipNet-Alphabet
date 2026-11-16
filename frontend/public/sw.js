const CACHE = 'lipnet-shell-v1'
self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(['/','/index.html'])))
  self.skipWaiting()
})
self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim())
})
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET' || !event.request.url.startsWith(self.location.origin)) return
  const cacheable = ['document', 'script', 'style', 'image', 'font'].includes(event.request.destination)
  if (!cacheable) return
  event.respondWith(caches.match(event.request).then(cached => {
    const network = fetch(event.request).then(response => {
      if (response.ok) {
        const copy = response.clone()
        void caches.open(CACHE).then(cache => cache.put(event.request, copy))
      }
      return response
    }).catch(() => cached || caches.match('/index.html'))
    return cached || network
  }))
})
