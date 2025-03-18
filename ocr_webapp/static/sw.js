// Nazwa cache
const CACHE_NAME = 'moja-pwa-v1';

// Pliki do cache
const urlsToCache = [
  '/',
  '/static/style.css',
  '/static/js/main.js',
  '/static/manifest.json',
//   '/static/images/icon-192x192.png',
//   '/static/images/icon-512x512.png'
];

// Instalacja Service Worker
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

// Fetch - obsługa żądań
self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
  );
});