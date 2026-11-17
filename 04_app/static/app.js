/* Tsunami Trigger Predictor — front-end logic */

const form = document.getElementById('predict-form');
const predictBtn = document.getElementById('predict-btn');
const predictLabel = document.getElementById('predict-label');
const resultBox = document.getElementById('result');
const placeholder = document.getElementById('placeholder');
const errorBox = document.getElementById('error-box');

// ---------- Preset buttons ----------
document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('mag').value = btn.dataset.mag;
        document.getElementById('depth').value = btn.dataset.depth;
        document.getElementById('lat').value = btn.dataset.lat;
        document.getElementById('lon').value = btn.dataset.lon;
        document.getElementById('year').value = btn.dataset.year;
        document.getElementById('month').value = btn.dataset.month;
    });
});


/* ================== PETA (Leaflet #map) ================== */
let map = null;
let epicMarker = null;
let baseLayer = null;

// Tema tiles peta (gratis, tanpa API key; butuh internet)
const TILE_STYLES = {
    dark: {
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        attribution: 'Tiles &copy; <a href="https://www.esri.com">Esri</a> · &copy; OpenStreetMap contributors',
        subdomains: 'abc',
    },
    light: {
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        attribution: 'Tiles &copy; <a href="https://www.esri.com">Esri</a> · &copy; OpenStreetMap contributors',
        subdomains: 'abc',
    },
    osm: {
        url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        subdomains: 'abc',
    },
};

function setBaseTile(styleKey) {
    if (!map || !baseLayer) return;
    const s = TILE_STYLES[styleKey] || TILE_STYLES.dark;
    map.removeLayer(baseLayer);
    baseLayer = L.tileLayer(s.url, {
        attribution: s.attribution,
        maxZoom: 18,
        subdomains: s.subdomains,
    }).addTo(map);
    // sinkronkan tombol aktif
    document.querySelectorAll('.tile-style-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.tile === styleKey));
}

function initMap() {
    if (!window.L || !document.getElementById('map')) return;

    map = L.map('map').setView([-2, 118], 4);

    // Base tiles — default dark
    baseLayer = L.tileLayer(TILE_STYLES.dark.url, {
        attribution: TILE_STYLES.dark.attribution,
        maxZoom: 18,
        subdomains: TILE_STYLES.dark.subdomains,
    }).addTo(map);

    // Tombol pilihan tema peta
    document.querySelectorAll('.tile-style-btn').forEach(btn => {
        btn.addEventListener('click', () => setBaseTile(btn.dataset.tile));
    });

    // --- Overlay 1: Tsunami historis (150 event terbesar) ---
    const histLayer = L.layerGroup();
    fetch('/static/maps/tsunami_hist.json')
        .then(r => r.json())
        .then(data => {
            (data.points || []).forEach(pt => {
                const m = (pt.mag || 7);
                const size = Math.max(8, Math.min(16, 8 + Math.round((m - 7) * 2)));  // 8..16 px
                const color = (pt.mag || 7) >= 8.0 ? '#f43f5e' : '#f8b44d';
                const icon = L.divIcon({
                    className: '',
                    html: `<div style="width:${size}px;height:${size}px;background:${color};border-radius:50%;opacity:1;border:1px solid #fff;box-shadow:0 0 5px rgba(0,0,0,.5)"></div>`,
                    iconSize: [size, size], iconAnchor: [size / 2, size / 2],
                });
                const txt =
                    `M${pt.mag} · ${pt.year ? pt.year : '?'}${pt.c ? ' · ' + pt.c : ''}` +
                    (pt.wh != null ? `<br>WH: ${pt.wh} m` : '');
                L.marker([pt.lat, pt.lon], { icon }).addTo(histLayer)
                    .bindPopup(`<b>Tsunami historis</b><br>${txt}`);
            });
        })
        .catch(() => {});

    // --- Overlay 2: Lokasi negara (label centroid) ---
    const countryLayer = L.layerGroup();
    fetch('/static/maps/countries.json')
        .then(r => r.json())
        .then(data => {
            (data.countries || []).forEach(c => {
                const icon = L.divIcon({
                    className: '',
                    html: `<div style="color:#8fd3ff;font-size:10px;text-shadow:0 0 3px #000;white-space:nowrap">${c.name}</div>`,
                    iconSize: [0, 0], iconAnchor: [0, 10],
                });
                L.marker([c.lat, c.lon], { icon, interactive: false }).addTo(countryLayer);
            });
        })
        .catch(() => {});

    // Layer control
    L.control.layers(null, {
        'Tsunami Historis': histLayer,
        'Lokasi Negara': countryLayer,
    }, { collapsed: false, position: 'topright' }).addTo(map);

    // Tooltip utk reset view
    L.control.zoom({ position: 'bottomright' }).addTo(map);
}

function updateMap(lat, lon, prediction, warningLabel, label) {
    if (!map) return;
    // Hapus marker epicenter lama
    if (epicMarker) map.removeLayer(epicMarker);

    const lbl = label ||
        `${(prediction === 'TSUNAMI' ? 'TSUNAMI' : 'NO')} · M? · ${lat.toFixed(2)}, ${lon.toFixed(2)}`;
    const icon = L.divIcon({
        className: '',
        html: `<div class="map-marker-label">${lbl}</div><div class="map-marker-epic"></div>`,
        iconSize: [150, 40], iconAnchor: [17, 20],
    });
    const color = prediction === 'TSUNAMI' ? '#f43f5e' : '#22c55e';
    epicMarker = L.marker([lat, lon], { icon, zIndexOffset: 1000 }).addTo(map);
    epicMarker.getElement().querySelector('.map-marker-epic').style.background =
        (prediction === 'TSUNAMI' ? '#f43f5e' : '#22c55e');
    const ht = (warningLabel && prediction === 'TSUNAMI')
        ? `<br><b>Kelas:</b> ${warningLabel}` : '';
    epicMarker.bindPopup(
        `<b>⚡ Epicenter</b><br><b>${prediction}</b>${ht}<br>lat ${lat}, lon ${lon}`
    );
    map.setView([lat, lon], 6);
}

// Inisialisasi saat halaman siap
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMap);
} else {
    initMap();
}

// ---------- Submit ----------
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(true);
    hide(errorBox);

    try {
        const resp = await fetch('/predict', {
            method: 'POST',
            body: new FormData(form),
        });
        const data = await resp.json();

        if (!resp.ok || !data.success) {
            showError(data.error || 'Terjadi kesalahan pada server.');
        } else {
            renderResult(data);
        }
    } catch (err) {
        showError('Tidak dapat menghubungi server: ' + err.message);
    } finally {
        setLoading(false);
    }
});

function setLoading(on) {
    predictBtn.disabled = on;
    predictLabel.textContent = on ? 'Menghitung…' : 'PREDICT';
}

function showError(msg) {
    document.getElementById('error-text').textContent = msg;
    show(errorBox);
    hide(resultBox);
    hide(document.getElementById('warning-box'));
    placeholder.classList.add('hidden');
}

function renderResult(data) {
    hide(placeholder);
    show(resultBox);

    // Peta: tampilkan marker epicenter dari koordinat hasil
    if (data.derived && typeof data.derived.lat === 'number' && typeof data.derived.lon === 'number') {
        const lbl = `${data.prediction === 'TSUNAMI' ? 'TSUNAMI' : 'NO'} · M${data.derived.mag != null ? data.derived.mag : '?'} · ${data.derived.lat.toFixed(2)}, ${data.derived.lon.toFixed(2)}`;
        const wl = data.warning ? `${data.warning.label} (${data.warning.range_m})` : null;
        updateMap(data.derived.lat, data.derived.lon, data.prediction, wl, lbl);
    }

    // Risk badge
    const badge = document.getElementById('risk-badge');
    badge.textContent = `TSUNAMI RISK: ${data.risk_label}`;
    badge.className = 'risk-badge ' + data.risk_class;

    // Gauge
    const pct = data.probability * 100;
    const fill = document.getElementById('gauge-fill');
    fill.style.width = pct.toFixed(1) + '%';
    document.getElementById('prob-num').textContent = pct.toFixed(2);

    // Threshold marker position
    document.getElementById('threshold-mark').style.left =
        (data.threshold * 100).toFixed(1) + '%';

    // Meta
    document.getElementById('meta-model').textContent = data.main_model;
    document.getElementById('meta-threshold').textContent =
        parseFloat(data.threshold).toFixed(2);

    // Estimasi ordinal: kelas tinggi air + probabilitas tiap kelas
    const warningBox = document.getElementById('warning-box');
    if (data.warning) {
        const applicable = data.warning.applicable !== false
            && data.prediction === 'TSUNAMI';
        document.getElementById('warning-class').textContent =
            applicable ? data.warning.label : 'Tidak berlaku';
        document.getElementById('warning-range').textContent =
            applicable ? '(' + data.warning.range_m + ')' : '';
        document.getElementById('warning-note').textContent = applicable
            ? ''
            : 'klasifikasi menyatakan bukan tsunami — kelas tidak dipakai';
        const probs = data.warning.probabilities || {};
        document.getElementById('class-probs').innerHTML = applicable
            ? Object.entries(probs).map(
                ([k, v]) => `<span class="class-prob"><b>${k}</b> ${(v * 100).toFixed(2)}%</span>`
              ).join('')
            : '';
        show(warningBox);
    } else {
        hide(warningBox);
    }

    // Derived features
    const d = data.derived;
    const items = [
        ['Tectonic zone', d.tectonic_zone],
        ['Era', d.era],
        ['Season', d.season],
        ['Magnitude', d.mag],
        ['Depth', d.depth + ' km'],
    ];
    document.getElementById('derived-list').innerHTML = items.map(
        ([k, v]) => `<li><span class="k">${k}</span><span class="v">${v}</span></li>`
    ).join('');

    // Comparison table
    const tbody = document.getElementById('cmp-body');
    tbody.innerHTML = data.comparison.map(c => {
        const cls = c.prediction === 'TSUNAMI' ? 'decision-tsunami' : 'decision-no';
        const isMain = c.name === data.main_model ? ' class="main-model"' : '';
        const h = c.warning_label
            ? `${c.warning_label} <span class="rule-tag">${c.warning_rule}</span>` : '—';
        return `<tr${isMain}>
            <td>${c.name}${isMain ? ' ★' : ''}</td>
            <td class="num">${(c.probability * 100).toFixed(2)}%</td>
            <td class="num">${c.threshold.toFixed(2)}</td>
            <td><span class="${cls}">${c.prediction}</span></td>
            <td class="num">${h}</td>
        </tr>`;
    }).join('');
}

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }
