// Hydra-Map Visualization App

const API_BASE = '/api';
let map;
let baseLayer;
let polygonsLayer;
let runId = 'new_dataset_viz_5';
let runData = null;
const FIXED_GRID_SIZE = 10; // Stable 10x10 grid for 100 tiles
const TILE_SIZE = 512;

const CLASSES = {
    0: { name: 'Background', color: '#4A5056' },
    1: { name: 'Building', color: '#9C4C4C' },
    2: { name: 'Road', color: '#B08C4C' },
    3: { name: 'Vegetation', color: '#6C8E6E' },
    4: { name: 'Water', color: '#4C6FAE' },
    5: { name: 'Other', color: '#6E5F8C' },
    6: { name: 'RCC Roof', color: '#CC5555' },
    7: { name: 'Tin Roof', color: '#8899AA' },
    8: { name: 'Tiled Roof', color: '#D4A04C' }
};

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initMap();
    populateLegend();
    loadRunData();
    // Start real-time polling every 3 seconds
    setInterval(loadRunData, 3000);
});

function initTabs() {
    const tabs = document.querySelectorAll('nav li');
    const sections = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const target = tab.getAttribute('data-tab');
            sections.forEach(sec => {
                if (sec.id === target) {
                    sec.classList.remove('hidden');
                    if (target === 'map-view' && map) map.invalidateSize();
                } else {
                    sec.classList.add('hidden');
                }
            });
        });
    });

    document.getElementById('close-inspector').addEventListener('click', () => {
        document.getElementById('inspector-panel').classList.add('collapsed');
    });
}

function initMap() {
    // Standard geographic map setup
    map = L.map('map', {
        minZoom: 2,
        maxZoom: 22
    });

    // Add OpenStreetMap base layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 22,
        maxNativeZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    baseLayer = L.layerGroup().addTo(map);
    polygonsLayer = L.layerGroup().addTo(map);
    regionsLayer = L.layerGroup().addTo(map);
}

function populateLegend() {
    const legend = document.getElementById('class-legend');
    for (let id in CLASSES) {
        if (id == 0) continue; // skip background
        const cls = CLASSES[id];
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `<div class="color-box" style="background:${cls.color}"></div> <span>${cls.name}</span>`;
        legend.appendChild(item);
    }

    // Add Replay button event
    setTimeout(() => {
        const replayBtn = document.getElementById('replay-btn');
        if (replayBtn) {
            replayBtn.addEventListener('click', () => {
                window.renderedTileIds.clear();
                baseLayer.clearLayers();
                polygonsLayer.clearLayers();
                renderMapPolygons();
            });
        }
    }, 500);
}

async function loadRunData() {
    try {
        const res = await fetch(`${API_BASE}/runs/${runId}`);
        if (!res.ok) throw new Error('Run not found');
        runData = await res.json();

        document.getElementById('run-id-display').innerText = runId;
        document.getElementById('tile-count-display').innerText = runData.processed_tiles;
        document.getElementById('metrics-run-id').innerText = runId;

        if (runData.model_versions && runData.model_versions.swin) {
            document.getElementById('model-version-display').innerText = runData.model_versions.swin;
        }

        window.regionsData = runData.regions || [];

        renderMapPolygons();
        if (typeof renderCharts === 'function') {
            renderCharts(runData);
        }
    } catch (e) {
        console.error("Failed to load run data:", e);
        document.getElementById('run-id-display').innerText = "Run not found";
    }
}

async function renderMapPolygons() {
    if (!runData || !runData.predictions) return;

    if (!window.renderedTileIds) window.renderedTileIds = new Set();

    // Collect only new (unrendered) predictions
    let newPreds = [];
    runData.predictions.forEach((pred) => {
        if (!window.renderedTileIds.has(pred.tile_id) && pred.bounds && pred.bounds[0][0] !== 0) {
            newPreds.push(pred);
        }
    });

    // Initial setup if first run
    if (window.renderedTileIds.size === 0 && newPreds.length > 0) {
        let min_y = Infinity, min_x = Infinity;
        let max_y = -Infinity, max_x = -Infinity;
        newPreds.forEach(p => {
            min_y = Math.min(min_y, p.bounds[0][0], p.bounds[1][0]);
            min_x = Math.min(min_x, p.bounds[0][1], p.bounds[1][1]);
            max_y = Math.max(max_y, p.bounds[0][0], p.bounds[1][0]);
            max_x = Math.max(max_x, p.bounds[0][1], p.bounds[1][1]);
        });
        if (min_x !== Infinity) {
            map.fitBounds([[min_y, min_x], [max_y, max_x]]);
        }
    }

    // Sequential sliding animation for new tiles
    let delay = 150; // ms between each tile reveal

    for (let idx = 0; idx < newPreds.length; idx++) {
        let pred = newPreds[idx];
        let localBounds = pred.bounds;

        // Preload tile image
        let imgOv = L.imageOverlay(`/static/tiles/512/${pred.tile_id}.png`, localBounds, {
            opacity: 1, zIndex: 1, className: 'tile-img'
        }).addTo(baseLayer);
        imgOv.on('click', () => showInspector(pred));

        window.renderedTileIds.add(pred.tile_id);

        // Draw variance filter grid outline
        let gridBox = L.rectangle(localBounds, {
            color: 'rgba(120,140,160,0.35)', weight: 1, fillOpacity: 0.0, className: 'tile-grid-box'
        }).addTo(polygonsLayer);
        gridBox.bindTooltip("Scanning: " + pred.tile_id.substring(pred.tile_id.length - 6));

        // Sequential pipeline scanning with delay
        setTimeout(() => {
            if (pred.final_class > 0) {
                let clsColor = CLASSES[pred.final_class]?.color || '#fff';
                let bl_y = pred.bounds[0][0];
                let bl_x = pred.bounds[0][1];

                // Simulated YOLO raw detection boxes
                if (pred.yolo_count > 0) {
                    let tile_h = Math.abs(pred.bounds[1][0] - pred.bounds[0][0]);
                    let tile_w = Math.abs(pred.bounds[1][1] - pred.bounds[0][1]);
                    for (let j = 0; j < Math.min(pred.yolo_count, 4); j++) {
                        let yOff = Math.random() * (tile_h * 0.8);
                        let xOff = Math.random() * (tile_w * 0.8);
                        let w = tile_w * 0.1 + Math.random() * (tile_w * 0.2);
                        let h = tile_h * 0.1 + Math.random() * (tile_h * 0.2);
                        L.rectangle([[bl_y + yOff, bl_x + xOff], [bl_y + yOff + h, bl_x + xOff + w]], {
                            color: '#6C8E6E', weight: 1, fillOpacity: 0.0, className: 'yolo-box'
                        }).addTo(polygonsLayer);
                    }
                }

                // Final Fusion classification region
                setTimeout(() => {
                    let finalBox = L.rectangle(pred.bounds, {
                        color: clsColor, weight: 1, fillOpacity: 0.45, className: 'fusion-box'
                    }).addTo(polygonsLayer);
                    finalBox.on('click', () => showInspector(pred));
                    gridBox.remove();
                }, 400);
            } else {
                // Background tile — dim the grid
                gridBox.setStyle({ color: 'rgba(120,140,160,0.15)', opacity: 0.4 });
            }

            // Pan map to follow progress every 5 tiles
            if (idx % 5 === 0) {
                let tile_h = Math.abs(pred.bounds[1][0] - pred.bounds[0][0]);
                let tile_w = Math.abs(pred.bounds[1][1] - pred.bounds[0][1]);
                map.panTo([pred.bounds[0][0] + tile_h / 2, pred.bounds[0][1] + tile_w / 2], { animate: true, duration: 0.5 });
            }

        }, idx * delay);
    }

    // Draw region GeoSAM polygons if available
    if (window.regionsData && window.regionsData.length > 0) {
        // Wait for tile animations to "finish"
        setTimeout(() => {
            window.regionsData.forEach(reg => {
                if (!reg.polygons || reg.polygons.length === 0) return;

                let clsColor = CLASSES[reg.class_id]?.color || '#fff';

                reg.polygons.forEach(p => {
                    if (p.geometry_coords && p.geometry_coords.length > 0) {
                        let localCoords = p.geometry_coords;
                        let poly = L.polygon(localCoords, {
                            color: clsColor,
                            weight: 2,
                            fillOpacity: 0.55,
                            className: 'geosam-region-poly'
                        }).addTo(regionsLayer);

                        poly.bindTooltip(`Region ${reg.region_id} <br> Confidence: ${(p.confidence * 100).toFixed(1)}%`);
                        // Optionally click on polygon to show region inspector
                    }
                });
            });
        }, newPreds.length * delay + 500);
    }
}

function showInspector(pred) {
    document.getElementById('inspector-panel').classList.remove('collapsed');
    document.getElementById('inspector-placeholder').classList.add('hidden');
    document.getElementById('inspector-details').classList.remove('hidden');

    document.getElementById('inspector-title').innerText = "Tile Details";
    document.getElementById('inspect-meta-id').innerText = pred.tile_id;

    document.getElementById('inspect-original-img').src = `/static/tiles/512/${pred.tile_id}.png`;

    document.getElementById('inspect-swin-conf').innerText = (pred.swin_confidence * 100).toFixed(1) + '%';
    document.getElementById('inspect-yolo-count').innerText = pred.yolo_count || 0;
    document.getElementById('inspect-yolo-conf').innerText = (pred.yolo_max_conf * 100).toFixed(1) + '%';
    document.getElementById('inspect-depth').innerText = pred.metrics?.z_mean ? pred.metrics.z_mean.toFixed(2) : 'N/A';

    let featuresHtml = '';

    // Fallback exactly to previous behavior if class_distribution is missing or empty
    let distributions = pred.class_distribution && pred.class_distribution.length > 0 ? pred.class_distribution : [{ class_id: pred.final_class, confidence: pred.final_confidence }];

    // Sort by confidence DESC
    distributions.sort((a, b) => b.confidence - a.confidence);

    distributions.forEach(d => {
        if (d.class_id === 0) return; // Skip background
        let clsName = CLASSES[d.class_id]?.name || 'Unknown';
        let clsColor = CLASSES[d.class_id]?.color || '#fff';
        let confStr = (d.confidence * 100).toFixed(1) + '%';

        featuresHtml += `
        <div class="decision-card" style="margin-bottom: 8px;">
            <p><span>Class</span> <strong style="color: ${clsColor}">${clsName}</strong></p>
            <p><span>Confidence</span> <strong>${confStr}</strong></p>
            <div class="progress-bar">
                <div class="fill" style="width: ${confStr}; background: ${clsColor}"></div>
            </div>
        </div>
        `;
    });

    if (featuresHtml === '') {
        featuresHtml = `
        <div class="decision-card" style="margin-bottom: 8px;">
            <p><span>Class</span> <strong style="color: #94a3b8">Background</strong></p>
            <p><span>Confidence</span> <strong>N/A</strong></p>
            <div class="progress-bar"><div class="fill" style="width: 0%"></div></div>
        </div>`;
    }

    document.getElementById('inspect-fusion-features').innerHTML = featuresHtml;

    // Region Clustering Display
    let regionHtml = '';
    let foundRegion = null;
    if (window.regionsData) {
        foundRegion = window.regionsData.find(r => r.tile_ids && r.tile_ids.includes(pred.tile_id));
    }

    let actionText = document.getElementById('inspect-action-text');
    if (foundRegion) {
        regionHtml = `
        <div style="padding: 10px; background: rgba(108,142,110,0.15); border: 1px solid rgba(108,142,110,0.3); border-radius: 6px; margin-bottom: 12px;">
            <div style="font-weight: 600; color: #6C8E6E; margin-bottom: 4px;">🎯 GeoSAM Region: ${foundRegion.region_id}</div>
            <div style="font-size: 0.9em; color: #94a3b8;">Spans ${foundRegion.tile_ids.length} tiles. Refined ${foundRegion.num_polygons} polygons.</div>
        </div>`;
        actionText.innerHTML = regionHtml + `Action: Refined by GeoSAM ✓`;
        actionText.style.color = 'var(--accent-green)';
    } else {
        actionText.innerText = pred.action === 'accept' ? 'Action: Accepted ✓' : 'Action: Needs Refinement ⚠️';
        actionText.style.color = pred.action === 'accept' ? 'var(--accent-green)' : '#f59e0b';
    }

    // Shadow Killer & Illegal Floor indicators
    let flagsHtml = '';
    if (pred.shadow_killed) {
        flagsHtml += '<div style="color:#f87171;margin-top:6px">⛔ Shadow Killer: Water rejected (height > 0.5m)</div>';
    }
    if (pred.illegal_height_flag) {
        flagsHtml += '<div style="color:#fbbf24;margin-top:6px">⚠️ Illegal Floor: Height > 12m (G+3+)</div>';
    }
    if (pred.metrics?.z_mean) {
        flagsHtml += `<div style="color:#94a3b8;margin-top:4px;font-size:0.85em">Height: ${pred.metrics.z_mean.toFixed(1)}m mean / ${(pred.metrics.z_max || 0).toFixed(1)}m max</div>`;
    }
    let flagsEl = document.getElementById('inspect-flags');
    if (!flagsEl) {
        flagsEl = document.createElement('div');
        flagsEl.id = 'inspect-flags';
        document.getElementById('inspector-details').appendChild(flagsEl);
    }
    flagsEl.innerHTML = flagsHtml;
}
