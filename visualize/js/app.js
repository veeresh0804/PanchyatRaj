// Hydra-Map Control Center App

const API_BASE = 'http://localhost:8000'; // Target FastAPI Backend
let map;
let baseLayer;
let polygonsLayer;
let swinLayer;
let yoloLayer;
let regionsLayer;

let currentRunId = null;
let pipelineSocket = null;
let deckgl = null;
let is3DView = false;

const CLASSES = {
    0: { name: 'Background', color: '#30363D' },
    1: { name: 'Building', color: '#F85149' },
    2: { name: 'Road', color: '#D29922' },
    3: { name: 'Vegetation', color: '#3FB950' },
    4: { name: 'Water', color: '#58A6FF' },
    5: { name: 'Other', color: '#BC8CFF' },
};

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initLayerToggles();
});

function initMap() {
    // Cyberpunk/Dark map styling
    map = L.map('map', {
        minZoom: 4,
        maxZoom: 22,
        zoomControl: false // Using custom controls
    });

    // Minimal dark matter basemap
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 22
    }).addTo(map);

    map.setView([17.3, 78.4], 14); // Default view, can be updated via dataset

    // Layers
    baseLayer = L.layerGroup().addTo(map);
    swinLayer = L.layerGroup().addTo(map);
    yoloLayer = L.layerGroup().addTo(map);
    polygonsLayer = L.layerGroup().addTo(map);
    regionsLayer = L.layerGroup().addTo(map);
}

function initLayerToggles() {
    document.getElementById('layer-swin').addEventListener('change', (e) => toggleLayer(swinLayer, e.target.checked));
    document.getElementById('layer-yolo').addEventListener('change', (e) => toggleLayer(yoloLayer, e.target.checked));
    document.getElementById('layer-fusion').addEventListener('change', (e) => toggleLayer(polygonsLayer, e.target.checked));
}

function toggleLayer(layerGroup, show) {
    if (show) {
        map.addLayer(layerGroup);
    } else {
        map.removeLayer(layerGroup);
    }
}

// ----------------------------------------------------
// PIPELINE CONTROL
// ----------------------------------------------------

async function runHydra() {
    const dataset = document.getElementById('dataset-select').value;
    const btn = document.getElementById('run-pipeline-btn');
    const statusText = document.getElementById('pipeline-status');

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> INITIALIZING...';
    statusText.classList.remove('hidden');
    statusText.innerText = `Dispatching workers for ${dataset}...`;

    try {
        const res = await fetch(`${API_BASE}/run?dataset=${dataset}`, {
            method: 'POST'
        });
        const data = await res.json();
        
        if(data.status === 'started' || data.status === 'running') {
            statusText.innerText = "Pipeline active. Collecting telemetry...";
            statusText.style.color = "var(--accent-green)";
            
            // Start WebSocket connection
            startWebSocket();
        }
    } catch (e) {
        console.error(e);
        statusText.innerText = "Error contacting cluster.";
        statusText.style.color = "var(--accent-red)";
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-bolt"></i> RUN PIPELINE';
    }
}

function startWebSocket() {
    if (pipelineSocket) pipelineSocket.close();
    
    const host = window.location.hostname || 'localhost';
    const wsUrl = `ws://${host}:8000/ws/status`;
    console.log("Connecting to WebSocket:", wsUrl);
    
    pipelineSocket = new WebSocket(wsUrl);

    pipelineSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("WS Data:", data);
        updateTelemetry(data);
    };

    pipelineSocket.onerror = (e) => {
        console.warn("WebSocket failed. Falling back to polling API:", e);
        startPolling(); // Fallback if WS fails
    };
}

let pollingInterval = null;
function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(async () => {
        try {
            const resp = await fetch('http://localhost:8000/status');
            const data = await resp.json();
            updateTelemetry(data);
            if (data.state === 'COMPLETED' || data.state === 'IDLE') {
                clearInterval(pollingInterval);
            }
        } catch (e) {
            console.error("Polling failed:", e);
        }
    }, 3000);
}

function updateTelemetry(data) {
    document.getElementById('metric-tiles-total').innerText = data.tiles_total || 0;
    document.getElementById('metric-tiles-done').innerText = data.tiles_done || 0;
    
    // Simulated objects parsing
    let objs = (data.tiles_done * 3.4).toFixed(0);
    document.getElementById('metric-objects').innerText = objs.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    
    let progress = (data.tiles_done / data.tiles_total) * 100;
    if(isNaN(progress)) progress = 0;
    
    document.getElementById('pipeline-progress').style.width = `${progress}%`;

    if (data.state === "COMPLETED" || data.state === "FAILED" || progress >= 100) {
        if(pipelineSocket) {
             pipelineSocket.close();
             pipelineSocket = null;
        }
        document.getElementById('pipeline-status').innerText = data.state === "FAILED" ? "PIPELINE FAILED" : "PIPELINE COMPLETE";
        document.getElementById('pipeline-status').style.color = data.state === "FAILED" ? "var(--accent-red)" : "var(--accent-green)";
        const btn = document.getElementById('run-pipeline-btn');
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-bolt"></i> RUN PIPELINE';

        // Power of PostGIS: Fetch real spatial results immediately
        if (data.run_id) {
            fetchGeoJSON(data.run_id);
        }
    }
}

async function fetchGeoJSON(runId) {
    try {
        const res = await fetch(`${API_BASE}/layers/geojson?run_id=${runId}`);
        const geojson = await res.json();
        
        polygonsLayer.clearLayers();
        L.geoJSON(geojson, {
            style: function(f) {
                const clsId = f.properties.class_id;
                return {
                    color: CLASSES[clsId]?.color || '#fff',
                    weight: 2,
                    fillOpacity: 0.5
                };
            },
            onEachFeature: function(f, layer) {
                layer.bindTooltip(`Class: ${CLASSES[f.properties.class_id]?.name}<br>Conf: ${Math.floor(f.properties.confidence*100)}%<br>Height: ${f.properties.height}m`);
                layer.on('click', () => {
                    // Inject properties into Inspector
                    document.getElementById('inspect-meta-id').innerText = f.properties.tile_id;
                    document.getElementById('inspect-fusion-conf').innerText = `${Math.floor(f.properties.confidence*100)}%`;
                    document.getElementById('inspect-depth').innerText = f.properties.height;
                    
                    const clsBadge = document.getElementById('inspect-final-class');
                    clsBadge.innerText = CLASSES[f.properties.class_id]?.name;
                    
                    document.getElementById('inspector-placeholder').classList.add('hidden');
                    document.getElementById('inspector-details').classList.remove('hidden');
                });
            }
        }).addTo(polygonsLayer);
        
        console.log(`Loaded ${geojson.features.length} polygons from PostGIS database.`);
    } catch(e) {
        console.error("Failed to fetch GeoJSON from PostGIS", e);
    }
}

// ----------------------------------------------------
// 3D VIEWER INTEGRATION
// ----------------------------------------------------
function toggle3DView() {
    is3DView = !is3DView;
    const mapEl = document.getElementById('map');
    
    if (is3DView) {
        // Initialize Deck.GL overlay
        if (!deckgl) {
            deckgl = new deck.DeckGL({
                container: mapEl,
                initialViewState: {
                    longitude: 78.4,
                    latitude: 17.3,
                    zoom: 15,
                    pitch: 45,
                    bearing: 0
                },
                controller: true,
                layers: [
                    new deck.GeoJsonLayer({
                        id: 'buildings-3d',
                        // Replace with dynamic API call for village height map, falling back to mock block structure for visualization
                        data: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/geojson/vancouver-blocks.json', 
                        extruded: true,
                        getFillColor: f => {
                            // Extract color mapping dynamically if passed from GeoSAM / Fusion
                            // E.g., if f.properties.class === 'building', return Red, etc.
                            // Standardizing on Building Red for now
                            return [248, 81, 73, 220];
                        },
                        getElevation: f => {
                            // Read precise Z-mean anomaly directly from DepthPro pipelines instead of math.random
                            return f.properties.height || f.properties.z_mean || (Math.random() * 10 + 2);
                        },
                        pickable: true,
                        onClick: info => {
                            if(info.object) {
                                console.log(`3D Object Clicked: Height ${info.object.properties.height || info.object.properties.z_mean}m`);
                            }
                        }
                    })
                ]
            });
        }
        
    } else {
        if (deckgl) {
            deckgl.finalize();
            deckgl = null;
        }
        map.invalidateSize(); // Resets leaflet visually
    }
}

// ----------------------------------------------------
// TILE INSPECTOR MOCK (Assuming API feeds tile info on click)
// ----------------------------------------------------

map.on('click', async (e) => {
    // In a real scenario we'd do: fetch(`/api/tile-info?lat=${e.latlng.lat}&lng=${e.latlng.lng}`)
    // For now we mock the inspector reveal:
    
    document.getElementById('inspector-placeholder').classList.add('hidden');
    document.getElementById('inspector-details').classList.remove('hidden');
    
    // Generating mock IDs based on coords
    let mockId = `T_${Math.abs(Math.floor(e.latlng.lat * 10000))}_${Math.abs(Math.floor(e.latlng.lng * 10000))}`;
    
    document.getElementById('inspect-meta-id').innerText = mockId;
    
    // Simulate data
    let swinConf = (0.85 + Math.random() * 0.1).toFixed(2);
    let yoloCount = Math.floor(Math.random() * 12);
    let fusConf = (swinConf * 0.6 + 0.4).toFixed(2);
    let depth = (Math.random() * 5).toFixed(1);
    
    document.getElementById('inspect-swin-conf').innerText = `${Math.floor(swinConf*100)}%`;
    document.getElementById('inspect-yolo-count').innerText = yoloCount;
    document.getElementById('inspect-fusion-conf').innerText = `${Math.floor(fusConf*100)}%`;
    document.getElementById('inspect-depth').innerText = depth;
    
    let isBuilding = Math.random() > 0.3;
    let clsBadge = document.getElementById('inspect-final-class');
    
    if(isBuilding) {
        clsBadge.innerText = "Building";
        clsBadge.style.color = "var(--color-building)";
        clsBadge.style.borderColor = "var(--color-building)";
        clsBadge.style.background = "rgba(248, 81, 73, 0.1)";
    } else {
        clsBadge.innerText = "Vegetation";
        clsBadge.style.color = "var(--color-veg)";
        clsBadge.style.borderColor = "var(--color-veg)";
        clsBadge.style.background = "rgba(63, 185, 80, 0.1)";
    }
    
    // Hide feedback form initially for each new click
    document.getElementById('feedback-form').classList.add('hidden');
});

// ----------------------------------------------------
// CONTINUOUS LEARNING FEEDBACK (HITL)
// ----------------------------------------------------
function toggleFeedbackForm() {
    document.getElementById('feedback-form').classList.toggle('hidden');
}

async function submitFeedback() {
    const tileId = document.getElementById('inspect-meta-id').innerText;
    const correctedClass = document.getElementById('feedback-class').value;
    const currentClassStr = document.getElementById('inspect-final-class').innerText;
    
    // Quick mock mapping for original class
    const clsMap = {"Background": 0, "Building": 1, "Road": 2, "Vegetation": 3, "Water": 4, "Other": 5};
    const originalClass = clsMap[currentClassStr] || 0;

    try {
        const response = await fetch(`${API_BASE}/feedback/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tile_id: tileId,
                original_class: originalClass,
                corrected_class: parseInt(correctedClass),
                notes: "Submitted via HITL Inspector UI"
            })
        });
        
        if(response.ok) {
            alert(`Thanks! Tile ${tileId} flagged for continuous learning queue.`);
            document.getElementById('feedback-form').classList.add('hidden');
        } else {
            console.error("Failed to submit feedback.");
        }
    } catch(err) {
        console.error("API Error sending feedback:", err);
    }
}

function exportData() {
    alert("Exporting latest pipeline run as OGC GeoPackage...");
}
