// Chart.js initialization for Pipeline Metrics

function renderCharts(data) {
    if (!data || !data.predictions || data.predictions.length === 0) return;

    // Process data for charts
    let classCounts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
    let confidences = [];
    let avgConf = 0;

    data.predictions.forEach(p => {
        if (p.final_class > 0) {
            classCounts[p.final_class] = (classCounts[p.final_class] || 0) + 1;
        }
        confidences.push(p.final_confidence * 100);
        avgConf += p.final_confidence;
    });

    avgConf = (avgConf / data.predictions.length * 100).toFixed(1);

    // Update summary metrics
    document.getElementById('metric-tiles').innerText = data.processed_tiles;
    document.getElementById('metric-conf').innerText = avgConf + '%';
    document.getElementById('metric-time').innerText = data.overall_time_sec ? data.overall_time_sec.toFixed(1) + 's' : 'N/A';

    // Chart 1: Class Distribution
    const ctxDist = document.getElementById('classDistChart').getContext('2d');

    let labels = [];
    let counts = [];
    let colors = [];

    for (let c in classCounts) {
        if (classCounts[c] > 0) {
            labels.push(CLASSES[c].name);
            counts.push(classCounts[c]);
            colors.push(CLASSES[c].color);
        }
    }

    if (window._classDistChartInst) window._classDistChartInst.destroy();
    window._classDistChartInst = new Chart(ctxDist, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Tiles detected',
                data: counts,
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: '#4A5056' } },
                x: { grid: { display: false } }
            }
        }
    });

    // Chart 2: Confidence Timeline
    const ctxConf = document.getElementById('confTimelineChart').getContext('2d');

    // Smooth the timeline by taking moving average or downsampling if too many points
    let downsampled = confidences;
    let timeLabels = Array.from({ length: downsampled.length }, (_, i) => i);

    if (window._confChartInst) window._confChartInst.destroy();
    window._confChartInst = new Chart(ctxConf, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [{
                label: 'Confidence (%)',
                data: downsampled,
                borderColor: '#4C6FAE',
                backgroundColor: 'transparent',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { max: 100, min: 0, grid: { color: '#4A5056' } },
                x: { display: false }
            }
        }
    });
}
