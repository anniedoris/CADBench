window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

const leaderboardSeriesSpecs = {
    mesh: {
        title: 'Mesh-to-CAD',
        pathSegment: 'mesh',
        models: [
            { key: 'cadfit', label: 'CADFit', color: '#08306b' },
            { key: 'cadrecode', label: 'CAD-Recode', color: '#9e8cc5' },
            { key: 'cadevolve', label: 'CADEvolve', color: '#74c6c6' },
            { key: 'cadrille_pc', label: 'Cadrille', color: '#6baed6' }
        ]
    },
    image: {
        title: 'Image-to-CAD',
        pathSegment: 'singleview',
        models: [
            { key: 'claude4.7', label: 'Claude Opus 4.7', color: '#005824' },
            { key: 'gemini3.1', label: 'Gemini 3.1 Pro', color: '#2ca25f' },
            { key: 'gpt5.4', label: 'GPT-5.4', color: '#8c510a' },
            { key: 'kimi_2.6', label: 'Kimi K2.6', color: '#c8d400' },
            { key: 'qwen3.59b', label: 'Qwen 3.59B', color: '#65a30d' },
            { key: 'qwen3.527b', label: 'Qwen 3.527B', color: '#f59e0b' },
            { key: 'cadcoder', label: 'CAD-Coder', color: '#e08010' }
        ]
    }
};

const leaderboardMetrics = {
    iou: {
        label: 'IoU',
        description: 'Average aligned IoU adjusted median across all benches.',
        sourceKey: 'average_aligned_iou_adjusted_median_across_all_benches',
        sourceNoteHtml: 'Loading IoU data from the mesh results summaries.',
        range: [0, 1],
        sortDirection: 'desc',
        formatter(value) {
            return value.toFixed(3);
        },
        entriesBySeries: {
            mesh: [],
            image: []
        }
    },
    siou: {
        label: 'SIoU',
        description: 'Average aligned surface IoU adjusted median across all benches.',
        sourceKey: 'average_aligned_surface_iou_adjusted_median_across_all_benches',
        sourceNoteHtml: 'Loading SIoU data from the mesh results summaries.',
        range: [0, 1],
        sortDirection: 'desc',
        formatter(value) {
            return value.toFixed(3);
        },
        entriesBySeries: {
            mesh: [],
            image: []
        }
    },
    vsr: {
        label: 'VSR',
        description: 'Average validity-success rate across all benches.',
        sourceKey: 'average_vsr_across_all_benches',
        sourceNoteHtml: 'Loading VSR data from the mesh results summaries.',
        range: [0, 1],
        sortDirection: 'desc',
        formatter(value) {
            return value.toFixed(3);
        },
        entriesBySeries: {
            mesh: [],
            image: []
        }
    },
    cd: {
        label: 'CD',
        description: 'Average aligned chamfer distance success-only median across all benches.',
        sourceKey: 'average_aligned_chamfer_distance_success_only_median_across_all_benches',
        sourceNoteHtml: 'Loading chamfer distance data from the mesh results summaries.',
        sortDirection: 'asc',
        formatter(value) {
            return value.toFixed(3);
        },
        entriesBySeries: {
            mesh: [],
            image: []
        }
    },
    token_count: {
        label: 'Token Count',
        description: 'Average token count success-only median across all benches.',
        sourceKey: 'average_token_count_success_only_median_across_all_benches',
        sourceNoteHtml: 'Loading token count data from the mesh results summaries.',
        sortDirection: 'asc',
        formatter(value) {
            return Math.round(value).toString();
        },
        entriesBySeries: {
            mesh: [],
            image: []
        }
    },
    op_count: {
        label: 'Op Count',
        description: 'Average total operation count success-only median across all benches.',
        sourceKey: 'average_total_operations_success_only_median_across_all_benches',
        sourceNoteHtml: 'Loading operation count data from the mesh results summaries.',
        sortDirection: 'asc',
        formatter(value) {
            return Math.round(value).toString();
        },
        entriesBySeries: {
            mesh: [],
            image: []
        }
    }
};

const leaderboardSummaryTextCache = new Map();
let leaderboardSnapshotPromise = null;
let liveBenchComplexitiesPromise = null;
const leaderboardAssetVersion = '20260601p';
const benchSlideData = {};

function escapeHtml(value) {
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function getSnapshotBenchComplexities(payload) {
    return payload && payload.complexity && payload.complexity.bench_complexities
        ? payload.complexity.bench_complexities
        : null;
}

async function loadBenchComplexities(snapshotPayload) {
    const snapshotBenchComplexities = getSnapshotBenchComplexities(snapshotPayload);

    if (!liveBenchComplexitiesPromise) {
        const cacheBuster = Date.now();
        const sourceCandidates = [
            'static/bench_complexities.json',
            'graph/bench_complexities.json',
            '../graph/bench_complexities.json'
        ];

        liveBenchComplexitiesPromise = (async () => {
            for (const sourceUrl of sourceCandidates) {
                try {
                    const response = await fetch(`${sourceUrl}?v=${cacheBuster}`, { cache: 'no-store' });
                    if (!response.ok) continue;

                    const payload = await response.json();
                    console.info(`Loaded benchmark complexities from ${sourceUrl}`);
                    return payload;
                } catch (error) {
                    // Try the next source shape; static hosting roots differ between local preview and deployment.
                }
            }

            console.warn('Using benchmark complexities from leaderboard-data.json snapshot.');
            return snapshotBenchComplexities;
        })();
    }

    return liveBenchComplexitiesPromise;
}

function parseLeaderboardMetric(summaryText, sourceKey) {
    const escapedSourceKey = sourceKey.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const matcher = new RegExp(`^${escapedSourceKey}:\\s*([0-9]*\\.?[0-9]+)\\s*$`, 'm');
    const match = summaryText.match(matcher);

    if (!match) {
        throw new Error(`Missing leaderboard metric: ${sourceKey}`);
    }

    return Number.parseFloat(match[1]);
}

async function loadLiveLeaderboardEntries(metricKey, seriesKey) {
    const metric = leaderboardMetrics[metricKey];
    const seriesSpec = leaderboardSeriesSpecs[seriesKey];

    return Promise.all(seriesSpec.models.map(async (model) => {
        const summaryCacheKey = `${model.key}:${seriesSpec.pathSegment}`;
        let summaryTextPromise = leaderboardSummaryTextCache.get(summaryCacheKey);

        if (!summaryTextPromise) {
            summaryTextPromise = fetch(`../tested_models/${model.key}/${seriesSpec.pathSegment}/results_summary.txt?v=${leaderboardAssetVersion}`, { cache: 'no-store' })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error(`Unable to load summary for ${model.key}`);
                    }

                    return response.text();
                });
            leaderboardSummaryTextCache.set(summaryCacheKey, summaryTextPromise);
        }

        const summaryText = await summaryTextPromise;

        return {
            name: model.label,
            value: parseLeaderboardMetric(summaryText, metric.sourceKey),
            color: model.color
        };
    }));
}

async function loadSnapshotLeaderboardEntries(metricKey, seriesKey) {
    if (!leaderboardSnapshotPromise) {
        leaderboardSnapshotPromise = fetch(`static/leaderboard-data.json?v=${leaderboardAssetVersion}`, { cache: 'no-store' })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Unable to load leaderboard snapshot');
                }

                return response.json();
            });
    }

    const payload = await leaderboardSnapshotPromise;
    const metricPayload = payload.metrics && payload.metrics[metricKey] ? payload.metrics[metricKey] : null;
    const seriesEntriesKey = `${seriesKey}_entries`;
    const snapshotEntries = metricPayload && Array.isArray(metricPayload[seriesEntriesKey])
        ? metricPayload[seriesEntriesKey]
        : [];

    if (snapshotEntries.length === 0) {
        throw new Error(`Leaderboard snapshot did not include ${seriesKey} entries for ${metricKey}`);
    }

    return snapshotEntries;
}

async function hydrateLeaderboardMetric(metricKey) {
    const metric = leaderboardMetrics[metricKey];

    try {
        const [meshEntries, imageEntries] = await Promise.all([
            loadLiveLeaderboardEntries(metricKey, 'mesh'),
            loadLiveLeaderboardEntries(metricKey, 'image')
        ]);
        metric.entriesBySeries.mesh = meshEntries;
        metric.entriesBySeries.image = imageEntries;
        metric.sourceNoteHtml = `Live from <code>tested_models/{model_name}/mesh/results_summary.txt</code> via <code>${metric.sourceKey}</code>.`;
        return;
    } catch (liveError) {
        console.warn(`Falling back to bundled leaderboard snapshot for ${metricKey}.`, liveError);
    }

    try {
        const [meshEntries, imageEntries] = await Promise.all([
            loadSnapshotLeaderboardEntries(metricKey, 'mesh'),
            loadSnapshotLeaderboardEntries(metricKey, 'image')
        ]);
        metric.entriesBySeries.mesh = meshEntries;
        metric.entriesBySeries.image = imageEntries;
        metric.sourceNoteHtml = `Loaded from <code>static/leaderboard-data.json</code> via <code>${metric.sourceKey}</code>.`;
    } catch (snapshotError) {
        console.error(`Unable to load leaderboard data for ${metricKey}.`, snapshotError);
        metric.entriesBySeries.mesh = [];
        metric.entriesBySeries.image = [];
        metric.sourceNoteHtml = `Unable to load ${metric.label} data from the model results summaries or the bundled snapshot.`;
    }
}

async function hydrateLeaderboardData() {
    await Promise.all(Object.keys(leaderboardMetrics).map((metricKey) => hydrateLeaderboardMetric(metricKey)));
}

async function hydrateBenchSlideData(slideIndex) {
    if (!leaderboardSnapshotPromise) {
        leaderboardSnapshotPromise = fetch(`static/leaderboard-data.json?v=${leaderboardAssetVersion}`, { cache: 'no-store' })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Unable to load leaderboard snapshot');
                }
                return response.json();
            });
    }

    const payload = await leaderboardSnapshotPromise;
    const slidePayload = payload.bench_slides && payload.bench_slides[String(slideIndex)];

    if (!slidePayload || !slidePayload.metrics) {
        throw new Error(`No bench slide data for index ${slideIndex}`);
    }

    const slideResult = {};
    for (const metricKey of Object.keys(leaderboardMetrics)) {
        const metricPayload = slidePayload.metrics[metricKey];
        slideResult[metricKey] = {
            mesh: (metricPayload && Array.isArray(metricPayload.mesh_entries)) ? metricPayload.mesh_entries : [],
            image: (metricPayload && Array.isArray(metricPayload.image_entries)) ? metricPayload.image_entries : []
        };
    }

    benchSlideData[slideIndex] = slideResult;
}

// ===== Pure chart utilities (shared by all leaderboard instances) =====

function getNiceTickStep(rawStep) {
    if (!Number.isFinite(rawStep) || rawStep <= 0) {
        return 1;
    }
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const residual = rawStep / magnitude;
    if (residual <= 1) return magnitude;
    if (residual <= 1.5) return 1.5 * magnitude;
    if (residual <= 2) return 2 * magnitude;
    if (residual <= 2.5) return 2.5 * magnitude;
    if (residual <= 5) return 5 * magnitude;
    return 10 * magnitude;
}

function getLeaderboardDomain(metric, entries) {
    if (metric.range) {
        return { minValue: metric.range[0], maxValue: metric.range[1] };
    }
    const tickCount = 5;
    const values = entries.map((e) => e.value).filter((v) => Number.isFinite(v));
    const dataMax = values.length > 0 ? Math.max(...values, 0) : 0;
    if (dataMax <= 0) return { minValue: 0, maxValue: 1 };
    const step = getNiceTickStep(dataMax / (tickCount - 1));
    return { minValue: 0, maxValue: step * (tickCount - 1) };
}

function renderLeaderboardAxisScale(axisScaleEl, metric, minValue, maxValue) {
    const tickCount = 5;
    const step = tickCount > 1 ? (maxValue - minValue) / (tickCount - 1) : 0;
    const ticks = [];
    for (let index = tickCount - 1; index >= 0; index -= 1) {
        const value = minValue + step * index;
        ticks.push(`<span>${metric.formatter ? metric.formatter(value) : value.toFixed(3)}</span>`);
    }
    axisScaleEl.innerHTML = ticks.join('');
}

function sortLeaderboardEntries(entries, metric) {
    return [...entries].sort((left, right) => {
        if (metric.sortDirection === 'asc' && left.value !== right.value) return left.value - right.value;
        if (metric.sortDirection !== 'asc' && right.value !== left.value) return right.value - left.value;
        return left.name.localeCompare(right.name);
    });
}

// ===== Shared chart + table renderers =====

function renderLeaderboardChart(elements, metric) {
    const { plot, axisTitle, axisScale, groupLabels, chart, xAxis, emptyState } = elements;

    function showPlot() {
        plot.hidden = false;
        plot.style.display = '';
        emptyState.hidden = true;
        emptyState.setAttribute('aria-hidden', 'true');
        emptyState.style.display = 'none';
        emptyState.textContent = '';
    }

    function showEmptyState(message) {
        plot.hidden = true;
        plot.style.display = 'none';
        groupLabels.innerHTML = '';
        chart.innerHTML = '';
        xAxis.innerHTML = '';
        emptyState.hidden = false;
        emptyState.setAttribute('aria-hidden', 'false');
        emptyState.style.display = 'flex';
        emptyState.textContent = message;
    }

    const meshEntries = metric.entriesBySeries && Array.isArray(metric.entriesBySeries.mesh)
        ? sortLeaderboardEntries(metric.entriesBySeries.mesh, metric)
        : [];
    const imageEntries = metric.entriesBySeries && Array.isArray(metric.entriesBySeries.image)
        ? sortLeaderboardEntries(metric.entriesBySeries.image, metric)
        : [];
    const allEntries = [...meshEntries, ...imageEntries];

    axisTitle.textContent = metric.label;

    if (allEntries.length === 0) {
        showEmptyState(`No bars available for ${metric.label} yet.`);
        return;
    }

    const { minValue, maxValue } = getLeaderboardDomain(metric, allEntries);
    const templateParts = [];
    const groupLabelParts = [];
    const chartParts = [];
    const xAxisParts = [];
    let columnIndex = 1;

    function appendSeries(seriesKey, entries) {
        const seriesSpec = leaderboardSeriesSpecs[seriesKey];
        if (entries.length === 0) return;

        if (templateParts.length > 0) {
            templateParts.push('32px');
            groupLabelParts.push(`<div class="leaderboard-divider" style="grid-column: ${columnIndex};"></div>`);
            chartParts.push('<div class="leaderboard-divider"></div>');
            xAxisParts.push('<div class="leaderboard-divider"></div>');
            columnIndex += 1;
        }

        templateParts.push(`repeat(${entries.length}, minmax(0, 1fr))`);
        groupLabelParts.push(
            `<div class="leaderboard-group-label" style="grid-column: ${columnIndex} / span ${entries.length};">${escapeHtml(seriesSpec.title)}</div>`
        );

        entries.forEach((entry) => {
            const height = maxValue === minValue
                ? 100
                : Math.max(0, Math.min(100, ((entry.value - minValue) / (maxValue - minValue)) * 100));

            chartParts.push(`
                <div class="leaderboard-bar-group">
                    <div class="leaderboard-bar-shell" style="--bar-height: ${height}%; --bar-color: ${entry.color};">
                        <span class="leaderboard-model-value">${metric.formatter(entry.value)}</span>
                        <div class="leaderboard-bar"></div>
                    </div>
                </div>
            `);
            xAxisParts.push(`<span class="leaderboard-model-name">${escapeHtml(entry.name)}</span>`);
            columnIndex += 1;
        });
    }

    appendSeries('mesh', meshEntries);
    appendSeries('image', imageEntries);

    const gridTemplateColumns = templateParts.join(' ');
    showPlot();
    renderLeaderboardAxisScale(axisScale, metric, minValue, maxValue);
    groupLabels.style.gridTemplateColumns = gridTemplateColumns;
    chart.style.gridTemplateColumns = gridTemplateColumns;
    xAxis.style.gridTemplateColumns = gridTemplateColumns;
    groupLabels.innerHTML = groupLabelParts.join('');
    chart.innerHTML = chartParts.join('');
    xAxis.innerHTML = xAxisParts.join('');
}

function renderLeaderboardTable(tableWrapper, sortMetricKey, entriesOverride) {
    if (!tableWrapper) return;

    const metricKeys = ['iou', 'siou', 'cd', 'vsr', 'token_count', 'op_count'];
    const date = 'May 11, 2026';

    const rows = [];
    Object.entries(leaderboardSeriesSpecs).forEach(([seriesKey, seriesSpec]) => {
        seriesSpec.models.forEach((model) => {
            const values = {};
            metricKeys.forEach((metricKey) => {
                const seriesData = entriesOverride
                    ? (entriesOverride[metricKey] || {})
                    : leaderboardMetrics[metricKey].entriesBySeries;
                const entries = seriesData[seriesKey];
                const entry = entries && entries.find((e) => e.name === model.label);
                values[metricKey] = (entry && Number.isFinite(entry.value)) ? entry.value : null;
            });
            rows.push({ name: model.label, color: model.color, series: seriesKey, values });
        });
    });

    const sortAsc = leaderboardMetrics[sortMetricKey] && leaderboardMetrics[sortMetricKey].sortDirection === 'asc';
    rows.sort((a, b) => {
        const aVal = a.values[sortMetricKey];
        const bVal = b.values[sortMetricKey];
        const aSort = aVal !== null ? aVal : (sortAsc ? Infinity : -Infinity);
        const bSort = bVal !== null ? bVal : (sortAsc ? Infinity : -Infinity);
        if (aSort !== bSort) return sortAsc ? aSort - bSort : bSort - aSort;
        return a.name.localeCompare(b.name);
    });

    function formatCell(metricKey, value) {
        if (value === null) return '&mdash;';
        return escapeHtml(leaderboardMetrics[metricKey].formatter(value));
    }

    function ac(metricKey) {
        return metricKey === sortMetricKey ? ' leaderboard-table-active-col' : '';
    }

    const headerDefs = [
        { label: 'Rank', mk: null },
        { label: 'Model', mk: null },
        { label: 'Date', mk: null },
        { label: 'IoU', mk: 'iou' },
        { label: 'SIoU', mk: 'siou' },
        { label: 'CD', mk: 'cd' },
        { label: 'VSR', mk: 'vsr' },
        { label: 'Token Count', mk: 'token_count' },
        { label: 'Op Count', mk: 'op_count' }
    ];
    const thead = `<thead><tr>${headerDefs.map(({ label, mk }) => {
        const arrow = mk === sortMetricKey
            ? (leaderboardMetrics[mk].sortDirection === 'asc' ? ' ↓' : ' ↑')
            : '';
        return `<th class="${ac(mk)}">${escapeHtml(label)}${arrow}</th>`;
    }).join('')}</tr></thead>`;

    const tbodyRows = rows.map((row, i) => {
        const rowClass = row.series === 'mesh' ? 'leaderboard-table-row-mesh' : 'leaderboard-table-row-image';
        const swatch = `<span class="leaderboard-table-swatch" style="background:${row.color};"></span>`;
        const cells = [
            `<td class="leaderboard-table-rank">${i + 1}</td>`,
            `<td class="leaderboard-table-model">${swatch}${escapeHtml(row.name)}</td>`,
            `<td class="leaderboard-table-date">${date}</td>`,
            `<td class="${ac('iou')}">${formatCell('iou', row.values.iou)}</td>`,
            `<td class="${ac('siou')}">${formatCell('siou', row.values.siou)}</td>`,
            `<td class="${ac('cd')}">${formatCell('cd', row.values.cd)}</td>`,
            `<td class="${ac('vsr')}">${formatCell('vsr', row.values.vsr)}</td>`,
            `<td class="${ac('token_count')}">${formatCell('token_count', row.values.token_count)}</td>`,
            `<td class="${ac('op_count')}">${formatCell('op_count', row.values.op_count)}</td>`
        ];
        return `<tr class="${rowClass}">${cells.join('')}</tr>`;
    });

    const legend = `
        <div class="leaderboard-table-legend">
            <span class="leaderboard-table-legend-item">
                <span class="leaderboard-table-legend-swatch leaderboard-table-legend-swatch--mesh"></span>
                Mesh-to-CAD
            </span>
            <span class="leaderboard-table-legend-item">
                <span class="leaderboard-table-legend-swatch leaderboard-table-legend-swatch--image"></span>
                Image-to-CAD
            </span>
        </div>`;
    tableWrapper.innerHTML = `${legend}<div class="leaderboard-table-wrapper"><table class="leaderboard-table" aria-label="Model scores">${thead}<tbody>${tbodyRows.join('')}</tbody></table></div>`;
}

// ===== Overall leaderboard =====

function setupLeaderboard() {
    const metricButtons = document.querySelectorAll('.overall-metric-btn');
    const elements = {
        plot: document.querySelector('[data-role="plot"]'),
        axisTitle: document.querySelector('[data-role="axis-title"]'),
        axisScale: document.querySelector('[data-role="axis-scale"]'),
        groupLabels: document.querySelector('[data-role="group-labels"]'),
        chart: document.querySelector('[data-role="chart"]'),
        xAxis: document.querySelector('[data-role="x-axis"]'),
        emptyState: document.querySelector('[data-role="empty-state"]'),
        tableWrapper: document.querySelector('[data-role="leaderboard-table"]')
    };

    if (
        metricButtons.length === 0 ||
        !elements.plot ||
        !elements.axisTitle ||
        !elements.axisScale ||
        !elements.groupLabels ||
        !elements.chart ||
        !elements.xAxis ||
        !elements.emptyState
    ) {
        return;
    }

    function setActiveMetric(metricKey) {
        const metric = leaderboardMetrics[metricKey];
        if (!metric) return;

        metricButtons.forEach((button) => {
            const isActive = button.dataset.metric === metricKey;
            button.classList.toggle('is-active', isActive);
            button.setAttribute('aria-pressed', String(isActive));
        });

        renderLeaderboardChart(elements, metric);
        renderLeaderboardTable(elements.tableWrapper, metricKey);
    }

    metricButtons.forEach((button) => {
        button.addEventListener('click', function() {
            setActiveMetric(button.dataset.metric);
        });
    });

    setActiveMetric('iou');
}

function renderComplexityLineChart(container, seriesList, splitMarkers) {
    const width = 920;
    const height = 340;
    const margin = { top: 84, right: 280, bottom: 66, left: 60 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const legendX = margin.left + plotWidth + 74;
    const legendY = margin.top - 32;
    const activeSeriesKeys = new Set(seriesList.map((series) => series.key));

    const xValues = seriesList.flatMap((series) => series.points.map((point) => point.x));
    const positiveXValues = xValues.filter((value) => value > 0);
    const minPositiveX = Math.min(...positiveXValues);
    const maxX = Math.max(...positiveXValues);
    const logMinX = Math.log(minPositiveX);
    const logMaxX = Math.log(maxX);
    const firstPointOffset = plotWidth * 0.06;

    function xScale(value) {
        if (value <= 0 || positiveXValues.length === 0) return margin.left;
        if (maxX === minPositiveX) return margin.left + firstPointOffset;
        return margin.left + firstPointOffset + ((Math.log(value) - logMinX) / (logMaxX - logMinX)) * (plotWidth - firstPointOffset);
    }

    function yScale(value) {
        return margin.top + (1 - value) * plotHeight;
    }

    function trianglePoints(x, y, size) {
        const top = y - size;
        const bottom = y + size * 0.75;
        const left = x - size;
        const right = x + size;
        return `${x},${top} ${right},${bottom} ${left},${bottom}`;
    }

    function renderSplitLabel(label, x) {
        const labelText = String(label);
        const hyphenIndex = labelText.indexOf('-');
        const lines = hyphenIndex >= 0
            ? [labelText.slice(0, hyphenIndex + 1), labelText.slice(hyphenIndex + 1)]
            : [labelText];
        const twoLineOffset = labelText === 'Extrude-Low' ? 56 : 28;
        const startY = lines.length > 1 ? margin.top - twoLineOffset : margin.top - 14;
        const tspans = lines.map((line, index) => (
            `<tspan x="${x.toFixed(1)}" dy="${index === 0 ? 0 : 13}">${escapeHtml(line)}</tspan>`
        )).join('');

        return `<text x="${x.toFixed(1)}" y="${startY}" text-anchor="middle" font-size="0.74rem" font-weight="800" fill="var(--text-light)">${tspans}</text>`;
    }

    const yTicks = [0, 0.25, 0.5, 0.75, 1];
    const yGrid = yTicks.map((value) => {
        const y = yScale(value).toFixed(1);
        return `<line x1="${margin.left}" y1="${y}" x2="${margin.left + plotWidth}" y2="${y}" stroke="#ddd" stroke-width="0.8" stroke-dasharray="3,3"/>`;
    }).join('');

    const splitStrips = splitMarkers.map((marker) => {
        const x = xScale(marker.x);
        const stripTop = marker.label === 'Extrude-Low' ? margin.top - 34 : margin.top;
        const stripHeight = plotHeight + (margin.top - stripTop);
        return `
            <rect x="${(x - 3).toFixed(1)}" y="${stripTop}" width="6" height="${stripHeight}" fill="rgba(148, 163, 184, 0.16)"/>
            ${renderSplitLabel(marker.label, x)}
        `;
    }).join('');

    const xTicks = [0, ...splitMarkers.map((marker) => marker.x)].map((value) => {
        const x = xScale(value).toFixed(1);
        return `
            <line x1="${x}" y1="${margin.top + plotHeight}" x2="${x}" y2="${margin.top + plotHeight + 6}" stroke="var(--text-light)" stroke-width="1"/>
            <text x="${x}" y="${margin.top + plotHeight + 20}" text-anchor="middle" font-size="0.82rem" font-weight="600" fill="var(--text-light)">${value}</text>
        `;
    }).join('');

    const yTickLabels = yTicks.map((value) => {
        const y = (yScale(value) + 4).toFixed(1);
        return `<text x="${margin.left - 8}" y="${y}" text-anchor="end" font-size="0.82rem" font-weight="600" fill="var(--text-light)">${value.toFixed(2)}</text>`;
    }).join('');

    function makeMarker(series, x, y, size) {
        return series.markerShape === 'circle'
            ? `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${size}" fill="${series.color}" stroke="#fff" stroke-width="1.2"/>`
            : `<polygon points="${trianglePoints(x, y, size)}" fill="${series.color}" stroke="#fff" stroke-width="1.2"/>`;
    }

    function renderChart() {
        const activeSeriesList = seriesList.filter((series) => activeSeriesKeys.has(series.key));
        let previousSeriesType = null;
        let legendRowIndex = 0;
        const legendItems = seriesList.map((series) => {
            const header = series.seriesType !== previousSeriesType
                ? (() => {
                    const y = legendY + legendRowIndex * 22;
                    legendRowIndex += 1;
                    previousSeriesType = series.seriesType;
                    return `<text x="${legendX}" y="${y + 4}" font-size="0.78rem" font-weight="800" fill="var(--text-light)">${series.seriesType === 'mesh' ? 'Mesh-to-CAD' : 'Image-to-CAD'}</text>`;
                })()
                : '';
            const y = legendY + legendRowIndex * 22;
            legendRowIndex += 1;
            const isActive = activeSeriesKeys.has(series.key);
            const opacity = isActive ? '1' : '0.42';
            const checkboxX = legendX;
            const checkboxY = y - 7;
            const markerX = legendX + 34;
            const marker = makeMarker(series, markerX + 14, y, series.markerShape === 'circle' ? 4.4 : 4.8);
            const check = isActive
                ? `<path d="M${checkboxX + 3.1} ${checkboxY + 6.5} L${checkboxX + 5.4} ${checkboxY + 9.2} L${checkboxX + 10.1} ${checkboxY + 3.3}" fill="none" stroke="var(--text-primary)" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>`
                : '';

            return `
                ${header}
                <g data-complexity-series-key="${escapeHtml(series.key)}" role="checkbox" aria-checked="${isActive ? 'true' : 'false'}" tabindex="0" style="cursor:pointer;">
                    <title>${isActive ? 'Hide' : 'Show'} ${escapeHtml(series.label)}</title>
                    <rect x="${checkboxX}" y="${checkboxY}" width="12" height="12" rx="2" fill="#fff" stroke="var(--border-color)" stroke-width="1.4"/>
                    ${check}
                    <g opacity="${opacity}">
                        <line x1="${markerX}" y1="${y}" x2="${markerX + 28}" y2="${y}" stroke="${series.color}" stroke-width="2.6" stroke-linecap="round"/>
                        ${marker}
                        <text x="${markerX + 42}" y="${y + 4}" font-size="0.82rem" font-weight="700" fill="var(--text-primary)">${escapeHtml(series.label)}</text>
                    </g>
                </g>
            `;
        }).join('');

        const annoGray = '#9ca3af';
        const annoMarkerCx = legendX + 6;
        const annoY1 = legendY + legendRowIndex * 22 + 8;
        legendRowIndex += 1;
        const annoY2 = legendY + legendRowIndex * 22 + 8;
        const legendAnnotation = `
            <polygon points="${trianglePoints(annoMarkerCx, annoY1, 4.8)}" fill="${annoGray}" stroke="#fff" stroke-width="1.2"/>
            <text x="${annoMarkerCx + 14}" y="${annoY1 + 4}" font-size="0.82rem" font-weight="700" fill="var(--text-primary)">CAD-Specialized</text>
            <circle cx="${annoMarkerCx.toFixed(1)}" cy="${annoY2.toFixed(1)}" r="4.4" fill="${annoGray}" stroke="#fff" stroke-width="1.2"/>
            <text x="${annoMarkerCx + 14}" y="${annoY2 + 4}" font-size="0.82rem" font-weight="700" fill="var(--text-primary)">General Purpose</text>
        `;

        const seriesMarks = activeSeriesList.map((series) => {
            const linePath = series.points
                .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xScale(point.x).toFixed(1)} ${yScale(point.y).toFixed(1)}`)
                .join(' ');
            const dots = series.points.map((point) => {
                const x = xScale(point.x);
                const y = yScale(point.y);
                const title = `${escapeHtml(series.label)}: ${escapeHtml(point.split)}, ${point.x} faces, IoU ${point.y.toFixed(3)}`;
                const marker = makeMarker(series, x, y, series.markerShape === 'circle' ? 4.4 : 4.8);
                return `
                    <g>
                        ${marker}
                        <title>${title}</title>
                    </g>
                `;
            }).join('');

            return `
                <path d="${linePath}" fill="none" stroke="${series.color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>
                ${dots}
            `;
        }).join('');

        container.innerHTML = `
            <svg viewBox="0 0 ${width} ${height}" style="width:100%;display:block;" role="img" aria-label="Model IoU by extrude split complexity">
                <g aria-label="Legend">
                    ${legendItems}
                    ${legendAnnotation}
                </g>
                ${splitStrips}
                ${yGrid}
                <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="var(--text-light)" stroke-width="1.5"/>
                <line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="var(--text-light)" stroke-width="1.5"/>
                ${xTicks}
                ${yTickLabels}
                ${seriesMarks}
                <text x="${(margin.left + plotWidth / 2).toFixed(1)}" y="${height - 16}" text-anchor="middle" font-size="0.82rem" font-weight="700" letter-spacing="0.08em" fill="var(--text-light)" style="text-transform:uppercase;">Split Median Face Count (Log Scale)</text>
                <text transform="rotate(-90)" x="${(-(margin.top + plotHeight / 2)).toFixed(1)}" y="16" text-anchor="middle" font-size="0.82rem" font-weight="700" letter-spacing="0.08em" fill="var(--text-light)" style="text-transform:uppercase;">IOU</text>
            </svg>
        `;

        container.querySelectorAll('[data-complexity-series-key]').forEach((legendItem) => {
            function toggleSeries() {
                const seriesKey = legendItem.getAttribute('data-complexity-series-key');
                if (activeSeriesKeys.has(seriesKey)) {
                    activeSeriesKeys.delete(seriesKey);
                } else {
                    activeSeriesKeys.add(seriesKey);
                }
                renderChart();
            }

            legendItem.addEventListener('click', toggleSeries);
            legendItem.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    toggleSeries();
                }
            });
        });
    }

    renderChart();
}

async function setupComplexityLineChart() {
    const root = document.getElementById('complexity-root');
    if (!root) return;

    root.innerHTML = `
        <div class="leaderboard-chart-card">
            <p class="leaderboard-source-note">Loading complexity trend...</p>
        </div>
    `;

    try {
        if (!leaderboardSnapshotPromise) {
            leaderboardSnapshotPromise = fetch(`static/leaderboard-data.json?v=${leaderboardAssetVersion}`, { cache: 'no-store' })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error('Unable to load leaderboard snapshot');
                    }
                    return response.json();
                });
        }

        const payload = await leaderboardSnapshotPromise;
        const benchComplexities = await loadBenchComplexities(payload);
        const extrudeComplexities = benchComplexities ? benchComplexities.extrude : null;
        const extrudeLabels = benchComplexities ? benchComplexities.extrude_labels || {} : {};
        const meshComplexityIou = payload && payload.complexity && payload.complexity.iou && payload.complexity.iou.mesh
            ? payload.complexity.iou.mesh
            : null;
        const imageComplexityIou = payload && payload.complexity && payload.complexity.iou && payload.complexity.iou.image
            ? payload.complexity.iou.image
            : null;

        if (!extrudeComplexities || !meshComplexityIou || !imageComplexityIou) {
            throw new Error('Complexity data missing from leaderboard snapshot');
        }

        const complexityRows = Object.entries(extrudeComplexities)
            .map(([split, complexityValue]) => ({
                split,
                label: extrudeLabels[split] || split,
                x: Number(complexityValue)
            }))
            .filter((point) => Number.isFinite(point.x))
            .sort((left, right) => left.x - right.x);
        const selectedImageModelKeys = ['gemini3.1', 'gpt5.4', 'kimi_2.6', 'cadcoder', 'claude4.7'];
        const selectedImageModels = selectedImageModelKeys
            .map((modelKey) => leaderboardSeriesSpecs.image.models.find((model) => model.key === modelKey))
            .filter(Boolean);
        const complexitySeriesSpecs = [
            ...leaderboardSeriesSpecs.mesh.models.map((model) => ({ ...model, seriesType: 'mesh', markerShape: 'triangle' })),
            ...selectedImageModels.map((model) => ({ ...model, seriesType: 'image', markerShape: model.key === 'cadcoder' ? 'triangle' : 'circle' }))
        ];
        const seriesList = complexitySeriesSpecs
            .map((model) => {
                const modelValues = model.seriesType === 'mesh'
                    ? meshComplexityIou[model.key]
                    : imageComplexityIou[model.key];
                const points = complexityRows
                    .map((point) => ({
                        split: point.split,
                        x: point.x,
                        y: Number(modelValues && modelValues[point.split])
                    }))
                    .filter((point) => Number.isFinite(point.y));

                return {
                    key: model.key,
                    label: model.label,
                    color: model.color,
                    seriesType: model.seriesType,
                    markerShape: model.markerShape,
                    points
                };
            })
            .filter((series) => series.points.length > 0);
        if (seriesList.length === 0 || complexityRows.length === 0) {
            throw new Error('No valid complexity points to render');
        }

        root.innerHTML = `
            <div class="leaderboard-chart-card">
                <div id="complexity-line-chart"></div>
            </div>
        `;

        const chartContainer = root.querySelector('#complexity-line-chart');
        renderComplexityLineChart(chartContainer, seriesList, complexityRows);
    } catch (error) {
        root.innerHTML = `
            <div class="leaderboard-empty-state" style="display:flex;" aria-live="polite">
                Unable to load complexity chart.
            </div>
        `;
        console.error('Complexity line chart error:', error);
    }
}

// ===== Benchmark family leaderboard carousel =====

function setupBenchLeaderboard() {
    const benchContainer = document.querySelector('[data-role="bench-carousel"]');
    if (!benchContainer) return;

    const BENCH_FAMILIES = [
        'CAD-Base',
        'CAD-Fusion',
        'CAD-Extrude',
        'CAD-All-Ops',
        'CAD-Mechanical',
        'CAD-Organic'
    ];

    const BENCH_SLIDE_IMAGES = {
        0: 'static/images/bench0_grid.png',
        1: 'static/images/bench0F_grid.png',
        2: 'static/images/bench1A_grid.png',
        3: 'static/images/bench1B_grid.png',
        4: 'static/images/bench2_grid.png',
        5: 'static/images/bench3_grid.png'
    };

    const metricBtnDefs = [
        { key: 'iou', label: 'IoU &uarr;', active: true },
        { key: 'siou', label: 'SIoU &uarr;' },
        { key: 'vsr', label: 'VSR &uarr;' },
        { key: 'cd', label: 'CD &darr;' },
        { key: 'token_count', label: 'Token Count' },
        { key: 'op_count', label: 'Op Count' }
    ];

    function makeSlideHTML(index) {
        return `
            <div class="bench-slide" data-bench-slide="${index}"${index !== 0 ? ' hidden' : ''}>
                <div class="leaderboard-chart-card">
                    <div class="leaderboard-plot" data-slide-role="plot">
                        <div class="leaderboard-y-axis">
                            <p class="leaderboard-axis-title" data-slide-role="axis-title">IoU</p>
                            <div class="leaderboard-axis-scale" data-slide-role="axis-scale" aria-hidden="true"></div>
                        </div>
                        <div class="leaderboard-plot-main">
                            <div class="leaderboard-group-labels" data-slide-role="group-labels" aria-hidden="true"></div>
                            <div class="leaderboard-chart" data-slide-role="chart" aria-live="polite"></div>
                            <div class="leaderboard-x-axis" data-slide-role="x-axis" aria-hidden="true"></div>
                            <p class="leaderboard-x-axis-title">Models</p>
                        </div>
                    </div>
                    <div class="leaderboard-empty-state" data-slide-role="empty-state" hidden aria-hidden="true" style="display: none;"></div>
                </div>
                <div class="leaderboard-table-card">
                    <div data-slide-role="table"></div>
                </div>
            </div>
        `;
    }

    benchContainer.innerHTML = `
        <div class="bench-carousel-nav">
            <button class="bench-carousel-btn bench-prev-btn" aria-label="Previous benchmark family">&#8592; Prev</button>
            <span class="bench-carousel-indicator">
                <span class="bench-family-name">${escapeHtml(BENCH_FAMILIES[0])}</span>
                <span class="bench-carousel-counter">1 / ${BENCH_FAMILIES.length}</span>
            </span>
            <button class="bench-carousel-btn bench-next-btn" aria-label="Next benchmark family">Next &#8594;</button>
        </div>
        <div class="bench-slide-image-container">
            <img class="bench-slide-image" src="${BENCH_SLIDE_IMAGES[0] || ''}" alt="${escapeHtml(BENCH_FAMILIES[0])} benchmark overview" ${BENCH_SLIDE_IMAGES[0] ? '' : 'hidden'} />
        </div>
        <div class="leaderboard-controls" role="toolbar" aria-label="Benchmark family leaderboard metric selector">
            ${metricBtnDefs.map(({ key, label, active }) =>
                `<button class="leaderboard-metric-btn bench-metric-btn${active ? ' is-active' : ''}" type="button" data-metric="${key}" aria-pressed="${active ? 'true' : 'false'}">${label}</button>`
            ).join('')}
        </div>
        <div class="bench-carousel-slides">
            ${BENCH_FAMILIES.map((_, i) => makeSlideHTML(i)).join('')}
        </div>
    `;

    const slideEls = Array.from(benchContainer.querySelectorAll('.bench-slide'));
    const benchMetricButtons = benchContainer.querySelectorAll('.bench-metric-btn');
    const familyNameEl = benchContainer.querySelector('.bench-family-name');
    const counterEl = benchContainer.querySelector('.bench-carousel-counter');
    const prevBtn = benchContainer.querySelector('.bench-prev-btn');
    const nextBtn = benchContainer.querySelector('.bench-next-btn');
    const slideImageEl = benchContainer.querySelector('.bench-slide-image');
    let currentSlide = 0;

    function getSlideElements(slideEl) {
        return {
            plot: slideEl.querySelector('[data-slide-role="plot"]'),
            axisTitle: slideEl.querySelector('[data-slide-role="axis-title"]'),
            axisScale: slideEl.querySelector('[data-slide-role="axis-scale"]'),
            groupLabels: slideEl.querySelector('[data-slide-role="group-labels"]'),
            chart: slideEl.querySelector('[data-slide-role="chart"]'),
            xAxis: slideEl.querySelector('[data-slide-role="x-axis"]'),
            emptyState: slideEl.querySelector('[data-slide-role="empty-state"]'),
            tableWrapper: slideEl.querySelector('[data-slide-role="table"]')
        };
    }

    function goToSlide(index) {
        slideEls.forEach((el, i) => { el.hidden = i !== index; });
        familyNameEl.textContent = BENCH_FAMILIES[index];
        counterEl.textContent = `${index + 1} / ${BENCH_FAMILIES.length}`;
        currentSlide = index;
        const imgSrc = BENCH_SLIDE_IMAGES[index];
        if (imgSrc) {
            slideImageEl.src = imgSrc;
            slideImageEl.alt = BENCH_FAMILIES[index] + ' benchmark overview';
            slideImageEl.hidden = false;
        } else {
            slideImageEl.hidden = true;
        }
    }

    prevBtn.addEventListener('click', function() {
        goToSlide((currentSlide - 1 + BENCH_FAMILIES.length) % BENCH_FAMILIES.length);
    });

    nextBtn.addEventListener('click', function() {
        goToSlide((currentSlide + 1) % BENCH_FAMILIES.length);
    });

    function setActiveBenchMetric(metricKey) {
        const metric = leaderboardMetrics[metricKey];
        if (!metric) return;

        benchMetricButtons.forEach((btn) => {
            const isActive = btn.dataset.metric === metricKey;
            btn.classList.toggle('is-active', isActive);
            btn.setAttribute('aria-pressed', String(isActive));
        });

        slideEls.forEach((slideEl, slideIndex) => {
            const els = getSlideElements(slideEl);
            const slideData = benchSlideData[slideIndex];
            if (slideData && slideData[metricKey]) {
                const benchMetric = Object.assign({}, metric, { entriesBySeries: slideData[metricKey] });
                renderLeaderboardChart(els, benchMetric);
                renderLeaderboardTable(els.tableWrapper, metricKey, slideData);
            } else {
                renderLeaderboardChart(els, metric);
                renderLeaderboardTable(els.tableWrapper, metricKey);
            }
        });
    }

    benchMetricButtons.forEach((btn) => {
        btn.addEventListener('click', function() {
            setActiveBenchMetric(btn.dataset.metric);
        });
    });

    setActiveBenchMetric('iou');
}

function renderModalityMeshChart(elements, meshEntries, metric) {
    const { plot, axisTitle, axisScale, groupLabels, chart, xAxis, emptyState } = elements;

    function showPlot() {
        plot.hidden = false;
        plot.style.display = '';
        emptyState.hidden = true;
        emptyState.setAttribute('aria-hidden', 'true');
        emptyState.style.display = 'none';
        emptyState.textContent = '';
    }

    function showEmptyState(message) {
        plot.hidden = true;
        plot.style.display = 'none';
        groupLabels.innerHTML = '';
        chart.innerHTML = '';
        xAxis.innerHTML = '';
        emptyState.hidden = false;
        emptyState.setAttribute('aria-hidden', 'false');
        emptyState.style.display = 'flex';
        emptyState.textContent = message;
    }

    const sorted = sortLeaderboardEntries(meshEntries, metric);
    if (sorted.length === 0) {
        showEmptyState('No mesh data available.');
        return;
    }

    const N = sorted.length;
    axisTitle.textContent = metric.label;

    const { minValue, maxValue } = getLeaderboardDomain(metric, sorted);
    showPlot();
    renderLeaderboardAxisScale(axisScale, metric, minValue, maxValue);

    // Grid: pairs of bars [clean, noisy] per model with small gaps between models
    const gridParts = [];
    for (let i = 0; i < N; i++) {
        if (i > 0) gridParts.push('8px');
        gridParts.push('1fr 1fr');
    }
    const gridTemplate = gridParts.join(' ');
    groupLabels.style.gridTemplateColumns = gridTemplate;
    chart.style.gridTemplateColumns = gridTemplate;
    xAxis.style.gridTemplateColumns = gridTemplate;

    const totalCols = 3 * N - 1;
    groupLabels.innerHTML = `<div class="leaderboard-group-label" style="grid-column: 1 / span ${totalCols};">Mesh-to-CAD</div>`;

    const chartParts = [];
    const xAxisParts = [];

    sorted.forEach((entry, i) => {
        const cleanHeight = maxValue === minValue
            ? 100
            : Math.max(0, Math.min(100, ((entry.value - minValue) / (maxValue - minValue)) * 100));
        const noisyVal = entry.noisy_value;
        const noisyHeight = (noisyVal != null && Number.isFinite(noisyVal))
            ? (maxValue === minValue ? 100 : Math.max(0, Math.min(100, ((noisyVal - minValue) / (maxValue - minValue)) * 100)))
            : 0;

        if (i > 0) {
            chartParts.push('<div class="leaderboard-divider"></div>');
            xAxisParts.push('<div class="leaderboard-divider"></div>');
        }

        chartParts.push(`
            <div class="leaderboard-bar-group">
                <div class="leaderboard-bar-shell" style="--bar-height: ${cleanHeight}%; --bar-color: ${entry.color};">
                    <span class="leaderboard-model-value">${metric.formatter(entry.value)}</span>
                    <div class="leaderboard-bar"></div>
                </div>
            </div>
            <div class="leaderboard-bar-group">
                <div class="leaderboard-bar-shell" style="--bar-height: ${noisyHeight}%; --bar-color: ${entry.color};">
                    ${(() => {
                        const d = entry.noisy_delta;
                        if (d == null || !Number.isFinite(d)) return '';
                        const label = d >= 0 ? `+${metric.formatter(d)}` : metric.formatter(d);
                        const cls = d >= 0 ? 'leaderboard-model-delta--positive' : 'leaderboard-model-delta--negative';
                        return `<span class="leaderboard-model-delta ${cls}">${escapeHtml(label)}</span>`;
                    })()}
                    <span class="leaderboard-model-value">${noisyVal != null ? metric.formatter(noisyVal) : ''}</span>
                    <div class="leaderboard-bar leaderboard-bar--noisy"></div>
                </div>
            </div>
        `);

        xAxisParts.push(`<span class="leaderboard-model-name" style="grid-column: span 2;">${escapeHtml(entry.name)}</span>`);
    });

    chart.innerHTML = chartParts.join('');
    xAxis.innerHTML = xAxisParts.join('');

    if (elements.legend) {
        elements.legend.innerHTML = `
            <div class="modality-legend-item">
                <div class="modality-legend-swatch"></div>
                <span>Default Mesh</span>
            </div>
            <div class="modality-legend-item">
                <div class="modality-legend-swatch leaderboard-bar--noisy"></div>
                <span>Noisy Mesh</span>
            </div>
        `;
    }
}

function renderModalityImageChart(elements, imageEntries, metric) {
    const { plot, axisTitle, axisScale, groupLabels, chart, xAxis, emptyState } = elements;

    function showPlot() {
        plot.hidden = false;
        plot.style.display = '';
        emptyState.hidden = true;
        emptyState.setAttribute('aria-hidden', 'true');
        emptyState.style.display = 'none';
        emptyState.textContent = '';
    }

    function showEmptyState(message) {
        plot.hidden = true;
        plot.style.display = 'none';
        groupLabels.innerHTML = '';
        chart.innerHTML = '';
        xAxis.innerHTML = '';
        emptyState.hidden = false;
        emptyState.setAttribute('aria-hidden', 'false');
        emptyState.style.display = 'flex';
        emptyState.textContent = message;
    }

    const sorted = sortLeaderboardEntries(imageEntries, metric);
    if (sorted.length === 0) {
        showEmptyState('No image data available.');
        return;
    }

    const N = sorted.length;
    axisTitle.textContent = metric.label;

    const { minValue, maxValue } = getLeaderboardDomain(metric, sorted);
    showPlot();
    renderLeaderboardAxisScale(axisScale, metric, minValue, maxValue);

    // Grid: 3 bars per model [default, photo, multi] with small gaps between models
    const gridParts = [];
    for (let i = 0; i < N; i++) {
        if (i > 0) gridParts.push('8px');
        gridParts.push('1fr 1fr 1fr');
    }
    const gridTemplate = gridParts.join(' ');
    groupLabels.style.gridTemplateColumns = gridTemplate;
    chart.style.gridTemplateColumns = gridTemplate;
    xAxis.style.gridTemplateColumns = gridTemplate;

    const totalCols = 4 * N - 1;
    groupLabels.innerHTML = `<div class="leaderboard-group-label" style="grid-column: 1 / span ${totalCols};">Image-to-CAD</div>`;

    function barHeight(val) {
        if (val == null || !Number.isFinite(val)) return 0;
        return maxValue === minValue ? 100 : Math.max(0, Math.min(100, ((val - minValue) / (maxValue - minValue)) * 100));
    }

    function deltaHtml(d) {
        if (d == null || !Number.isFinite(d)) return '';
        const label = d >= 0 ? `+${metric.formatter(d)}` : metric.formatter(d);
        const cls = d >= 0 ? 'leaderboard-model-delta--positive' : 'leaderboard-model-delta--negative';
        return `<span class="leaderboard-model-delta leaderboard-model-delta--sm ${cls}">${escapeHtml(label)}</span>`;
    }

    const chartParts = [];
    const xAxisParts = [];

    sorted.forEach((entry, i) => {
        const defaultH = barHeight(entry.value);
        const photoH = barHeight(entry.photo_value);
        const multiH = barHeight(entry.multi_value);

        if (i > 0) {
            chartParts.push('<div class="leaderboard-divider"></div>');
            xAxisParts.push('<div class="leaderboard-divider"></div>');
        }

        chartParts.push(`
            <div class="leaderboard-bar-group">
                <div class="leaderboard-bar-shell" style="--bar-height: ${defaultH}%; --bar-color: ${entry.color};">
                    <span class="leaderboard-model-value">${metric.formatter(entry.value)}</span>
                    <div class="leaderboard-bar"></div>
                </div>
            </div>
            <div class="leaderboard-bar-group">
                <div class="leaderboard-bar-shell" style="--bar-height: ${photoH}%; --bar-color: ${entry.color};">
                    ${deltaHtml(entry.photo_delta)}
                    <span class="leaderboard-model-value">${entry.photo_value != null ? metric.formatter(entry.photo_value) : ''}</span>
                    <div class="leaderboard-bar leaderboard-bar--photo"></div>
                </div>
            </div>
            <div class="leaderboard-bar-group">
                <div class="leaderboard-bar-shell" style="--bar-height: ${multiH}%; --bar-color: ${entry.color};">
                    ${deltaHtml(entry.multi_delta)}
                    <span class="leaderboard-model-value">${entry.multi_value != null ? metric.formatter(entry.multi_value) : ''}</span>
                    <div class="leaderboard-bar leaderboard-bar--multi"></div>
                </div>
            </div>
        `);

        xAxisParts.push(`<span class="leaderboard-model-name" style="grid-column: span 3;">${escapeHtml(entry.name)}</span>`);
    });

    chart.innerHTML = chartParts.join('');
    xAxis.innerHTML = xAxisParts.join('');

    if (elements.legend) {
        elements.legend.innerHTML = `
            <div class="modality-legend-item">
                <div class="modality-legend-swatch"></div>
                <span>Single-view</span>
            </div>
            <div class="modality-legend-item">
                <div class="modality-legend-swatch leaderboard-bar--photo"></div>
                <span>Photorealistic</span>
            </div>
            <div class="modality-legend-item">
                <div class="modality-legend-swatch leaderboard-bar--multi"></div>
                <span>Multi-view</span>
            </div>
        `;
    }
}

function setupModalityLeaderboard() {
    const root = document.getElementById('modality-leaderboard-root');
    if (!root) return;
    const container = root.querySelector('[data-role="modality-carousel"]');
    if (!container) return;

    const MODALITY_SLIDES = ['Mesh-to-CAD', 'Image-to-CAD'];

    function makeSlideHTML(index) {
        return `
            <div class="modality-slide"${index !== 0 ? ' hidden' : ''}>
                <div class="leaderboard-chart-card">
                    <div class="leaderboard-plot" data-slide-role="plot">
                        <div class="leaderboard-y-axis">
                            <p class="leaderboard-axis-title" data-slide-role="axis-title">IoU</p>
                            <div class="leaderboard-axis-scale" data-slide-role="axis-scale" aria-hidden="true"></div>
                        </div>
                        <div class="leaderboard-plot-main">
                            <div class="leaderboard-group-labels" data-slide-role="group-labels" aria-hidden="true"></div>
                            <div class="leaderboard-chart" data-slide-role="chart" aria-live="polite"></div>
                            <div class="leaderboard-x-axis" data-slide-role="x-axis" aria-hidden="true"></div>
                            <p class="leaderboard-x-axis-title">Models</p>
                        </div>
                    </div>
                    <div class="leaderboard-empty-state" data-slide-role="empty-state" hidden aria-hidden="true" style="display: none;"></div>
                    <div class="leaderboard-modality-legend" data-slide-role="modality-legend"></div>
                </div>
            </div>
        `;
    }

    container.innerHTML = `
        <div class="bench-carousel-nav">
            <button class="bench-carousel-btn modality-prev-btn" aria-label="Previous modality">&#8592; Prev</button>
            <span class="bench-carousel-indicator">
                <span class="modality-slide-name">${escapeHtml(MODALITY_SLIDES[0])}</span>
                <span class="modality-carousel-counter">1 / ${MODALITY_SLIDES.length}</span>
            </span>
            <button class="bench-carousel-btn modality-next-btn" aria-label="Next modality">Next &#8594;</button>
        </div>
        <div class="bench-carousel-slides">
            ${MODALITY_SLIDES.map((_, i) => makeSlideHTML(i)).join('')}
        </div>
    `;

    const slideEls = Array.from(container.querySelectorAll('.modality-slide'));
    const slideNameEl = container.querySelector('.modality-slide-name');
    const counterEl = container.querySelector('.modality-carousel-counter');
    const prevBtn = container.querySelector('.modality-prev-btn');
    const nextBtn = container.querySelector('.modality-next-btn');
    let currentSlide = 0;

    function getSlideElements(slideEl) {
        return {
            plot: slideEl.querySelector('[data-slide-role="plot"]'),
            axisTitle: slideEl.querySelector('[data-slide-role="axis-title"]'),
            axisScale: slideEl.querySelector('[data-slide-role="axis-scale"]'),
            groupLabels: slideEl.querySelector('[data-slide-role="group-labels"]'),
            chart: slideEl.querySelector('[data-slide-role="chart"]'),
            xAxis: slideEl.querySelector('[data-slide-role="x-axis"]'),
            emptyState: slideEl.querySelector('[data-slide-role="empty-state"]'),
            legend: slideEl.querySelector('[data-slide-role="modality-legend"]'),
            tableWrapper: null
        };
    }

    function getSlideMetric(slideIndex) {
        const base = leaderboardMetrics.iou;
        const entriesBySeries = slideIndex === 0
            ? { mesh: base.entriesBySeries.mesh, image: [] }
            : { mesh: [], image: base.entriesBySeries.image };
        return Object.assign({}, base, { entriesBySeries });
    }

    function goToSlide(index) {
        currentSlide = index;
        slideEls.forEach((el, i) => { el.hidden = i !== index; });
        if (slideNameEl) slideNameEl.textContent = MODALITY_SLIDES[index];
        if (counterEl) counterEl.textContent = `${index + 1} / ${MODALITY_SLIDES.length}`;
    }

    slideEls.forEach((slideEl, slideIndex) => {
        const els = getSlideElements(slideEl);
        if (slideIndex === 0) {
            renderModalityMeshChart(els, leaderboardMetrics.iou.entriesBySeries.mesh, leaderboardMetrics.iou);
        } else {
            const EXCLUDED_IMAGE = new Set(['Qwen 3.527B', 'Qwen 3.59B']);
            const filteredImage = leaderboardMetrics.iou.entriesBySeries.image.filter(e => !EXCLUDED_IMAGE.has(e.name));
            renderModalityImageChart(els, filteredImage, leaderboardMetrics.iou);
        }
    });

    prevBtn.addEventListener('click', function() {
        goToSlide((currentSlide - 1 + MODALITY_SLIDES.length) % MODALITY_SLIDES.length);
    });
    nextBtn.addEventListener('click', function() {
        goToSlide((currentSlide + 1) % MODALITY_SLIDES.length);
    });
}

$(document).ready(function() {
    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    bulmaSlider.attach();

    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

    // Hydrate data once, then wire up both leaderboards
    hydrateLeaderboardData().then(function() {
        setupLeaderboard();
        setupModalityLeaderboard();
        setupComplexityLineChart();
        return Promise.all([0, 1, 2, 3, 4, 5].map(function(i) { return hydrateBenchSlideData(i); }));
    }).then(function() {
        setupBenchLeaderboard();
    });

})
