/* ==========================================================
   OnionQualityAI — application logic
   On-device onion grading (Grade A vs Unserviceable) with
   ONNX Runtime Web, lot summaries and exportable reports.
   ========================================================== */
(() => {
    'use strict';

    const NATIVE_ANDROID = typeof window !== 'undefined' && !!window.AndroidInference;

    /* ------------------------------------------------------
       CONFIGURATION
       ------------------------------------------------------ */
    const CONFIG = {
        modelUrl: 'onion_quality_model.onnx',
        infoUrl: 'model_information.json',

        // Fallbacks – overridden by model_information.json when present.
        imageSize: 224,
        classes: ['Grade_A', 'Unserviceable'], // output index order

        // ImageNet normalisation (must match training).
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],

        // Predictions below this confidence are flagged for manual review.
        lowConfidence: 0.80,

        maxFileBytes: 20 * 1024 * 1024,
        maxSamples: 500,
        captureQuality: 0.92,
        thumbSize: 112,

        // How each model class is presented in the UI and reports.
        classMeta: {
            Grade_A:       { key: 'gradeA', short: 'Grade A', long: 'Grade A',              tone: 'good' },
            Unserviceable: { key: 'urs',    short: 'URS',     long: 'Unserviceable (URS)',  tone: 'bad'  }
        }
    };


    /* ------------------------------------------------------
       STATE
       ------------------------------------------------------ */
    const state = {
        session: null,
        classes: CONFIG.classes.slice(),
        imageSize: CONFIG.imageSize,
        modelInfo: null,
        modelPromise: null,
        modelOk: false,

        stream: null,

        samples: [],
        nextNum: 1,
        selectedId: null,
        badgeId: null,
        badgeTimer: null,

        queue: Promise.resolve(),
        clearTimer: null,
        defaultLotId: ''
    };

    const itemEls = new Map();   // sample id -> <li>


    /* ------------------------------------------------------
       DOM
       ------------------------------------------------------ */
    const $ = (id) => document.getElementById(id);

    const els = {
        modelPill: $('modelPill'),
        modelStatus: $('modelStatus'),
        status: $('status'),

        viewfinder: $('viewfinder'),
        camera: $('camera'),
        preview: $('preview'),
        badge: $('vfBadge'),
        flash: $('flash'),

        cameraToggle: $('cameraToggle'),
        captureButton: $('captureButton'),
        dropzone: $('dropzone'),
        fileInput: $('fileInput'),
        nativeCamera: $('nativeCamera'),

        lotId: $('lotId'),
        inspector: $('inspector'),
        clearButton: $('clearButton'),
        csvButton: $('csvButton'),
        printButton: $('printButton'),

        gradeAPct: $('gradeAPct'),
        gradeACount: $('gradeACount'),
        ursPct: $('ursPct'),
        ursCount: $('ursCount'),
        totalCount: $('totalCount'),
        totalNote: $('totalNote'),
        splitBar: $('splitBar'),
        barA: $('barA'),
        barU: $('barU'),
        reviewNote: $('reviewNote'),
        list: $('sampleList'),
        empty: $('emptyState'),

        modelName: $('modelName'),
        mAccuracy: $('mAccuracy'),
        mPrecision: $('mPrecision'),
        mRecall: $('mRecall'),
        mF1: $('mF1'),

        report: $('report')
    };


    /* ------------------------------------------------------
       HELPERS
       ------------------------------------------------------ */
    const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;
    const pad = (n, width = 3) => String(n).padStart(width, '0');
    const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

    const escapeHtml = (value) =>
        String(value).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));

    const formatDateTime = (date) =>
        new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);

    const stamp = (date = new Date()) => {
        const p = (n) => String(n).padStart(2, '0');
        return `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}`;
    };

    const softmax = (logits) => {
        const max = Math.max(...logits);
        const exps = logits.map((v) => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map((v) => v / sum);
    };

    const isImageFile = (file) =>
        (file.type && file.type.startsWith('image/')) ||
        /\.(jpe?g|png|webp|gif|bmp|heic|heif)$/i.test(file.name);

    const metaFor = (className) =>
        CONFIG.classMeta[className] || {
            key: className, short: className, long: className, tone: 'good'
        };

    function setStatus(message, tone = 'info') {
        els.status.textContent = message;
        els.status.dataset.tone = tone;
    }

    function setModelState(stateName, text) {
        els.modelPill.dataset.state = stateName;
        els.modelStatus.textContent = text;
    }

    const iconSvg = (id) =>
        `<svg class="icon" aria-hidden="true"><use href="#i-${id}"/></svg>`;


    /* ------------------------------------------------------
       MODEL
       ------------------------------------------------------ */
    async function loadModelInfo() {
        try {
            const response = await fetch(CONFIG.infoUrl, { cache: 'no-cache' });
            if (!response.ok) return null;
            return await response.json();
        } catch {
            return null;
        }
    }

    function renderModelInfo(info) {
        const fmt = (v) => (typeof v === 'number' ? pct(v) : '—');
        els.modelName.textContent = (info && info.model) || 'MobileNetV3-Small';
        els.mAccuracy.textContent = fmt(info && info.test_accuracy);
        els.mPrecision.textContent = fmt(info && info.precision);
        els.mRecall.textContent = fmt(info && info.recall);
        els.mF1.textContent = fmt(info && info.f1_score);
    }

    async function loadModel() {
        setModelState('loading', 'Loading AI');
        setStatus('Loading AI model…');

        try {
            if (NATIVE_ANDROID) {
                const info = await loadModelInfo();
                if (info) {
                    state.modelInfo = info;
                    if (Array.isArray(info.classes) && info.classes.length) state.classes = info.classes;
                    if (Number.isInteger(info.image_size) && info.image_size > 0) state.imageSize = info.image_size;
                }
                renderModelInfo(info);
                state.modelOk = true;
                setModelState('ready', 'AI ready');
                setStatus('AI model ready. Capture or upload an onion.', 'success');
                return true;
            }

            if (location.protocol === 'file:') {
                throw new Error(
                    'Open this app through the local server (run_server.bat → http://localhost:8000). ' +
                    'Browsers block model loading from file://.'
                );
            }
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime could not be loaded. Check your internet connection and reload.');
            }

            const [info, session] = await Promise.all([
                loadModelInfo(),
                ort.InferenceSession.create(CONFIG.modelUrl, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                })
            ]);

            if (info) {
                state.modelInfo = info;
                if (Array.isArray(info.classes) && info.classes.length) state.classes = info.classes;
                if (Number.isInteger(info.image_size) && info.image_size > 0) state.imageSize = info.image_size;
            }
            renderModelInfo(info);

            state.session = session;
            await warmUp();

            state.modelOk = true;
            setModelState('ready', 'AI ready');
            setStatus('AI model ready. Capture or upload an onion.', 'success');
            return true;

        } catch (error) {
            console.error('Model load failed:', error);
            renderModelInfo(state.modelInfo);
            state.modelOk = false;
            setModelState('error', 'AI error');
            setStatus(`Model failed to load. ${error.message || 'Check the browser console.'}`, 'error');
            return false;
        }
    }

    // First inference is slow (graph compile); do it once up front.
    async function warmUp() {
        const s = state.imageSize;
        const dummy = new ort.Tensor('float32', new Float32Array(3 * s * s), [1, 3, s, s]);
        await state.session.run({ [state.session.inputNames[0]]: dummy });
    }


    /* ------------------------------------------------------
       IMAGE PRE-PROCESSING
       ------------------------------------------------------ */
    let workCanvas = null;
    let workCtx = null;

    function getWorkCanvas(size) {
        if (!workCanvas || workCanvas.width !== size) {
            workCanvas = document.createElement('canvas');
            workCanvas.width = workCanvas.height = size;
            workCtx = workCanvas.getContext('2d', { willReadFrequently: true });
            workCtx.imageSmoothingQuality = 'high';
        }
        return workCtx;
    }

    // Decode a File honouring EXIF orientation (phone photos).
    async function decodeImage(file) {
        if ('createImageBitmap' in window) {
            try {
                return await createImageBitmap(file, { imageOrientation: 'from-image' });
            } catch { /* fall through to <img> decoding */ }
        }
        const url = URL.createObjectURL(file);
        try {
            const img = new Image();
            img.decoding = 'async';
            img.src = url;
            await img.decode();
            return img;
        } finally {
            URL.revokeObjectURL(url);
        }
    }

    // Whole image → size×size (same as Resize((size, size)) at training time).
    function toTensor(source) {
        const size = state.imageSize;
        const ctx = getWorkCanvas(size);
        ctx.drawImage(source, 0, 0, size, size);

        const { data } = ctx.getImageData(0, 0, size, size);
        const plane = size * size;
        const out = new Float32Array(3 * plane);
        const { mean, std } = CONFIG;

        for (let i = 0, p = 0; i < plane; i++, p += 4) {
            out[i]             = (data[p]     / 255 - mean[0]) / std[0];
            out[plane + i]     = (data[p + 1] / 255 - mean[1]) / std[1];
            out[2 * plane + i] = (data[p + 2] / 255 - mean[2]) / std[2];
        }
        return new ort.Tensor('float32', out, [1, 3, size, size]);
    }

    function sourceToDataUrl(source, maxSide = 1200) {
        const w = source.naturalWidth || source.videoWidth || source.width;
        const h = source.naturalHeight || source.videoHeight || source.height;
        if (!w || !h) throw new Error('Image dimensions are unavailable');

        const scale = Math.min(1, maxSide / Math.max(w, h));
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(w * scale));
        canvas.height = Math.max(1, Math.round(h * scale));
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.9);
    }

    function makeThumbnail(source) {
        const size = CONFIG.thumbSize;
        const w = source.naturalWidth || source.width;
        const h = source.naturalHeight || source.height;
        const side = Math.min(w, h);

        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = size;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(source, (w - side) / 2, (h - side) / 2, side, side, 0, 0, size, size);
        return canvas.toDataURL('image/jpeg', 0.75);
    }


    /* ------------------------------------------------------
       INFERENCE QUEUE
       ------------------------------------------------------ */
    function enqueue(sample) {
        state.queue = state.queue
            .then(() => analyze(sample))
            .catch((error) => console.error('Queue error:', error));
    }

    async function analyze(sample) {
        if (!state.samples.includes(sample)) return;        // removed while queued

        sample.status = 'analyzing';
        refreshSample(sample);
        renderViewer();

        try {
            const ready = await state.modelPromise;
            if (!ready) throw new Error('Model unavailable');

            const started = performance.now();

            const bitmap = await decodeImage(sample.file);
            sample.thumb = makeThumbnail(bitmap);

            let probs;
            let best = 0;

            if (NATIVE_ANDROID) {
                const startedNative = performance.now();
                let result = null;
                let lastError = null;

                // A handful of phone photos are too large / an odd format for a single
                // pass to decode natively (this is what used to surface as
                // "could not analyze image" even though most photos worked fine).
                // Retry once at a smaller size before giving up on the sample.
                for (const maxSide of [1200, 700]) {
                    try {
                        const payload = sourceToDataUrl(bitmap, maxSide);
                        result = JSON.parse(window.AndroidInference.analyze(payload));
                        if (result.ok) break;
                        lastError = new Error(result.error || 'Native inference failed');
                    } catch (err) {
                        lastError = err;
                        result = null;
                    }
                }

                if (!result || !result.ok) throw (lastError || new Error('Native inference failed'));
                probs = result.probs || [];
                if (Array.isArray(probs) && probs.length) {
                    probs.forEach((p, i) => { if (p > probs[best]) best = i; });
                } else {
                    throw new Error('Native inference returned no probabilities');
                }
                sample.ms = Number.isFinite(result.ms) ? Math.round(result.ms) : Math.round(performance.now() - startedNative);
            } else {
                const tensor = toTensor(bitmap);
                const session = state.session;
                const outputs = await session.run({ [session.inputNames[0]]: tensor });
                const logits = Array.from(outputs[session.outputNames[0]].data);
                probs = softmax(logits);
                probs.forEach((p, i) => { if (p > probs[best]) best = i; });
            }

            if (typeof bitmap.close === 'function') bitmap.close();

            const className = state.classes[best] ?? `Class ${best}`;
            const meta = metaFor(className);

            sample.className = className;
            sample.classKey = meta.key;
            sample.probs = probs;
            sample.confidence = probs[best];
            sample.low = sample.confidence < CONFIG.lowConfidence;
            if (!sample.ms) sample.ms = Math.round(performance.now() - started);
            sample.status = 'done';

        } catch (error) {
            console.error('Analysis failed:', error);
            sample.status = 'error';
            sample.error = state.modelOk
                ? 'Could not read this image'
                : 'Model unavailable';
        }

        if (!state.samples.includes(sample)) return;

        refreshSample(sample);
        renderSummary();
        renderViewer();

        if (sample.status === 'done') {
            const meta = metaFor(sample.className);
            setStatus(
                `Sample #${pad(sample.num)}: ${meta.long} · ${pct(sample.confidence)} confidence` +
                (sample.low ? ' · low confidence, verify manually' : ''),
                sample.low ? 'warn' : 'success'
            );
            if (state.stream && state.badgeId === sample.id) scheduleBadgeClear();
        } else {
            setStatus(`Sample #${pad(sample.num)} could not be analysed.`, 'error');
        }
    }


    /* ------------------------------------------------------
       SAMPLES
       ------------------------------------------------------ */
    function addFiles(fileList, { fromCamera = false } = {}) {
        const files = Array.from(fileList || []);
        if (!files.length) return;

        let added = 0, badType = 0, tooBig = 0, overCap = 0, last = null;

        for (const file of files) {
            if (state.samples.length >= CONFIG.maxSamples) { overCap++; continue; }
            if (!isImageFile(file)) { badType++; continue; }
            if (file.size > CONFIG.maxFileBytes) { tooBig++; continue; }

            const num = state.nextNum++;
            const sample = {
                id: `s${num}`,
                num,
                name: file.name || `image-${num}`,
                file,
                url: URL.createObjectURL(file),
                thumb: '',
                status: 'pending',
                className: null,
                classKey: null,
                probs: null,
                confidence: 0,
                low: false,
                ms: 0,
                error: '',
                addedAt: new Date()
            };

            state.samples.push(sample);
            const li = createSampleEl(sample);
            itemEls.set(sample.id, li);
            els.list.prepend(li);
            enqueue(sample);

            added++;
            last = sample;
        }

        if (last) select(last.id, { fromCamera });

        renderSummary();
        renderViewer();

        const skipped = badType + tooBig + overCap;
        if (added && !skipped) {
            setStatus(fromCamera
                ? 'Captured. Analysing…'
                : `${plural(added, 'image')} added. Analysing…`);
        } else if (added) {
            setStatus(`${plural(added, 'image')} added, ${skipped} skipped (unsupported type, over ${CONFIG.maxFileBytes / 1048576} MB, or lot is full).`, 'warn');
        } else {
            const why = overCap ? `The lot is limited to ${CONFIG.maxSamples} samples.`
                      : tooBig ? `Images must be under ${CONFIG.maxFileBytes / 1048576} MB.`
                      : 'Please choose JPG, PNG or WEBP images.';
            setStatus(why, 'error');
        }
    }

    function removeSample(id) {
        const index = state.samples.findIndex((s) => s.id === id);
        if (index === -1) return;

        const [sample] = state.samples.splice(index, 1);
        URL.revokeObjectURL(sample.url);

        const li = itemEls.get(id);
        if (li) li.remove();
        itemEls.delete(id);

        if (state.selectedId === id) {
            const next = state.samples[state.samples.length - 1] || null;
            state.selectedId = next ? next.id : null;
        }
        if (state.badgeId === id) state.badgeId = null;

        applySelectionClasses();
        renderSummary();
        renderViewer();
        setStatus(`Sample #${pad(sample.num)} removed.`);
    }

    function clearLot() {
        state.samples.forEach((s) => URL.revokeObjectURL(s.url));
        state.samples = [];
        state.selectedId = null;
        state.badgeId = null;
        itemEls.clear();
        els.list.replaceChildren();
        resetClearButton();

        els.preview.removeAttribute('src');
        renderSummary();
        renderViewer();
        setStatus('Lot cleared. Ready for a new set of samples.');
    }

    function select(id, { fromCamera = false } = {}) {
        state.selectedId = id;
        state.badgeId = fromCamera ? id : null;
        clearTimeout(state.badgeTimer);
        applySelectionClasses();
        renderViewer();
    }

    function applySelectionClasses() {
        itemEls.forEach((li, id) => {
            const on = id === state.selectedId;
            li.classList.toggle('is-selected', on);
            const btn = li.querySelector('.sample-main');
            if (on) btn.setAttribute('aria-current', 'true');
            else btn.removeAttribute('aria-current');
        });
    }

    function scheduleBadgeClear() {
        clearTimeout(state.badgeTimer);
        state.badgeTimer = setTimeout(() => {
            state.badgeId = null;
            renderViewer();
        }, 3500);
    }


    /* ------------------------------------------------------
       LIST ITEM RENDERING
       ------------------------------------------------------ */
    function createSampleEl(sample) {
        const li = document.createElement('li');
        li.className = 'sample';
        li.dataset.id = sample.id;
        li.innerHTML = `
            <button type="button" class="sample-main">
                <img class="thumb" alt="">
                <span class="sample-info">
                    <span class="sample-title">
                        <span class="sample-num"></span>
                        <span class="sample-name"></span>
                    </span>
                    <span class="sample-sub"></span>
                </span>
                <span class="tag"></span>
            </button>
            <button type="button" class="icon-btn sample-remove">${iconSvg('x')}</button>`;

        li.querySelector('.sample-num').textContent = `#${pad(sample.num)}`;
        li.querySelector('.sample-name').textContent = sample.name;
        li.querySelector('.sample-name').title = sample.name;

        li.querySelector('.sample-main').addEventListener('click', () => select(sample.id));
        li.querySelector('.sample-remove').addEventListener('click', () => removeSample(sample.id));
        li.querySelector('.sample-remove').setAttribute('aria-label', `Remove sample ${pad(sample.num)}`);

        updateSampleEl(li, sample);
        return li;
    }

    function refreshSample(sample) {
        const li = itemEls.get(sample.id);
        if (li) updateSampleEl(li, sample);
    }

    function updateSampleEl(li, sample) {
        const thumb = li.querySelector('.thumb');
        if (sample.thumb && thumb.getAttribute('src') !== sample.thumb) thumb.src = sample.thumb;

        const sub = li.querySelector('.sample-sub');
        const tag = li.querySelector('.tag');
        sub.classList.remove('is-low');

        switch (sample.status) {
            case 'pending':
                sub.textContent = 'Queued';
                tag.className = 'tag tag-pending';
                tag.textContent = 'Queued';
                break;

            case 'analyzing':
                sub.textContent = 'Analysing…';
                tag.className = 'tag tag-pending';
                tag.innerHTML = '<span class="spinner" aria-hidden="true"></span>Analysing';
                break;

            case 'done': {
                const meta = metaFor(sample.className);
                sub.textContent = `${pct(sample.confidence)} confidence · ${sample.ms} ms` +
                                  (sample.low ? ' · Low confidence' : '');
                sub.classList.toggle('is-low', sample.low);
                tag.className = `tag tag-${meta.tone}`;
                tag.textContent = meta.short;
                break;
            }

            default:
                sub.textContent = sample.error || 'Analysis failed';
                tag.className = 'tag tag-error';
                tag.textContent = 'Error';
        }

        li.querySelector('.sample-main').setAttribute(
            'aria-label',
            `Sample ${pad(sample.num)}, ${sample.status === 'done'
                ? `${metaFor(sample.className).long}, ${pct(sample.confidence)} confidence`
                : sample.status}`
        );
    }


    /* ------------------------------------------------------
       VIEWFINDER
       ------------------------------------------------------ */
    function renderViewer() {
        const cameraOn = !!state.stream;
        const selected = state.samples.find((s) => s.id === state.selectedId) || null;

        let mode = 'empty';
        if (cameraOn) mode = 'camera';
        else if (selected) mode = 'preview';

        els.viewfinder.dataset.mode = mode;
        els.viewfinder.dataset.scanning =
            String(mode === 'preview' && selected && (selected.status === 'pending' || selected.status === 'analyzing'));

        if (mode === 'preview' && els.preview.dataset.id !== selected.id) {
            els.preview.src = selected.url;
            els.preview.dataset.id = selected.id;
            els.preview.alt = `Sample ${pad(selected.num)}: ${selected.name}`;
        }

        // Result badge: always in preview mode, briefly after a live capture.
        const badgeSample = cameraOn
            ? state.samples.find((s) => s.id === state.badgeId)
            : selected;

        if (!badgeSample) {
            els.badge.hidden = true;
            return;
        }

        els.badge.hidden = false;

        if (badgeSample.status === 'done') {
            const meta = metaFor(badgeSample.className);
            els.badge.dataset.tone = meta.tone;
            els.badge.innerHTML =
                `${iconSvg(meta.tone === 'good' ? 'check' : 'alert')}` +
                `<span>${escapeHtml(meta.long)}</span>` +
                `<small>${pct(badgeSample.confidence)}</small>`;
        } else if (badgeSample.status === 'error') {
            els.badge.dataset.tone = 'error';
            els.badge.innerHTML = `${iconSvg('alert')}<span>Could not analyse</span>`;
        } else {
            els.badge.dataset.tone = 'pending';
            els.badge.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Analysing…</span>';
        }
    }

    function flashViewfinder() {
        els.flash.classList.remove('on');
        void els.flash.offsetWidth;            // restart animation
        els.flash.classList.add('on');
    }


    /* ------------------------------------------------------
       LOT SUMMARY
       ------------------------------------------------------ */
    function computeSummary() {
        const done = state.samples.filter((s) => s.status === 'done');
        const gradeA = done.filter((s) => s.classKey === 'gradeA').length;
        const urs = done.filter((s) => s.classKey === 'urs').length;
        const review = done.filter((s) => s.low).length;
        const busy = state.samples.filter((s) => s.status === 'pending' || s.status === 'analyzing').length;
        const failed = state.samples.filter((s) => s.status === 'error').length;
        const avgMs = done.length ? Math.round(done.reduce((a, s) => a + s.ms, 0) / done.length) : 0;

        return { done, total: done.length, gradeA, urs, review, busy, failed, avgMs };
    }

    function renderSummary() {
        const s = computeSummary();
        const has = s.total > 0;

        els.gradeAPct.textContent = has ? pct(s.gradeA / s.total) : '—';
        els.ursPct.textContent = has ? pct(s.urs / s.total) : '—';
        els.gradeACount.textContent = plural(s.gradeA, 'sample');
        els.ursCount.textContent = plural(s.urs, 'sample');

        els.totalCount.textContent = String(state.samples.length);
        els.totalNote.textContent =
            s.busy ? `${s.busy} in progress` :
            has ? `Avg ${s.avgMs} ms` :
            'None yet';

        const a = has ? (s.gradeA / s.total) * 100 : 0;
        const u = has ? (s.urs / s.total) * 100 : 0;
        els.barA.style.width = `${a}%`;
        els.barU.style.width = `${u}%`;
        els.splitBar.setAttribute('aria-label', has
            ? `Grade A ${pct(s.gradeA / s.total)}, URS ${pct(s.urs / s.total)}`
            : 'No samples analysed yet');

        if (s.review) {
            els.reviewNote.hidden = false;
            els.reviewNote.innerHTML =
                `${iconSvg('alert')}${plural(s.review, 'sample')} to verify`;
        } else {
            els.reviewNote.hidden = true;
        }

        const any = state.samples.length > 0;
        els.list.hidden = !any;
        els.empty.hidden = any;
        els.clearButton.disabled = !any;
        els.csvButton.disabled = !has;
        els.printButton.disabled = !has;
    }

    function resetClearButton() {
        clearTimeout(state.clearTimer);
        els.clearButton.classList.remove('is-confirm');
        els.clearButton.querySelector('span').textContent = 'Clear';
    }

    function onClearClick() {
        if (els.clearButton.classList.contains('is-confirm')) {
            clearLot();
            return;
        }
        els.clearButton.classList.add('is-confirm');
        els.clearButton.querySelector('span').textContent = 'Confirm clear';
        state.clearTimer = setTimeout(resetClearButton, 3500);
    }


    /* ------------------------------------------------------
       CAMERA
       ------------------------------------------------------ */
    const liveCameraSupported = () =>
        window.isSecureContext && !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

    function cameraErrorMessage(error) {
        switch (error && error.name) {
            case 'NotAllowedError':
            case 'SecurityError':
                return 'Camera permission was denied. Allow access in your browser settings, or upload a photo instead.';
            case 'NotFoundError':
            case 'OverconstrainedError':
                return 'No camera was found on this device.';
            case 'NotReadableError':
                return 'The camera is being used by another app.';
            default:
                return 'The camera could not be started.';
        }
    }

    // How long we'll wait for the live preview to actually produce a frame
    // before giving up on it and opening the device's own camera app instead.
    const CAMERA_READY_TIMEOUT_MS = 4000;

    function setCameraUi(on, ready = on) {
        // `on` = stream requested/running (controls the Start/Stop label).
        // `ready` = video is actually producing frames (controls the Capture button).
        els.captureButton.disabled = !ready;
        els.cameraToggle.classList.toggle('is-live', on);
        els.cameraToggle.innerHTML =
            `${iconSvg(on ? 'stop' : 'camera')}<span>${on ? 'Stop camera' : 'Start camera'}</span>`;
    }

    async function startCamera() {
        if (!liveCameraSupported()) {
            // Live preview needs HTTPS/localhost; fall back to the OS camera picker.
            setStatus('Live preview needs HTTPS or localhost. Opening your device camera instead.', 'warn');
            els.nativeCamera.click();
            return;
        }

        els.cameraToggle.disabled = true;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

            state.stream = stream;
            els.camera.srcObject = stream;

            const track = stream.getVideoTracks()[0];
            if (track) track.addEventListener('ended', stopCamera);

            setCameraUi(true, false);
            renderViewer();
            setStatus('Starting camera…');

            const gotFrame = await waitForVideoReady(els.camera, CAMERA_READY_TIMEOUT_MS);

            if (!state.stream) return; // stopped while we were waiting

            if (!gotFrame) {
                // The stream opened but never actually rendered a frame — a known
                // quirk on some Android WebViews. Don't leave a dead preview and a
                // capture button that silently does nothing: fall back to the
                // device's own camera app, which always works.
                stopCamera();
                setStatus('Live preview isn\u2019t working on this device. Opening your device camera instead.', 'warn');
                els.nativeCamera.click();
                return;
            }

            await els.camera.play().catch(() => {});
            setCameraUi(true, true);
            renderViewer();
            setStatus('Camera ready. Position the onion inside the frame, then tap Capture.', 'success');

        } catch (error) {
            console.error('Camera error:', error);
            state.stream = null;
            setCameraUi(false);
            setStatus(cameraErrorMessage(error), 'error');
        } finally {
            els.cameraToggle.disabled = false;
        }
    }

    // Resolves true once the <video> element has real dimensions (a frame is
    // flowing), or false if that doesn't happen within `timeoutMs`.
    function waitForVideoReady(video, timeoutMs) {
        if (video.videoWidth && video.videoHeight) return Promise.resolve(true);
        return new Promise((resolve) => {
            let done = false;
            const finish = (ok) => {
                if (done) return;
                done = true;
                video.removeEventListener('loadedmetadata', onReady);
                video.removeEventListener('loadeddata', onReady);
                clearTimeout(timer);
                resolve(ok);
            };
            const onReady = () => { if (video.videoWidth && video.videoHeight) finish(true); };
            video.addEventListener('loadedmetadata', onReady);
            video.addEventListener('loadeddata', onReady);
            const timer = setTimeout(() => finish(false), timeoutMs);
        });
    }

    function stopCamera() {
        if (state.stream) {
            state.stream.getTracks().forEach((t) => t.stop());
            state.stream = null;
        }
        els.camera.srcObject = null;
        clearTimeout(state.badgeTimer);
        state.badgeId = null;
        setCameraUi(false);
        renderViewer();
    }

    function toggleCamera() {
        if (state.stream) {
            stopCamera();
            setStatus('Camera stopped.');
        } else {
            startCamera();
        }
    }

    function capture() {
        const video = els.camera;
        if (!state.stream || !video.videoWidth) {
            setStatus('Camera is still starting — wait a moment and try again.', 'warn');
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

        flashViewfinder();

        canvas.toBlob((blob) => {
            if (!blob) {
                setStatus('Capture failed. Please try again.', 'error');
                return;
            }
            const file = new File([blob], `camera-${pad(state.nextNum)}.jpg`, { type: 'image/jpeg' });
            addFiles([file], { fromCamera: true });
        }, 'image/jpeg', CONFIG.captureQuality);
    }


    /* ------------------------------------------------------
       EXPORT: CSV + PRINTABLE REPORT
       ------------------------------------------------------ */
    function lotMeta() {
        return {
            lotId: els.lotId.value.trim() || state.defaultLotId,
            inspector: els.inspector.value.trim(),
            generated: new Date()
        };
    }

    const doneSamples = () =>
        state.samples.filter((s) => s.status === 'done').sort((a, b) => a.num - b.num);

    function csvCell(value) {
        let text = value == null ? '' : String(value);
        if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;           // block spreadsheet formula injection
        return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    }

    function downloadCsv() {
        const meta = lotMeta();
        const rows = doneSamples();
        const sum = computeSummary();

        const header = [
            'Lot ID', 'Inspector', 'Report generated', 'Sample', 'File',
            'Result', 'Confidence %', 'Grade A probability %', 'URS probability %',
            'Manual review', 'Inference ms'
        ];

        const idxA = state.classes.indexOf('Grade_A');
        const idxU = state.classes.indexOf('Unserviceable');

        const lines = [header.map(csvCell).join(',')];
        const generated = meta.generated.toISOString();

        rows.forEach((s) => {
            lines.push([
                meta.lotId,
                meta.inspector,
                generated,
                pad(s.num),
                s.name,
                metaFor(s.className).long,
                (s.confidence * 100).toFixed(1),
                idxA >= 0 ? (s.probs[idxA] * 100).toFixed(1) : '',
                idxU >= 0 ? (s.probs[idxU] * 100).toFixed(1) : '',
                s.low ? 'Yes' : 'No',
                s.ms
            ].map(csvCell).join(','));
        });

        lines.push('');
        lines.push(['Summary', 'Samples analysed', sum.total].map(csvCell).join(','));
        lines.push(['Summary', 'Grade A %', (sum.gradeA / sum.total * 100).toFixed(1)].map(csvCell).join(','));
        lines.push(['Summary', 'URS %', (sum.urs / sum.total * 100).toFixed(1)].map(csvCell).join(','));
        lines.push(['Summary', 'Flagged for manual review', sum.review].map(csvCell).join(','));

        const safeId = meta.lotId.replace(/[^\w.-]+/g, '_').slice(0, 40) || 'lot';
        const fileName = `onion-quality_${safeId}_${stamp(meta.generated)}.csv`;
        const csvText = lines.join('\r\n');

        // Android WebView cannot download blob: links, so hand the file to the native app.
        if (window.AndroidApp && typeof window.AndroidApp.saveCsv === 'function') {
            window.AndroidApp.saveCsv(fileName, csvText);
            setStatus('Choose where to save the CSV report.', 'info');
            return;
        }

        const blob = new Blob(['\uFEFF' + csvText], { type: 'text/csv;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);

        setStatus('CSV report downloaded.', 'success');
    }

    function renderReport() {
        const meta = lotMeta();
        const rows = doneSamples();
        const sum = computeSummary();
        const info = state.modelInfo || {};
        const fmt = (v) => (typeof v === 'number' ? pct(v) : '—');

        const bodyRows = rows.map((s) => {
            const m = metaFor(s.className);
            return `<tr>
                <td>${pad(s.num)}</td>
                <td>${escapeHtml(s.name)}</td>
                <td class="res-${m.tone}">${escapeHtml(m.long)}</td>
                <td class="num">${pct(s.confidence)}</td>
                <td class="${s.low ? 'flag' : ''}">${s.low ? 'Verify manually' : '—'}</td>
            </tr>`;
        }).join('');

        els.report.innerHTML = `
            <div class="report-head">
                <div>
                    <div class="report-brand">OnionQualityAI</div>
                    <h1>Onion Quality Inspection Report</h1>
                </div>
                <p>Generated ${escapeHtml(formatDateTime(meta.generated))}</p>
            </div>

            <dl class="report-meta">
                <div><dt>Lot / centre ID</dt><dd>${escapeHtml(meta.lotId)}</dd></div>
                <div><dt>Inspector</dt><dd>${escapeHtml(meta.inspector || '—')}</dd></div>
                <div><dt>Samples analysed</dt><dd>${sum.total}</dd></div>
                <div><dt>Manual review</dt><dd>${sum.review}</dd></div>
            </dl>

            <div class="report-summary">
                <div class="good"><span>Grade A</span><strong>${pct(sum.gradeA / sum.total)}</strong><small>${plural(sum.gradeA, 'sample')}</small></div>
                <div class="bad"><span>URS (Unserviceable)</span><strong>${pct(sum.urs / sum.total)}</strong><small>${plural(sum.urs, 'sample')}</small></div>
                <div><span>Low-confidence</span><strong>${sum.review}</strong><small>below ${pct(CONFIG.lowConfidence, 0)} confidence</small></div>
            </div>

            <h2>Sample results</h2>
            <table>
                <thead><tr><th>#</th><th>File</th><th>Result</th><th class="num">Confidence</th><th>Review</th></tr></thead>
                <tbody>${bodyRows}</tbody>
            </table>

            <div class="report-foot">
                <h2>Model</h2>
                <p>
                    ${escapeHtml(info.model || 'MobileNetV3-Small')} · ${state.imageSize}×${state.imageSize} input ·
                    test accuracy ${fmt(info.test_accuracy)}, precision ${fmt(info.precision)},
                    recall ${fmt(info.recall)}, F1 ${fmt(info.f1_score)}.
                </p>
                <p style="margin-top:6pt">
                    Percentages are calculated by sample (image) count. This is an AI-assisted assessment;
                    the final acceptance decision rests with the authorised inspector.
                </p>
                <div class="report-sign">
                    <div>Inspector signature</div>
                    <div>Date</div>
                </div>
            </div>`;
    }

    function printReport() {
        if (!doneSamples().length) return;
        renderReport();
        if (window.AndroidApp && typeof window.AndroidApp.printPage === 'function') {
            window.AndroidApp.printPage();      // WebView ignores window.print()
        } else {
            window.print();
        }
    }


    /* ------------------------------------------------------
       EVENTS
       ------------------------------------------------------ */
    function bindEvents() {
        els.cameraToggle.addEventListener('click', toggleCamera);
        els.captureButton.addEventListener('click', capture);

        els.fileInput.addEventListener('change', (e) => {
            addFiles(e.target.files);
            e.target.value = '';
        });
        els.nativeCamera.addEventListener('change', (e) => {
            addFiles(e.target.files, { fromCamera: false });
            e.target.value = '';
        });

        // Drag & drop (dropzone and viewfinder)
        const dropTargets = [els.dropzone, els.viewfinder];
        dropTargets.forEach((target) => {
            target.addEventListener('dragenter', (e) => { e.preventDefault(); els.dropzone.classList.add('is-dragover'); });
            target.addEventListener('dragover',  (e) => { e.preventDefault(); els.dropzone.classList.add('is-dragover'); });
            target.addEventListener('dragleave', () => els.dropzone.classList.remove('is-dragover'));
            target.addEventListener('drop', (e) => {
                e.preventDefault();
                els.dropzone.classList.remove('is-dragover');
                addFiles(e.dataTransfer && e.dataTransfer.files);
            });
        });
        // Stop the browser navigating away when a file is dropped elsewhere.
        window.addEventListener('dragover', (e) => e.preventDefault());
        window.addEventListener('drop', (e) => e.preventDefault());

        // Paste an image from the clipboard
        window.addEventListener('paste', (e) => {
            const files = Array.from((e.clipboardData && e.clipboardData.files) || []);
            if (files.length) addFiles(files);
        });

        // Space bar captures while the live camera is on.
        window.addEventListener('keydown', (e) => {
            if (e.code !== 'Space' || !state.stream) return;
            const tag = e.target && e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'BUTTON') return;
            e.preventDefault();
            capture();
        });

        els.clearButton.addEventListener('click', onClearClick);
        els.csvButton.addEventListener('click', downloadCsv);
        els.printButton.addEventListener('click', printReport);

        // Ctrl/Cmd+P should print the report too.
        window.addEventListener('beforeprint', () => {
            if (doneSamples().length) renderReport();
            else els.report.innerHTML = '<p>No samples have been analysed yet.</p>';
        });

        // Release the camera when leaving the page or hiding the tab.
        window.addEventListener('pagehide', stopCamera);
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && state.stream) {
                stopCamera();
                setStatus('Camera paused while the tab was in the background.');
            }
        });
    }


    /* ------------------------------------------------------
       INIT
       ------------------------------------------------------ */
    function init() {
        state.defaultLotId = `LOT-${stamp()}`;
        els.lotId.value = state.defaultLotId;

        bindEvents();
        renderSummary();
        renderViewer();
        setCameraUi(false);

        state.modelPromise = loadModel();
    }

    init();

})();
