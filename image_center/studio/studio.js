'use strict';

const $ = id => document.getElementById(id);
const roles = { reference: '通用参考', identity: '人物身份（仅脸部）', clothing: '服装', background: '背景',
    pose: '姿势', style: '画面风格', object: '道具' };
const phases = { queued: '排队中', waiting: '等待引擎', loading: '加载模型', preparing: '准备输入',
    conditioning: '编码提示词和参考图', condition: '编码提示词和参考图', vae_encode: '编码参考图',
    sampling: '采样', vae_decode: '解码图片', decoding: '解码图片', generating: '生成中',
    retrying_memory: '正在以低显存模式重试', cancelling: '正在取消' };
const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
const samplePrompt = '一位年轻成年女性在宿舍窗边拍日常手机自拍。平视正脸，胸部以上近景，脸部清晰无遮挡，深色头发扎成低马尾，穿薄荷绿针织衫。左侧有一只黄色陶瓷杯，右侧台灯发出暖光，窗外是夜色。保留自然的皮肤纹理和清晰的发丝。';
const MiB = 1024 * 1024;
const references = [];
const slots = { a: null, b: null };
const deleting = new Set();
let workflow;
let history = [];
let historyRequest = 0;
let busy = false;
let activeJob = null;
const jobStorageKey = 'rp-image-center-active-job';

async function request(path, options) {
    let response;
    try { response = await fetch(path, { cache: 'no-store', signal: AbortSignal.timeout(options?.method === 'POST' ? 30000 : 6000), ...options }); }
    catch (error) { throw new Error('无法连接本机生图中心。请确认 start.cmd 已启动服务。', { cause: error }); }
    const data = await response.json().catch(() => null);
    if (!response.ok) {
        const detail = data?.error?.message || data?.detail;
        const message = response.status === 429 ? '生图队列已满，请等当前任务完成后重试。'
            : typeof detail === 'string' ? detail : `请求失败（HTTP ${response.status}）`;
        throw Object.assign(new Error(message), { status: response.status });
    }
    if (!data) throw new Error('本机生图中心返回的数据无效。');
    return data;
}

function showMessage(element, message) {
    element.textContent = message;
    element.hidden = !message;
}

function make(tag, text, className) {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    if (className) element.className = className;
    return element;
}

function syncControls() {
    if (!workflow) return;
    const editing = references.length > 0;
    const branch = editing ? workflow.edit : workflow.t2i;
    $('workflow-label').textContent = editing ? '参考编辑' : '文生图';
    $('sampling-label').textContent = `${branch.steps} 步 · ${workflow.sampler} · CFG ${workflow.cfg}`;
    $('reference-count').textContent = `${references.length} / ${workflow.max_references}`;
    $('character-control').hidden = !Object.hasOwn(workflow.character_models, $('model').value);
    $('gh-strength').disabled = !editing;
    $('character-value').textContent = Number($('character-strength').value).toFixed(2);
    $('gh-value').textContent = Number($('gh-strength').value).toFixed(2);
    const custom = $('size').value === 'custom';
    $('custom-size').hidden = !custom;
    $('width').disabled = !custom;
    $('height').disabled = !custom;
    $('submit').disabled = busy || !!activeJob;
    $('resume').hidden = busy || !activeJob;
    for (const button of $('history').querySelectorAll('[data-watch-job]')) button.disabled = busy || !!activeJob;
}

function renderReferences() {
    const list = $('reference-list');
    list.replaceChildren();
    references.forEach((ref, index) => {
        const item = make('li');
        const preview = make('img');
        preview.src = ref.url;
        preview.alt = '';
        const details = make('div', '', 'file-detail');
        const name = make('span', `第${index + 1}张 · ${ref.file.name}`);
        name.title = ref.file.name;
        const select = make('select');
        select.setAttribute('aria-label', `第${index + 1}张用途`);
        for (const [value, label] of Object.entries(roles)) select.append(new Option(label, value));
        select.value = ref.role;
        select.addEventListener('change', () => { ref.role = select.value; });
        const remove = make('button', '移除', 'text-button');
        remove.type = 'button';
        remove.setAttribute('aria-label', `移除第${index + 1}张`);
        remove.addEventListener('click', () => {
            references.splice(index, 1);
            URL.revokeObjectURL(ref.url);
            showMessage($('file-error'), '');
            renderReferences();
            $('files').focus();
        });
        details.append(name, select);
        item.append(preview, details, remove);
        list.append(item);
    });
    syncControls();
}

function addReferences(files) {
    if (!workflow) return;
    const errors = [];
    let total = references.reduce((bytes, ref) => bytes + ref.file.size, 0);
    for (const file of files) {
        if (references.length >= workflow.max_references) {
            errors.push(`最多 ${workflow.max_references} 张参考图。`);
            break;
        }
        if (!/\.(?:png|jpe?g|webp)$/i.test(file.name) && !/^image\/(?:png|jpeg|webp)$/i.test(file.type)) {
            errors.push(`${file.name} 请选择 PNG、JPEG 或 WebP。`);
            continue;
        }
        if (file.size > 6 * MiB || total + file.size > 24 * MiB) {
            errors.push(`${file.name} 超出限制，单张最多 6 MiB，合计最多 24 MiB。`);
            continue;
        }
        references.push({ file, role: 'reference', url: URL.createObjectURL(file) });
        total += file.size;
    }
    showMessage($('file-error'), errors.join(' '));
    renderReferences();
}

function resultLabel(job) {
    const size = `${job.width}×${job.height}`;
    const renderSize = job.render_size?.join('×');
    const loras = job.loras?.length ? job.loras.map(lora => {
        const character = workflow?.character_models[job.model];
        const name = lora.name === character?.lora.name ? 'SQX' : 'GH';
        return `${name} ${lora.strength.toFixed(2)}`;
    }) : [];
    if (!job.loras && workflow?.character_models[job.model]) loras.push('SQX');
    const mode = job.mode === 'edit' ? '参考编辑' : '文生图';
    return [size, renderSize && renderSize !== size ? `低显存渲染 ${renderSize}` : '', mode, ...loras].filter(Boolean).join(' · ');
}

function centerPixels(viewport) {
    viewport.scrollLeft = (viewport.scrollWidth - viewport.clientWidth) / 2;
}

function deleteButton(job) {
    const button = make('button', '🗑︎', 'delete-result');
    button.type = 'button';
    button.dataset.deleteJob = job.id;
    button.title = '删除图片';
    button.setAttribute('aria-label', job.seed >= 0 ? `删除种子 ${job.seed} 的图片` : '删除图片');
    button.disabled = deleting.has(job.id);
    button.addEventListener('click', () => void deleteResult(job));
    return button;
}

function renderSlot(slot) {
    const job = slots[slot];
    const pane = $(`pane-${slot}`);
    const label = slot.toUpperCase();
    const header = make('div', '', 'pane-header');
    const info = make('span', job ? resultLabel(job) : '待选图片');
    info.title = info.textContent;
    header.append(make('span', label, 'pane-badge'), info);
    const viewport = make('div', '', 'pane-viewport');
    const picture = make('div', '', 'result-picture');
    picture.append(viewport);
    if (job) {
        const download = make('a', '下载 PNG');
        download.href = job.image_url;
        download.download = `${job.id}.png`;
        header.append(download);
        const image = make('img');
        image.src = job.image_url;
        image.alt = `结果 ${label}，${resultLabel(job)}`;
        image.addEventListener('error', () => viewport.replaceChildren(make('p', '图片暂时无法读取', 'pane-empty')));
        image.addEventListener('load', () => {
            if ($('stage').classList.contains('actual')) centerPixels(viewport);
        });
        viewport.tabIndex = 0;
        viewport.setAttribute('aria-label', `结果 ${label} 的图片预览`);
        viewport.append(image);
        picture.append(deleteButton(job));
    } else {
        viewport.append(make('p', `从下方选择结果放入 ${label}`, 'pane-empty'));
    }
    pane.replaceChildren(header, picture);
}

function selectResult(slot, job) {
    slots[slot] = job;
    renderSlot(slot);
    for (const button of $('history').querySelectorAll(`button[data-slot="${slot}"]`)) {
        button.setAttribute('aria-pressed', String(button.dataset.job === job.id));
    }
}

function renderHistory() {
    const list = $('history');
    list.replaceChildren();
    $('history-status').textContent = history.length ? '' : '暂无任务记录。';
    for (const job of history) {
        const item = make('li');
        const details = make('div', '', 'result-details');
        if (job.status !== 'completed') {
            item.classList.add('history-status-only');
            const label = { failed: '生成失败', cancelled: '已取消', interrupted: '生成中断' }[job.status]
                || phases[job.phase] || '生成中';
            details.append(make('p', `${label} · ${resultLabel(job)}`));
            if (job.error) details.append(make('p', job.error, 'result-meta'));
            if (!terminal.has(job.status)) {
                const open = make('button', '查看任务', 'secondary');
                open.type = 'button';
                open.dataset.watchJob = job.id;
                open.disabled = busy || !!activeJob;
                open.addEventListener('click', () => {
                    if (busy || activeJob) return;
                    rememberJob(job.id);
                    void resumeJob();
                });
                details.append(open);
            }
            item.append(details);
            list.append(item);
            continue;
        }
        const thumbnail = make('img');
        thumbnail.loading = 'lazy';
        thumbnail.src = job.image_url;
        thumbnail.alt = '';
        details.append(make('p', resultLabel(job), 'result-summary'), make('p', job.seed >= 0 ? `种子 ${job.seed}` : '自动种子', 'result-meta'));
        for (const slot of ['a', 'b']) {
            const button = make('button', `放入 ${slot.toUpperCase()}`, 'secondary');
            button.type = 'button';
            button.dataset.slot = slot;
            button.dataset.job = job.id;
            button.setAttribute('aria-pressed', String(slots[slot]?.id === job.id));
            button.addEventListener('click', () => selectResult(slot, job));
            details.append(button);
        }
        if (job.seed >= 0) {
            const useSeed = make('button', '用此种子', 'text-button');
            useSeed.type = 'button';
            useSeed.addEventListener('click', () => { $('seed').value = job.seed; $('seed').focus(); });
            details.append(useSeed);
        }
        item.append(thumbnail, details, deleteButton(job));
        list.append(item);
    }
}

async function deleteResult(job) {
    if (deleting.has(job.id) || !window.confirm('永久删除这张生成图片？')) return;
    deleting.add(job.id);
    showMessage($('history-error'), '');
    const setDisabled = disabled => {
        for (const button of document.querySelectorAll(`[data-delete-job="${job.id}"]`)) button.disabled = disabled;
    };
    setDisabled(true);
    try {
        await request(job.image_url, { method: 'DELETE' });
        history = history.filter(record => record.id !== job.id);
        for (const slot of ['a', 'b']) {
            if (slots[slot]?.id !== job.id) continue;
            slots[slot] = null;
            renderSlot(slot);
        }
        renderHistory();
        await refreshHistory();
        $('refresh').focus();
    } catch (error) {
        showMessage($('history-error'), `删除失败。${error.message}`);
    } finally {
        deleting.delete(job.id);
        setDisabled(false);
    }
}

async function refreshHistory() {
    const currentRequest = ++historyRequest;
    $('refresh').disabled = true;
    try {
        const data = await request('/v1/jobs');
        if (currentRequest !== historyRequest) return;
        history = data.data.slice(0, 12);
        const latestImage = history.find(job => job.status === 'completed');
        if (!slots.a && latestImage) selectResult('a', latestImage);
        renderHistory();
    } catch (error) {
        if (currentRequest === historyRequest) $('history-status').textContent = `读取记录失败。${error.message}`;
    } finally {
        if (currentRequest === historyRequest) $('refresh').disabled = false;
    }
}

function randomSeed() {
    $('seed').value = crypto.getRandomValues(new Uint32Array(1))[0];
}

function buildPayload() {
    let size = $('size').value;
    if (size === 'custom') {
        const dimensions = ['width', 'height'].map(id => Number($(id).value));
        if (dimensions.some(value => value < 256 || value % workflow.alignment !== 0)) {
            throw new Error(`宽和高至少为 256，且须为 ${workflow.alignment} 的倍数。`);
        }
        size = dimensions.join('x');
    }
    if (size !== 'auto') {
        const [width, height] = size.split('x').map(Number);
        if (width * height > workflow.max_output_pixels) {
            throw new Error(`当前服务允许的最大画幅为 ${workflow.max_output_pixels.toLocaleString()} 像素。`);
        }
    }
    const model = $('model').value;
    const prompt = $('prompt').value.trim();
    if (!prompt) throw new Error('请输入提示词。');
    const data = { model, size, prompt, seed: $('seed').value === '' ? -1 : Number($('seed').value) };
    if (Object.hasOwn(workflow.character_models, model)) data.character_strength = Number($('character-strength').value);
    if (!references.length) return { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) };
    const body = new FormData();
    for (const [key, value] of Object.entries(data)) body.set(key, value);
    body.set('consistency_strength', $('gh-strength').value);
    body.set('reference_roles', JSON.stringify(references.map(ref => ref.role)));
    for (const ref of references) body.append('image[]', ref.file);
    return { body };
}

function updateJob(job) {
    const active = !terminal.has(job.status);
    const progress = job.progress;
    $('progress').hidden = !active;
    $('cancel').hidden = !active;
    $('cancel').disabled = job.status === 'cancelling';
    if (active) {
        const stepped = progress?.unit === 'steps' && progress.total > 0;
        if (stepped) {
            $('progress').max = progress.total;
            $('progress').value = progress.completed ?? 0;
        } else $('progress').removeAttribute('value');
        const stage = phases[job.phase] || '生成中';
        const retry = job.attempt > 1 && job.phase !== 'retrying_memory' ? '低显存模式 · ' : '';
        $('job-status').textContent = retry + stage + (stepped ? ` ${progress.completed ?? 0} / ${progress.total}` : '');
        return;
    }
    const status = { completed: '图片已生成', cancelled: '已取消', failed: '生成失败', interrupted: '生成中断' };
    $('job-status').textContent = job.status === 'completed' && job.attempt > 1 ? '低显存模式已生成图片' : status[job.status];
    const error = /CUDA out of memory/i.test(job.error || '')
        ? 'GPU 显存不足。可以缩小画幅，或减少参考图后重试。'
        : job.error;
    showMessage($('job-error'), error || '');
}

async function watchJob(job) {
    let failures = 0;
    while (true) {
        updateJob(job);
        if (terminal.has(job.status)) return job;
        await new Promise(resolve => setTimeout(resolve, failures ? 2000 : 1000));
        try {
            job = await request(`/v1/jobs/${job.id}`);
            failures = 0;
            showMessage($('job-error'), '');
        } catch (error) {
            if (error.status === 404 || ++failures >= 4) throw error;
            showMessage($('job-error'), `暂时无法读取进度，正在重试。${error.message}`);
        }
    }
}

function rememberJob(id) {
    activeJob = id;
    try {
        if (id) localStorage.setItem(jobStorageKey, id);
        else localStorage.removeItem(jobStorageKey);
    } catch (_) { /* The browser may block storage. */ }
}

async function runJob(getJob) {
    showMessage($('job-error'), '');
    busy = true;
    syncControls();
    $('job-status').textContent = activeJob ? '正在恢复任务' : '正在提交';
    try {
        const submitted = await getJob();
        rememberJob(submitted.id);
        const job = await watchJob(submitted);
        rememberJob(null);
        if (job.status === 'completed') selectResult(slots.a ? 'b' : 'a', job);
        await refreshHistory();
    } catch (error) {
        if (error.status === 404) rememberJob(null);
        $('job-status').textContent = activeJob ? '任务状态暂时未知' : '';
        $('progress').hidden = true;
        $('cancel').hidden = !activeJob;
        showMessage($('job-error'), error.message);
    } finally {
        busy = false;
        syncControls();
    }
}

function resumeJob() {
    if (activeJob && !busy) return runJob(() => request(`/v1/jobs/${activeJob}`));
}

function generate(event) {
    event.preventDefault();
    if (!workflow || busy || activeJob) return;
    return runJob(() => request('/v1/jobs', { method: 'POST', ...buildPayload() }));
}

async function cancelJob() {
    if (!activeJob) return;
    $('cancel').disabled = true;
    try {
        const job = await request(`/v1/jobs/${activeJob}/cancel`, { method: 'POST' });
        if (activeJob === job.id) updateJob(job);
    } catch (error) {
        $('cancel').disabled = false;
        showMessage($('job-error'), error.message);
    }
}

function setZoom(actual) {
    $('stage').classList.toggle('actual', actual);
    $('actual').setAttribute('aria-pressed', String(actual));
    $('fit').setAttribute('aria-pressed', String(!actual));
    if (actual) for (const viewport of $('stage').querySelectorAll('.pane-viewport')) centerPixels(viewport);
}

async function initialize() {
    renderSlot('a');
    renderSlot('b');
    $('files').disabled = true;
    try {
        const [description, health] = await Promise.all([request('/v1/workflow'), request('/health')]);
        workflow = description;
        $('model').append(new Option('Qwen Image 2.1 Turbo', workflow.model));
        for (const [id, character] of Object.entries(workflow.character_models)) {
            const label = id.endsWith('-sqx') ? 'Qwen Turbo + 苏祈雪 LoRA' : id;
            $('model').append(new Option(label, id));
            $('model').value = id;
            $('character-strength').value = character.strength;
        }
        $('gh-strength').value = workflow.edit.lora_strength;
        const backend = (health.backend || workflow.runtime.backend).toUpperCase();
        $('connection').textContent = `本机服务已连接 · ${backend} · 显存预算 ${workflow.runtime.max_vram_gib} GiB`;
        $('connection').title = `保留系统内存 ${workflow.runtime.reserve_ram_gib} GiB · CPU ${workflow.runtime.cpu_threads} 线程`;
        $('connection').dataset.ready = 'true';
        $('files').disabled = false;
        syncControls();
        let saved;
        try { saved = localStorage.getItem(jobStorageKey); } catch (_) { /* The browser may block storage. */ }
        if (/^img_[a-f0-9]{32}$/.test(saved || '')) {
            activeJob = saved;
            void resumeJob();
        }
        await refreshHistory();
    } catch (error) {
        $('connection').textContent = '本机服务连接失败';
        $('history-status').textContent = '刷新页面可重试。';
        showMessage($('job-error'), error.message);
    }
}

$('generator').addEventListener('submit', generate);
$('resume').addEventListener('click', resumeJob);
$('cancel').addEventListener('click', cancelJob);
$('random-seed').addEventListener('click', randomSeed);
$('example').addEventListener('click', () => {
    if (!$('prompt').value.trim()) $('prompt').value = samplePrompt;
    else $('prompt').setRangeText(`\n${samplePrompt}`, $('prompt').value.length, $('prompt').value.length, 'end');
    $('prompt').focus();
});
for (const id of ['model', 'size', 'character-strength', 'gh-strength']) $(id).addEventListener('input', syncControls);
$('files').addEventListener('change', event => {
    addReferences(event.target.files);
    event.target.value = '';
});
$('dropzone').addEventListener('dragover', event => {
    event.preventDefault();
    $('dropzone').classList.add('drag-over');
});
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('drag-over'));
$('dropzone').addEventListener('drop', event => {
    event.preventDefault();
    $('dropzone').classList.remove('drag-over');
    addReferences(event.dataTransfer.files);
});
$('refresh').addEventListener('click', refreshHistory);
$('fit').addEventListener('click', () => setZoom(false));
$('actual').addEventListener('click', () => setZoom(true));
initialize();
