'use strict';

const { inputs, defaultBlock, platforms, mlaKernelBlock, mainLayers, stateGroups, layersPerStateGroup, draftLayers, calculate } = KVCache;
const byId = (id) => document.getElementById(id);
const hardware = byId('hardware');
const weight = byId('weight');
const scenario = byId('scenario');
const recover = byId('recover');
const draft = byId('draft');
const spec = byId('spec');
const mlaDtype = byId('mla-dtype');
const indexDtype = byId('index-dtype');
const draftDtype = byId('draft-dtype');
const blockSize = byId('block-size');
const results = byId('results');
const layout = byId('layout');
const detail = byId('detail');
let selected = 3;
let showMiddleSlots = false;
let showBlockZoom = false;
let current;

const fmt = new Intl.NumberFormat('zh-CN');
const memory = (bytes) => bytes >= 2 ** 30
  ? `${(bytes / 2 ** 30).toFixed(3)} GiB`
  : `${(bytes / 2 ** 20).toFixed(2)} MiB`;
const lengths = ['4k+1k', '16k+1k', '64k+1k', '128k+1k', '1M'];
const topology = ({ platform, dcp }) => `DP4${dcp > 1 ? ` / DCP${dcp}` : ''} / TP${platforms[platform].tp}`;
const weightLabel = (value) => value === 'native' ? 'Moonshot 原生 MXFP4' : value;
const blockInputPrefix = ({ requestedBlock, localBlock }) => requestedBlock === localBlock ? '' : `${requestedBlock} → `;

function describe({ recover: enabled, spec: count }) {
  return `RecoverSSM ${enabled ? '开' : '关'} · DSpark ${count ? `${count} token` : '关'}`;
}

function geometryEvidence(value, config) {
  if (config.weight === 'native') return '条件推算 · 原生权重加载未验证';
  if (hasAlternateDtype(config)) return 'dtype 对照 · 部署未验证';
  if (config.platform === 'a3') return '模板与源码复核 · 占用推算';
  return config.dcp === 8 && value.localBlock === defaultBlock ? '页几何有启动日志 · 占用推算' : '源码推算 · 未实测';
}

function hasAlternateDtype(config) {
  const baseline = platforms[config.platform].cacheDtype;
  return config.mlaDtype !== baseline || config.indexDtype !== baseline || (config.spec > 0 && config.draftDtype !== 'BF16');
}

function selectPlatform() {
  const platform = hardware.value;
  const preset = platforms[platform];
  const dcpOptions = platform === 'a5' ? [preset.dcp, 1] : [preset.dcp];
  weight.replaceChildren(...preset.weights.map(value => new Option(weightLabel(value), value)));
  scenario.replaceChildren(...dcpOptions.map(dcp => new Option(topology({ platform, dcp }), dcp)));
  blockSize.replaceChildren(...preset.blockOptions.map(size => new Option(size, size)));
  blockSize.value = String(preset.defaultBlock);
  mlaDtype.replaceChildren(...preset.dsaDtypes.map(value => new Option(value, value)));
  indexDtype.replaceChildren(...preset.dsaDtypes.map(value => new Option(value, value)));
  draftDtype.replaceChildren(...preset.draftDtypes.map(value => new Option(value === 'BF16' ? value : `${value}（容量估算）`, value)));
  recover.checked = preset.recover;
  draft.checked = true;
  spec.value = '3';
  render();
}

function render() {
  spec.disabled = !draft.checked;
  draftDtype.disabled = !draft.checked;
  recover.disabled = !draft.checked;
  if (recover.disabled) recover.checked = false;
  current = { platform: hardware.value, weight: weight.value, dcp: Number(scenario.value), recover: recover.checked,
    spec: draft.checked ? Number(spec.value) : 0, requestedBlock: Number(blockSize.value),
    mlaDtype: mlaDtype.value, indexDtype: indexDtype.value, draftDtype: draftDtype.value };
  byId('native-weight-note').hidden = current.weight !== 'native';
  byId('dtype-note').hidden = !hasAlternateDtype(current);
  const headings = lengths.map((length, index) =>
    `<th scope="col"><button type="button" class="length" data-row="${index}">${length}</button></th>`).join('');
  const values = inputs.map((input, index) => {
    const value = memory(calculate(input, current).bytes);
    return `<td><button type="button" class="value" data-row="${index}" aria-label="${lengths[index]}，${value}，查看明细">${value}</button></td>`;
  }).join('');
  results.tHead.rows[0].innerHTML = `<th scope="col">max-model-len</th>${headings}`;
  results.tBodies[0].innerHTML = `<tr><th scope="row">估算显存</th>${values}</tr>`;
  updateSelection();
}

function updateSelection() {
  for (const button of results.querySelectorAll('button[data-row]')) {
    const active = Number(button.dataset.row) === selected;
    button.parentElement.classList.toggle('selected-length', active);
    if (button.classList.contains('length')) button.setAttribute('aria-pressed', active);
  }
  renderLayout();
}

function renderLayout() {
  const value = calculate(inputs[selected], current);
  const groups = Array.from({ length: stateGroups }, (_, index) => index);
  const { pages, regions } = value;
  const blockMarks = (count) => `<span class="block-marks" aria-hidden="true">${'<i></i>'.repeat(Math.min(count, 12))}${count > 12 ? '<b>…</b>' : ''}</span>`;
  const mainSlots = Array.from({ length: mainLayers }, (_, slot) => {
    const empty = slot === layersPerStateGroup;
    const targetLayer = empty ? 92 : slot * 4 + 3;
    const collapse = slot === 3
      ? '<button id="collapse-middle-slots" type="button" class="slot-collapse" data-layout-panel="middle-slots" aria-controls="middle-slots" aria-expanded="true" aria-label="收起第 3–22 行">收起</button>'
      : '';
    return `<tr><th scope="row">第 ${slot} 行${collapse}</th><td class="mla">MLA 第 ${targetLayer} 层</td>${groups.map((group) => empty
      ? '<td class="empty">空槽</td>' : `<td class="kda">KDA 第 ${slot * 4 + group} 层</td>`).join('')}</tr>`;
  });
  const sideRow = (name, layers, kind, text) => `<tr><th scope="row">${name} 区 × ${layers} 层</th><td class="${kind}">${text}</td>${groups.map(() => '<td class="empty">预留空页</td>').join('')}</tr>`;
  byId('layout-caption').innerHTML = `${lengths[selected]} · ${topology(current)} · ${weightLabel(current.weight)} · ${describe(current)} · block ${blockInputPrefix(value)}<strong>${value.localBlock} token / 卡</strong> · ${geometryEvidence(value, current)}`;
  layout.innerHTML = `
    <div class="flow">
      <div><small>MLA 历史 blocks</small><strong>${value.historyIds} 个</strong><small>⌈${fmt.format(value.limit)} ÷ ${fmt.format(value.coverage)}⌉</small></div>
      <div><small>KDA 常驻状态块</small><strong>3 组 × ${value.stateCopies} 块</strong><small>每组 23 层，持有各自的 ID</small></div>
      <div><small>一个逻辑 block 的显存 / 卡</small><strong>${memory(value.bytesPerId)}</strong><small>${fmt.format(value.bytesPerId)} B</small></div>
      <div><small>请求合计 / 卡</small><strong>${memory(value.bytes)}</strong><small>(${value.historyIds} + ${value.stateIds}) × 单 block 显存</small></div>
    </div>
    <div class="axis-key" id="layout-reading-guide">
      <div><strong>纵向看存储行（L）。</strong> 例如第 0 行存放模型第 3 层的 MLA，以及第 0、1、2 层的 KDA。</div>
      <div><strong>横向看缓存块（B）。</strong> B 是缓存池中的块编号。同一组的各层共用一批编号，列标题显示这一组占用多少块。</div>
    </div>
    <div class="scroll"><table class="pool-map" aria-describedby="layout-reading-guide">
      <caption>Main 区的 MLA / KDA 叠放，以及独立的 Indexer / Draft 区</caption>
      <thead><tr><th scope="col">存储行（L）</th><th scope="col">MLA 历史 · ${value.historyIds} 块${blockMarks(value.historyIds)}<button type="button" class="layout-toggle" data-layout-panel="block-zoom" aria-controls="block-zoom" aria-expanded="${showBlockZoom}">${showBlockZoom ? '收起单块细节' : '查看单块内部'}</button></th>${groups.map((group) => `<th scope="col">KDA ${group + 1} 组 · ${value.stateCopies} 块${blockMarks(value.stateCopies)}</th>`).join('')}</tr></thead>
      <tbody id="block-zoom" ${showBlockZoom ? '' : 'hidden'}><tr><td colspan="5" class="block-zoom-cell">${renderBlockZoom(value, current.dcp)}</td></tr></tbody>
      <tbody>${mainSlots.slice(0, 3).join('')}<tr id="middle-slots-placeholder" ${showMiddleSlots ? 'hidden' : ''}><th scope="row">第 3–22 行</th><td colspan="4" class="empty"><button id="expand-middle-slots" type="button" class="layout-toggle" data-layout-panel="middle-slots" aria-controls="middle-slots" aria-expanded="false">展开第 3–22 行</button></td></tr></tbody>
      <tbody id="middle-slots" ${showMiddleSlots ? '' : 'hidden'}>${mainSlots.slice(3, -1).join('')}</tbody>
      <tbody>${mainSlots.at(-1)}</tbody>
      <tbody>${sideRow('Indexer', mainLayers, 'indexer', `${current.indexDtype} K${current.indexDtype === 'BF16' ? '' : ' + scale'}`)}${current.spec ? sideRow('Draft', draftLayers, 'draft', `${current.draftDtype} MLA KV`) : ''}</tbody>
    </table></div>
    <h3>一个池 block 的显存怎么算</h3>
    <div class="scroll"><table class="pool-map" aria-label="单池 block 显存分项"><thead><tr><th scope="col">区域</th><th scope="col">物理页数 × 每页字节</th><th scope="col">每个池 ID</th></tr></thead><tbody>
      <tr><th scope="row">Main</th><td>${mainLayers} × ${fmt.format(pages.main)} B</td><td>${memory(regions.main)}</td></tr>
      <tr><th scope="row">Indexer</th><td>${mainLayers} × ${fmt.format(pages.index)} B</td><td>${memory(regions.index)}</td></tr>
      <tr><th scope="row">Draft</th><td>${current.spec ? draftLayers : 0} × ${fmt.format(pages.draft)} B</td><td>${memory(regions.draft)}</td></tr>
    </tbody></table></div>
    <p class="caption">MLA 历史块数按逻辑覆盖取整，投机预留受本档 <code>max-model-len</code> 限制。KDA 每组常驻 1 个运行状态块，RecoverSSM 关闭时再计 <code>spec_num</code> 个投机状态块。</p>`;
}

function renderBlockZoom({ coverage, pages, localBlock, requestedBlock, upstreamQuantumBytes, alignmentMultiple }, dcp) {
  const tiles = localBlock / mlaKernelBlock;
  const tileBytes = pages.main / tiles;
  const mlaParts = Array.from({ length: tiles }, (_, index) => ({
    label: `Kernel 小块 ${index + 1}`, bytes: tileBytes, css: 'mla', text: index + 1,
  }));
  const kdaParts = [
    { label: 'Conv', bytes: pages.conv, css: 'conv' },
    { label: 'Recurrent', bytes: pages.recurrent, css: 'kda' },
    { label: 'Padding', bytes: pages.padding, css: 'empty' },
  ];
  const track = (name, parts) => {
    let offset = 0;
    const buttons = parts.map(({ label, bytes, css, text }) => {
      const start = offset;
      offset += bytes;
      const caption = `${label} · ${fmt.format(bytes)} B · 偏移 [${fmt.format(start)}, ${fmt.format(offset)}) B`;
      return `<button type="button" class="byte-segment ${css}" data-byte-caption="${caption}" aria-label="${name} ${caption}" title="${caption}">${text ?? ''}</button>`;
    });
    const columns = parts.map(({ bytes }) => `minmax(0, ${bytes}fr)`).join(' ');
    return `<div class="byte-track" role="group" aria-label="${name}" style="grid-template-columns:${columns}">${buttons.join('')}</div>`;
  };
  return `<div class="block-zoom">
    <div><strong>调度逻辑大块 · ${fmt.format(coverage)} token</strong><span>${dcp} 卡 × ${fmt.format(localBlock)} token / 卡</span></div>
    <p class="caption">先选块：${requestedBlock} × ⌈${fmt.format(pages.state)} B ÷ ${fmt.format(upstreamQuantumBytes)} B⌉ = ${requestedBlock} × ${alignmentMultiple} = ${localBlock} token。再按 ${current.mlaDtype} MLA 物理页计算字节宽度。</p>
    <div class="byte-scale" aria-label="单卡物理页内字节偏移"><span>0 B</span><span>${fmt.format(pages.main)} B</span></div>
    <div class="byte-label"><strong>MLA 物理页</strong><span>${current.mlaDtype} · ${tiles} 小块 × ${mlaKernelBlock} token，每小块 ${fmt.format(tileBytes)} B</span></div>
    ${track('MLA 物理页', mlaParts)}
    <div class="byte-label"><strong>KDA 对齐页</strong><span>${fmt.format(pages.state)} B 状态 + ${fmt.format(pages.padding)} B padding</span></div>
    ${track('KDA 对齐页', kdaParts)}
    <div class="byte-legend">${kdaParts.map(({ label, bytes, css }) => `<span><i class="${css}" aria-hidden="true"></i>${label} ${fmt.format(bytes)} B</span>`).join('')}</div>
    <p id="byte-readout" class="caption" aria-live="polite">点击任意分段查看页内字节偏移。</p>
    <p class="caption">两行使用同一字节标尺。KDA 对齐到单卡 MLA 的整页宽度 ${fmt.format(pages.main)} B。</p>
  </div>`;
}

layout.addEventListener('click', (event) => {
  const segment = event.target.closest('[data-byte-caption]');
  if (segment) {
    byId('byte-readout').textContent = segment.dataset.byteCaption;
    return;
  }
  const button = event.target.closest('[data-layout-panel]');
  if (!button) return;
  const panel = byId(button.dataset.layoutPanel);
  const expanded = panel.hidden;
  panel.hidden = !expanded;
  if (panel.id === 'middle-slots') {
    showMiddleSlots = expanded;
    byId('middle-slots-placeholder').hidden = expanded;
    byId(expanded ? 'collapse-middle-slots' : 'expand-middle-slots').focus();
  } else {
    showBlockZoom = expanded;
    button.setAttribute('aria-expanded', expanded);
    button.textContent = expanded ? '收起单块细节' : '查看单块内部';
  }
});

function openDetail(row) {
  const config = current;
  const value = calculate(inputs[row], config);
  byId('detail-title').textContent = `${memory(value.bytes)} / 卡`;
  byId('detail-caption').textContent = `${lengths[row]} · ${topology(config)} · ${weightLabel(config.weight)} · ${describe(config)} · block ${blockInputPrefix(value)}${value.localBlock} token / 卡 · ${geometryEvidence(value, config)}`;
  byId('detail-rows').innerHTML = `<p class="caption">历史 ID = ⌈${fmt.format(value.limit)} / ${fmt.format(value.coverage)}⌉ = ${value.historyIds}。投机预留受本档 <code>max-model-len</code> 截断。KDA 常驻 = 3 组 × ${value.stateCopies} 块 = ${value.stateIds} ID。每个 ID 占 ${memory(value.bytesPerId)} / 卡。</p>${value.rows.map((entry) => `<details class="breakdown"><summary><span>${entry.name}</span><strong>${memory(entry.bytes)}</strong></summary><p>${entry.shape}</p><p><code>${entry.formula} = ${fmt.format(entry.bytes)} B</code></p><p class="tags">${entry.tags.map((tag) => `<span>${tag}</span>`).join('')}</p></details>`).join('')}`;
  detail.showModal();
}

results.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-row]');
  if (!button) return;
  selected = Number(button.dataset.row);
  updateSelection();
  if (button.classList.contains('value')) openDetail(selected);
});

for (const control of [weight, scenario, recover, draft, spec, blockSize, mlaDtype, indexDtype, draftDtype]) control.addEventListener('change', render);
hardware.addEventListener('change', selectPlatform);
byId('reset').addEventListener('click', () => {
  hardware.value = 'a5';
  selected = 3;
  selectPlatform();
});

selectPlatform();
