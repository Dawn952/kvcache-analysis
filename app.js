'use strict';

const { inputs, defaultBlock, mlaKernelBlock, mainLayers, stateGroups, layersPerStateGroup, draftLayers, calculate } = KVCache;
const byId = (id) => document.getElementById(id);
const scenario = byId('scenario');
const recover = byId('recover');
const draft = byId('draft');
const spec = byId('spec');
const blockSize = byId('block-size');
const results = byId('results');
const layout = byId('layout');
const detail = byId('detail');
let selected = 3;
let layoutColumn = 0;
let showMiddleSlots = false;
let showBlockZoom = false;
let comparisons;

const fmt = new Intl.NumberFormat('zh-CN');
const memory = (bytes) => bytes >= 2 ** 30
  ? `${(bytes / 2 ** 30).toFixed(3)} GiB`
  : `${(bytes / 2 ** 20).toFixed(2)} MiB`;
const lengths = ['4k+1k', '16k+1k', '64k+1k', '128k+1k', '1M'];
const topology = (dcp) => dcp === 8 ? 'DP4 / DCP8 / TP8' : 'DP4 / TP8';

function describe({ recover: enabled, spec: count }) {
  return `RecoverSSM ${enabled ? '开' : '关'} · DSpark ${count ? `${count} token` : '关'}`;
}

function geometryEvidence(value, dcp) {
  return dcp === 8 && value.localBlock === defaultBlock ? '页几何有历史日志' : '源码推算 · 未实测';
}

function render() {
  layoutColumn = 0;
  spec.disabled = !draft.checked;
  const current = { dcp: Number(scenario.value), recover: recover.checked,
    spec: draft.checked ? Number(spec.value) : 0, requestedBlock: Number(blockSize.value) };
  const geometry = calculate(inputs[0], current);
  byId('block-size-summary').innerHTML = `<span>传入 <strong>${geometry.requestedBlock}</strong> → 生效 <strong>${geometry.localBlock}</strong> token / 卡</span><small>${geometryEvidence(geometry, current.dcp)}</small>`;
  const alternateDcp = current.dcp === 8 ? 1 : 8;
  comparisons = [current, { ...current, recover: !current.recover }, { ...current, spec: 0 },
    { ...current, spec: 3 }, { ...current, dcp: alternateDcp }];
  const alternateLabel = alternateDcp === 1 ? 'DP4 / TP8（推算）' : topology(alternateDcp);
  const headings = ['当前配置', `RecoverSSM ${current.recover ? '关' : '开'}`, 'DSpark 关闭', 'DSpark 3 token', alternateLabel];
  document.querySelectorAll('.config-heading').forEach((heading, column) => {
    const { bytesPerId } = calculate(inputs[0], comparisons[column]);
    heading.innerHTML = `${headings[column]}<small>${memory(bytesPerId)} / block</small>`;
  });
  results.innerHTML = inputs.map((input, index) => {
    const cells = comparisons.map((config, column) => {
      const value = calculate(input, config);
      return `<td><button type="button" class="value" data-row="${index}" data-column="${column}" aria-label="${lengths[index]}，${topology(config.dcp)}，${describe(config)}，${memory(value.bytes)}，查看明细">${memory(value.bytes)}<small>${value.historyIds} MLA + ${value.stateIds} KDA</small></button></td>`;
    }).join('');
    return `<tr><th scope="row"><button type="button" class="length" data-row="${index}" aria-pressed="${index === selected}">${lengths[index]}</button></th>${cells}</tr>`;
  }).join('');
  renderLayout();
}

function renderLayout() {
  const current = comparisons[layoutColumn];
  const value = calculate(inputs[selected], current);
  const groups = Array.from({ length: stateGroups }, (_, index) => index);
  const { pages, regions } = value;
  const blockMarks = (count) => `<span class="block-marks" aria-hidden="true">${'<i></i>'.repeat(Math.min(count, 12))}${count > 12 ? '<b>…</b>' : ''}</span>`;
  const mainSlots = Array.from({ length: mainLayers }, (_, slot) => {
    const empty = slot === layersPerStateGroup;
    const targetLayer = empty ? 92 : slot * 4 + 3;
    const collapse = slot === 3
      ? '<button id="collapse-middle-slots" type="button" class="slot-collapse" data-layout-panel="middle-slots" aria-controls="middle-slots" aria-expanded="true" aria-label="收起中间 20 个槽位">收起</button>'
      : '';
    return `<tr><th scope="row">L 槽 ${slot}${collapse}</th><td class="mla">MLA ${targetLayer}</td>${groups.map((group) => empty
      ? '<td class="empty">空槽</td>' : `<td class="kda">KDA ${slot * 4 + group}</td>`).join('')}</tr>`;
  });
  const sideRow = (name, layers, kind, text) => `<tr><th scope="row">${name} 区 × ${layers} 层</th><td class="${kind}">${text}</td>${groups.map(() => '<td class="empty">预留空页</td>').join('')}</tr>`;
  byId('layout-caption').textContent = `${lengths[selected]} · ${topology(current.dcp)} · ${describe(current)} · block ${value.requestedBlock} → ${value.localBlock} · ${geometryEvidence(value, current.dcp)}`;
  layout.innerHTML = `
    <div class="flow">
      <div><small>MLA 历史 blocks</small><strong>${value.historyIds} 个</strong><small>⌈(${fmt.format(value.limit)} + ${current.spec}) ÷ ${fmt.format(value.coverage)}⌉</small></div>
      <div><small>KDA 状态 blocks</small><strong>3 组 × ${value.stateCopies} 个</strong><small>每组 23 层，持有各自的 ID</small></div>
      <div><small>一个逻辑 block 的显存 / 卡</small><strong>${memory(value.bytesPerId)}</strong><small>${fmt.format(value.bytesPerId)} B</small></div>
      <div><small>请求合计 / 卡</small><strong>${memory(value.bytes)}</strong><small>(${value.historyIds} + ${value.stateIds}) × 单 block 显存</small></div>
    </div>
    <div class="axis-key"><code>L → B → N → H → C</code><span>物理层槽 → 池 block ID → 本地 token → KV head → 通道</span></div>
    <p class="caption">沿一行看同一物理层槽里的 B 轴。每列展开为所示数量的互斥池 ID。单元格标注模型层号。</p>
    <div class="scroll"><table class="pool-map">
      <caption>Main 区的 MLA / KDA 叠放，以及独立的 Indexer / Draft 区</caption>
      <thead><tr><th scope="col">L 轴 · 物理层槽</th><th scope="col">MLA 历史 · ${value.historyIds} ID${blockMarks(value.historyIds)}<button type="button" class="layout-toggle" data-layout-panel="block-zoom" aria-controls="block-zoom" aria-expanded="${showBlockZoom}">${showBlockZoom ? '收起 B 轴细节' : '放大 B 轴 block'}</button></th>${groups.map((group) => `<th scope="col">KDA ${group + 1} 组 · ${value.stateCopies} ID${blockMarks(value.stateCopies)}</th>`).join('')}</tr></thead>
      <tbody id="block-zoom" ${showBlockZoom ? '' : 'hidden'}><tr><td colspan="5" class="block-zoom-cell">${renderBlockZoom(value, current.dcp)}</td></tr></tbody>
      <tbody>${mainSlots.slice(0, 3).join('')}<tr id="middle-slots-placeholder" ${showMiddleSlots ? 'hidden' : ''}><th scope="row">L 槽 3–22</th><td colspan="4" class="empty"><button id="expand-middle-slots" type="button" class="layout-toggle" data-layout-panel="middle-slots" aria-controls="middle-slots" aria-expanded="false">展开中间 20 个槽位</button></td></tr></tbody>
      <tbody id="middle-slots" ${showMiddleSlots ? '' : 'hidden'}>${mainSlots.slice(3, -1).join('')}</tbody>
      <tbody>${mainSlots.at(-1)}</tbody>
      <tbody>${sideRow('Indexer', mainLayers, 'indexer', 'FP8 K + scale')}${current.spec ? sideRow('Draft', draftLayers, 'draft', 'BF16 MLA KV') : ''}</tbody>
    </table></div>
    <h3>一个池 block 的显存怎么算</h3>
    <div class="scroll"><table class="pool-map" aria-label="单池 block 显存分项"><thead><tr><th scope="col">区域</th><th scope="col">物理页数 × 每页字节</th><th scope="col">每个池 ID</th></tr></thead><tbody>
      <tr><th scope="row">Main</th><td>${mainLayers} × ${fmt.format(pages.main)} B</td><td>${memory(regions.main)}</td></tr>
      <tr><th scope="row">Indexer</th><td>${mainLayers} × ${fmt.format(pages.index)} B</td><td>${memory(regions.index)}</td></tr>
      <tr><th scope="row">Draft</th><td>${current.spec ? draftLayers : 0} × ${fmt.format(pages.draft)} B</td><td>${memory(regions.draft)}</td></tr>
    </tbody></table></div>
    <p class="caption">上下文变长时，MLA 历史块数按逻辑覆盖取整。KDA 每组采用固定块数预算，按 2 个基础块的规划上界，RecoverSSM 关闭时再加 numspec 个。关闭 DSpark 会去掉每个池 ID 的 Draft 区。DCP 切换改变历史 ID 的 token 覆盖。</p>`;
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
    <p class="caption">先选块：${requestedBlock} × ⌈${fmt.format(pages.state)} B ÷ ${fmt.format(upstreamQuantumBytes)} B⌉ = ${requestedBlock} × ${alignmentMultiple} = ${localBlock} token。再按打包后的 MLA 页计算字节宽度。</p>
    <div class="byte-scale" aria-label="单卡物理页内字节偏移"><span>0 B</span><span>${fmt.format(pages.main)} B</span></div>
    <div class="byte-label"><strong>MLA 物理页</strong><span>${tiles} 小块 × ${mlaKernelBlock} token，每小块 ${fmt.format(tileBytes)} B</span></div>
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
    button.textContent = expanded ? '收起 B 轴细节' : '放大 B 轴 block';
  }
});

function openDetail(row, column) {
  const config = comparisons[column];
  const value = calculate(inputs[row], config);
  byId('detail-title').textContent = `${memory(value.bytes)} / 卡`;
  byId('detail-caption').textContent = `${lengths[row]} · ${topology(config.dcp)} · ${describe(config)} · block ${value.requestedBlock} → ${value.localBlock} · ${geometryEvidence(value, config.dcp)}`;
  byId('detail-rows').innerHTML = `<p class="caption">历史 ID = ⌈(${fmt.format(value.limit)} + ${config.spec}) / ${fmt.format(value.coverage)}⌉ = ${value.historyIds}。KDA = 3 组 × ${value.stateCopies} 份 = ${value.stateIds} ID。每个 ID 占 ${memory(value.bytesPerId)} / 卡。</p>${value.rows.map((entry) => `<details class="breakdown"><summary><span>${entry.name}</span><strong>${memory(entry.bytes)}</strong></summary><p>${entry.shape}</p><p><code>${entry.formula} = ${fmt.format(entry.bytes)} B</code></p><p class="tags">${entry.tags.map((tag) => `<span>${tag}</span>`).join('')}</p></details>`).join('')}`;
  detail.showModal();
}

results.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-row]');
  if (!button) return;
  selected = Number(button.dataset.row);
  layoutColumn = Number(button.dataset.column ?? 0);
  for (const lengthButton of results.querySelectorAll('.length')) {
    lengthButton.setAttribute('aria-pressed', Number(lengthButton.dataset.row) === selected);
  }
  renderLayout();
  if (button.dataset.column !== undefined) openDetail(selected, Number(button.dataset.column));
});

for (const control of [scenario, recover, draft, spec, blockSize]) control.addEventListener('change', render);
byId('reset').addEventListener('click', () => {
  scenario.value = '8';
  recover.checked = true;
  draft.checked = true;
  spec.value = '3';
  blockSize.value = String(defaultBlock);
  selected = 3;
  render();
});

render();
