'use strict';

const { inputs, Ki, localBlock, calculate } = KVCache;
const byId = (id) => document.getElementById(id);
const scenario = byId('scenario');
const recover = byId('recover');
const draft = byId('draft');
const spec = byId('spec');
const results = byId('results');
const detail = byId('detail');
let selected = 3;
let comparisons;

const fmt = new Intl.NumberFormat('zh-CN');
const memory = (bytes) => bytes >= 2 ** 30
  ? `${(bytes / 2 ** 30).toFixed(3)} GiB`
  : `${(bytes / 2 ** 20).toFixed(2)} MiB`;
const lengths = ['4k', '16k', '64k', '128k', '1M − 1k'];
const topology = (dcp) => dcp === 8 ? 'DP4 / DCP8 / TP8' : 'DP4 / TP8';

function describe({ recover: enabled, spec: count }) {
  return `RecoverSSM ${enabled ? '开' : '关'} · DSpark ${count ? `${count} token` : '关'}`;
}

function render() {
  spec.disabled = !draft.checked;
  const current = { dcp: Number(scenario.value), recover: recover.checked, spec: draft.checked ? Number(spec.value) : 0 };
  const alternateDcp = current.dcp === 8 ? 1 : 8;
  comparisons = [current, { ...current, recover: !current.recover }, { ...current, spec: 0 },
    { ...current, spec: 3 }, { ...current, dcp: alternateDcp }];
  byId('recover-heading').textContent = `RecoverSSM ${current.recover ? '关' : '开'}`;
  byId('topology-heading').textContent = topology(alternateDcp);
  results.innerHTML = inputs.map((input, index) => {
    const cells = comparisons.map((config, column) => {
      const value = calculate(input, config);
      return `<td><button type="button" class="value" data-row="${index}" data-column="${column}" aria-label="${lengths[index]}，${topology(config.dcp)}，${describe(config)}，${memory(value.bytes)}，查看明细">${memory(value.bytes)}</button></td>`;
    }).join('');
    return `<tr><th scope="row"><button type="button" class="length" data-row="${index}" aria-pressed="${index === selected}">${lengths[index]}</button></th><td>1k</td><td>${fmt.format(input + Ki)}</td>${cells}</tr>`;
  }).join('');
  byId('scenario-note').textContent = current.dcp === 8
    ? 'DCP8 使用 A5 Atlas 历史布局。DP4 / TP8 对照按 DCP=1 推算，待部署验证。'
    : 'DP4 / TP8 使用 DCP=1，沿用同一本地页几何作对照，待部署验证。';
  renderLayout();
}

function renderLayout() {
  const current = comparisons[0];
  const value = calculate(inputs[selected], current);
  byId('layout-caption').textContent = `${lengths[selected]} + 1k · ${describe(comparisons[0])} · ${memory(value.bytes)} / 卡`;
  byId('layout').innerHTML = `<div class="flow">
    <div><small>序列预算</small><strong>${fmt.format(value.limit)} token</strong></div>
    <div><small>逻辑 block</small><strong>${fmt.format(value.coverage)} token</strong></div>
    <div><small>单卡物理页</small><strong>${fmt.format(localBlock)} token</strong></div>
    <div><small>池 ID</small><strong>${value.historyIds} 历史 + ${value.stateIds} 状态</strong></div>
  </div><div class="scroll"><table class="pool-map"><caption>每个池 ID 的物理槽位</caption><thead><tr><th scope="col">ID 归属</th><th scope="col">Main · 24 槽</th><th scope="col">Indexer · 24 槽</th><th scope="col">Draft · ${current.spec ? 5 : 0} 槽</th></tr></thead><tbody>
    <tr><th scope="row">历史 ID × ${value.historyIds}</th><td>24 层 Attention KV</td><td>24 层 Indexer K</td><td>${current.spec ? '5 层 draft MLA KV' : '未分配'}</td></tr>
    <tr><th scope="row">状态 ID × ${value.stateIds}</th><td>23 层 KDA 对齐页 + 1 空槽</td><td>24 空槽</td><td>${current.spec ? '5 空槽' : '未分配'}</td></tr>
  </tbody></table></div><div class="diagram"><div class="bar" aria-hidden="true">${value.rows.map((row, index) => `<span class="c${index}" style="flex:${row.bytes}"></span>`).join('')}</div>
  <ul class="legend">${value.rows.map((row, index) => `<li><i class="c${index}" aria-hidden="true"></i>${row.name}<span>${memory(row.bytes)}</span></li>`).join('')}</ul></div>`;
}

function openDetail(row, column) {
  const config = comparisons[column];
  const value = calculate(inputs[row], config);
  byId('detail-title').textContent = `${memory(value.bytes)} / 卡`;
  byId('detail-caption').textContent = `${lengths[row]} + 1k · ${topology(config.dcp)} · ${describe(config)}`;
  byId('detail-rows').innerHTML = `<p class="caption">历史 ID = ⌈(${fmt.format(value.limit)} + ${config.spec}) / ${fmt.format(value.coverage)}⌉ = ${value.historyIds}。状态 ID = 3 组 × ${value.stateCopies} 份 = ${value.stateIds}。</p>${value.rows.map((entry) => `<details class="breakdown"><summary><span>${entry.name}</span><strong>${memory(entry.bytes)}</strong></summary><p>${entry.shape}</p><p><code>${entry.formula} = ${fmt.format(entry.bytes)} B</code></p><p class="tags">${entry.tags.map((tag) => `<span>${tag}</span>`).join('')}</p></details>`).join('')}`;
  detail.showModal();
}

results.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-row]');
  if (!button) return;
  selected = Number(button.dataset.row);
  for (const lengthButton of results.querySelectorAll('.length')) {
    lengthButton.setAttribute('aria-pressed', Number(lengthButton.dataset.row) === selected);
  }
  renderLayout();
  if (button.dataset.column !== undefined) openDetail(selected, Number(button.dataset.column));
});

for (const control of [scenario, recover, draft, spec]) control.addEventListener('change', render);
byId('reset').addEventListener('click', () => {
  scenario.value = '8';
  recover.checked = true;
  draft.checked = true;
  spec.value = '3';
  selected = 3;
  render();
});

render();
