'use strict';

const { inputs, defaultBlock, platforms, mlaKernelBlock, mainLayers, stateGroups, layersPerStateGroup, draftLayers,
  k3Preset, k3Platforms, customTpOptions, customDcpOptions, validCustomParallel, gqaFitsK3Page,
  calculate: calculateK31, calculateK3 } = KVCache;
const byId = (id) => document.getElementById(id);
const modelChoice = byId('model');
const hardware = byId('hardware');
const weight = byId('weight');
const scenario = byId('scenario');
const customDp = byId('custom-dp');
const customTp = byId('custom-tp');
const customDcp = byId('custom-dcp');
const recover = byId('recover');
const draft = byId('draft');
const spec = byId('spec');
const mlaDtype = byId('mla-dtype');
const indexDtype = byId('index-dtype');
const draftDtype = byId('draft-dtype');
const draftKinds = Array.from(document.querySelectorAll('input[name="draft-kind"]'));
const mlaDraftLayers = byId('mla-draft-layers');
const blockSize = byId('block-size');
const results = byId('results');
const layout = byId('layout');
const layoutCaption = byId('layout-caption');
const blockAdjustment = byId('block-adjustment');
const layoutVisual = byId('layout-visual');
const detail = byId('detail');
let selected = 3;
let selectedMlaId = 0;
let showMiddleSlots = true;
let current;
let presetDcp;
const draftSpecByKind = {};

const fmt = new Intl.NumberFormat('zh-CN');
const memory = (bytes) => bytes >= 2 ** 30
  ? `${(bytes / 2 ** 30).toFixed(3)} GiB`
  : `${(bytes / 2 ** 20).toFixed(2)} MiB`;
const lengths = ['4k+1k', '16k+1k', '64k+1k', '128k+1k', '1M'];
const topology = ({ platform, dcp, dp = 4, tp = platforms[platform].tp }) => `DP${dp}${dcp > 1 ? ` / DCP${dcp}` : ''} / TP${tp}`;
const weightLabel = (value) => value === 'native' ? 'Moonshot 原生 MXFP4' : value;
const blockInputPrefix = ({ requestedBlock, localBlock }) => requestedBlock === localBlock ? '' : `${requestedBlock} → `;
const mlaLayer = (slot) => slot === layersPerStateGroup ? 92 : slot * (stateGroups + 1) + stateGroups;
const kdaLayer = (slot, group) => slot * (stateGroups + 1) + group;
const examplePoolId = (position) => [29, 81, 12, 54][position] ?? 100 + position;
const calculate = (input, config) => config.modelId === 'k3' ? calculateK3(input, config) : calculateK31(input, config);
const activeDraftKind = () => draftKinds.find(input => input.checked).value;

function describe({ recover: enabled, spec: count }, { draftInfo }) {
  const draftLabel = count ? `${draftInfo?.kind ?? 'MLA'} ${draftInfo?.layers ?? draftLayers} 层 / ${count} token` : '关';
  return `RecoverSSM ${enabled ? '开' : '关'} · DSpark ${draftLabel}`;
}

function geometryEvidence(value, config) {
  if (config.customParallel) {
    if (config.weight === 'native') return { label: '自定义并行策略 · 容量推算 · 原生权重加载待验证' };
    return { label: '自定义并行策略 · 容量推算 · 部署待验证' };
  }
  if (config.modelId === 'k3') {
    if (value.draftInfo?.kind === 'GQA') {
      return { label: config.dcp > 1 ? 'GQA / DCP 结构对照 · 当前草稿组合无法部署' : 'GQA 分组按配置推算' };
    }
    if (config.platform === 'a3' && config.dcp === 16 && !config.recover && config.spec === 7 && value.draftInfo?.layers === 5) {
      return { label: '单卡页宽 384 token · A3 1M Decode 模板',
        url: 'https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.html#53-long-context-and-1m-pd-separation-deployment' };
    }
    return { label: '当前布局按配置推算' };
  }
  if (config.weight === 'native') return { label: '原生权重加载待验证' };
  if (hasAlternateDtype(config)) return { label: 'dtype 对照 · 部署未验证' };
  const baseline = config.weight === platforms[config.platform].weights[0] && config.dcp === platforms[config.platform].dcp;
  if (baseline && config.spec === 3 && config.recover === platforms[config.platform].recover) {
    if (config.platform === 'a3') return { label: '单卡页宽参考 A3 模板与源码',
      url: 'https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.1-Pro.html#long-context-and-1m-pd-separation-deployment' };
    if (value.requestedBlock === defaultBlock) return { label: '历史启动日志的页宽：传入 768，生效 1536 token / 卡',
      url: 'https://github.com/Dawn952/kimi-k31-a5-analysis/blob/main/data/memory-dcp-evidence.json' };
  }
  return { label: '当前布局按配置推算' };
}

function evidenceMarkup(value) {
  const { label, url } = geometryEvidence(value, current);
  return url ? `<a href="${url}">${label}</a>` : label;
}

function lengthEvidence(row) {
  return current.modelId === 'k3' && current.platform === 'a3' && current.dcp === 1 && row === lengths.length - 1
    ? '1M 仅作容量对照，关闭 DCP 的 A3 模板为 256K。'
    : '';
}

function hasAlternateDtype(config) {
  const baseline = platforms[config.platform].cacheDtype;
  return config.mlaDtype !== baseline || config.indexDtype !== baseline || (config.spec > 0 && config.draftDtype !== 'BF16');
}

function updateCustomDcp() {
  const tp = Number(customTp.value);
  const previous = customDcp.value;
  const choices = customDcpOptions(hardware.value, tp);
  customDcp.replaceChildren(new Option(tp ? '选择 DCP' : '先选 TP', ''),
    ...choices.map(dcp => new Option(dcp === 1 ? '1（关闭）' : dcp, dcp)));
  customDcp.disabled = choices.length === 0;
  if (choices.includes(Number(previous))) customDcp.value = previous;
}

function applySharedParallel(query) {
  const setOption = (control, key) => {
    const value = query.get(key);
    if (Array.from(control.options).some(option => option.value === value)) control.value = value;
  };
  setOption(weight, 'weight');
  setOption(blockSize, 'block');
  setOption(mlaDtype, 'mla');
  setOption(indexDtype, 'index');
  setOption(draftDtype, 'draftdtype');
  setOption(spec, 'spec');
  if (modelChoice.value === 'k3') {
    const kind = draftKinds.find(input => input.value === query.get('kind'));
    if (kind) kind.checked = true;
    setOption(mlaDraftLayers, 'layers');
  }
  if (query.has('draft')) draft.checked = query.get('draft') === '1';
  if (query.has('recover')) recover.checked = query.get('recover') === '1';
  const input = query.get('input');
  if (input && inputs.includes(Number(input))) selected = inputs.indexOf(Number(input));
  scenario.value = 'custom';
  const dp = query.get('dp');
  if (dp && /^[1-9]\d*$/.test(dp) && Number.isSafeInteger(Number(dp))) customDp.value = dp;
  setOption(customTp, 'tp');
  updateCustomDcp();
  setOption(customDcp, 'dcp');
  render();
}

function selectPlatform() {
  const platform = hardware.value;
  const k3 = modelChoice.value === 'k3';
  const k3Platform = k3Platforms[platform];
  const preset = k3 ? { weights: ['W4A8'], blockOptions: [k3Platform.blockSize], defaultBlock: k3Platform.blockSize,
    dsaDtypes: ['BF16'], draftDtypes: ['BF16'], recover: false } : platforms[platform];
  const dcpOptions = k3 ? k3Platform.dcpOptions : platform === 'a5' ? [preset.dcp, 1] : [preset.dcp];
  weight.replaceChildren(...preset.weights.map(value => new Option(weightLabel(value), value)));
  scenario.replaceChildren(...dcpOptions.map(dcp => new Option(
    topology({ platform, dcp }) + (dcp === 1 && dcpOptions.length > 1 ? '（DCP 关闭）' : ''), dcp)),
    new Option('自定义', 'custom'));
  presetDcp = dcpOptions[0];
  customDp.value = '';
  customTp.replaceChildren(new Option('选择 TP', ''), ...customTpOptions(platform).map(tp => new Option(tp, tp)));
  updateCustomDcp();
  blockSize.replaceChildren(...preset.blockOptions.map(size => new Option(size, size)));
  blockSize.value = String(preset.defaultBlock);
  mlaDtype.replaceChildren(...preset.dsaDtypes.map(value => new Option(value, value)));
  indexDtype.replaceChildren(...preset.dsaDtypes.map(value => new Option(value, value)));
  draftDtype.replaceChildren(...preset.draftDtypes.map(value => new Option(value === 'BF16' ? value : `${value}（容量估算）`, value)));
  recover.checked = preset.recover;
  draft.checked = true;
  spec.value = String(k3 ? k3Platform.spec : 3);
  delete draftSpecByKind.mla;
  delete draftSpecByKind.gqa;
  if (k3) mlaDraftLayers.value = String(k3Platform.mlaDraftLayers);
  render();
}

function selectModel(preferredHardware = hardware.value) {
  const k3 = modelChoice.value === 'k3';
  const modelName = modelChoice.selectedOptions[0].textContent;
  hardware.replaceChildren(...Object.keys(platforms).map(id => new Option(platforms[id].name, id)));
  if (Object.hasOwn(platforms, preferredHardware)) hardware.value = preferredHardware;
  byId('page-title').textContent = document.title = `${modelName} KVCache 对比台`;
  byId('layout-title').textContent = `${modelName} 的 LBNHC 布局`;
  byId('attention-options-title').textContent = k3 ? 'MLA 历史缓存' : 'DSA 稀疏注意力';
  byId('index-dtype-field').hidden = k3;
  byId('k3-method').hidden = !k3;
  byId('k31-method').hidden = k3;
  draftKinds[0].checked = true;
  selectPlatform();
}

function render() {
  const k3 = modelChoice.value === 'k3';
  const kind = activeDraftKind();
  const mla = kind === 'mla';
  const draftEnabled = draft.checked;
  const customScenario = scenario.value === 'custom';
  const platform = hardware.value;
  const dp = customScenario ? customDp.valueAsNumber : 4;
  const tp = customScenario ? Number(customTp.value) : platforms[platform].tp;
  const dcp = customScenario ? Number(customDcp.value) : Number(scenario.value);
  const gqaDraft = k3 && !mla && draftEnabled;
  const validTopology = validCustomParallel(platform, dp, tp, dcp);
  const gqaIssue = gqaDraft && (tp > 0 && !gqaFitsK3Page(tp) || dcp > 1);
  const pendingScenario = customScenario && (!validTopology || gqaIssue);
  byId('custom-parallel').hidden = !customScenario;
  if (!customScenario) presetDcp = dcp;
  const size = byId('custom-parallel-size');
  size.hidden = !customScenario || !validTopology;
  if (customScenario && validTopology) size.textContent = `DP × TP = ${fmt.format(BigInt(dp) * BigInt(tp))} 个并行 rank，DCP 复用 TP rank。`;
  byId('copy-parallel-link').disabled = pendingScenario;
  byId('parallel-link-status').textContent = '';
  byId('parallel-link-fallback').hidden = true;
  const maxSpec = k3 && mla ? k3Platforms[hardware.value].maxMlaSpec : 7;
  for (const option of spec.options) option.hidden = option.disabled = Number(option.value) > maxSpec;
  if (Number(spec.value) > maxSpec) spec.value = String(maxSpec);
  byId('draft-fields').hidden = !draftEnabled;
  byId('draft-kind-field').hidden = !k3;
  byId('draft-format').hidden = !k3 || !draftEnabled;
  byId('mla-draft-layers-field').hidden = !k3 || !mla;
  byId('gqa-draft-layers-field').hidden = !k3 || mla;
  byId('draft-dtype-field').hidden = k3;
  byId('draft-layout-note').hidden = !k3;
  byId('draft-layout-note').textContent = mla ? '沿用主模型的历史池块 ID，草稿层接在 MLA 层之后。'
    : '使用独立的历史块表。Ascend 的 GQA 布局按容量推算。';
  recover.disabled = !draftEnabled;
  current = { modelId: modelChoice.value, draftKind: kind, mlaDraftLayerCount: Number(mlaDraftLayers.value), platform,
    weight: weight.value, customParallel: customScenario, dp, tp, dcp, recover: draftEnabled && recover.checked,
    spec: draftEnabled ? Number(spec.value) : 0, requestedBlock: Number(blockSize.value),
    mlaDtype: mlaDtype.value, indexDtype: indexDtype.value, draftDtype: draftDtype.value };
  byId('native-weight-note').hidden = current.weight !== 'native';
  byId('dtype-note').hidden = k3 || !hasAlternateDtype(current);
  byId('results-scope').hidden = pendingScenario;
  const customNote = byId('custom-scenario-note');
  customNote.hidden = !pendingScenario;
  if (pendingScenario) {
    customNote.textContent = gqaIssue ? 'GQA Draft 的共用页估算要求 TP 至少为 4、DCP 为 1。'
      : customDp.value && (!Number.isSafeInteger(dp) || dp < 1) ? 'DP 须为正整数。' : '填写 DP、TP 和 DCP 后显示估算。';
  }
  byId('cache-layout').hidden = pendingScenario;
  results.setAttribute('aria-describedby', pendingScenario ? 'custom-scenario-note' : 'results-scope');
  results.tBodies[0].innerHTML = lengths.map((length, index) => {
    if (pendingScenario) return `<tr role="row"><th scope="row" role="rowheader">${length}</th><td role="cell"></td></tr>`;
    const input = inputs[index];
    const value = memory(calculate(input, current).bytes);
    const scopeLabel = lengthEvidence(index) ? '<small>容量对照</small>' : '';
    return `<tr role="row">
      <th scope="row" role="rowheader"><button type="button" class="length" data-row="${index}" aria-controls="cache-layout">${length}${scopeLabel}</button></th>
      <td role="cell"><button type="button" class="value" data-row="${index}" aria-haspopup="dialog" aria-label="${length}，${value}，查看明细">${value}</button></td>
    </tr>`;
  }).join('');
  if (!pendingScenario) updateSelection();
}

function updateSelection() {
  for (const button of results.querySelectorAll('button[data-row]')) {
    const active = Number(button.dataset.row) === selected;
    button.parentElement.classList.toggle('selected-length', active);
    if (button.classList.contains('length')) button.setAttribute('aria-pressed', active);
  }
  renderLayout();
}

function renderBlockSize({ requestedBlock, localBlock, upstreamQuantumBytes, alignmentMultiple, pages }) {
  const effective = `${localBlock} token / 卡`;
  if (requestedBlock === localBlock) {
    blockAdjustment.hidden = true;
    blockAdjustment.textContent = '';
    return effective;
  }

  const bytesPerToken = upstreamQuantumBytes / requestedBlock;
  const doubled = alignmentMultiple > 2
    ? `翻倍到 ${requestedBlock * 2} token 只有 ${fmt.format(upstreamQuantumBytes * 2)} B，仍然不够。`
    : '';
  blockAdjustment.innerHTML = `
    <p><strong>启动时选块。</strong> ${current.mlaDtype} MLA 按 ${bytesPerToken} B/token 计算。${requestedBlock} token 只提供 ${fmt.format(upstreamQuantumBytes)} B，KDA 单层状态需要 ${fmt.format(pages.state)} B。${doubled}取最小够用的整数倍 ${requestedBlock} × ${alignmentMultiple} = ${localBlock} token，对应 ${fmt.format(upstreamQuantumBytes * alignmentMultiple)} B。</p>
    <p><strong>分配物理页。</strong> ${current.mlaDtype === 'BF16' ? '按同样的字节宽度' : '计入 BF16 RoPE 和 scale 后'}，MLA 页宽为 ${localBlock} × ${pages.main / localBlock} B = ${fmt.format(pages.main)} B。KDA 按此页宽对齐。</p>
    <p class="sources"><a href="https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/platforms/interface.py#L1076-L1128">vLLM 选块源码</a><a href="https://github.com/Dawn952/vllm-ascend/blob/3634a7eae94897d1a7b14d00807323e5af6b4a04/vllm_ascend/platform.py#L470-L489">Ascend 物理页宽源码</a></p>`;
  return `${requestedBlock} → <button type="button" class="explanation-toggle" aria-controls="block-adjustment" aria-expanded="${!blockAdjustment.hidden}">${effective}</button>`;
}

function allocationFormula({ historyIds, stateCopies, draftInfo, poolLayers, pages, bytes, bytesPerId }, explainStates = false) {
  const draftIds = draftInfo?.ids ?? 0;
  const copies = explainStates && stateCopies > 1
    ? `<button type="button" class="explanation-toggle" aria-label="解释每组为何有 ${stateCopies} 个状态块" aria-controls="kda-state-explanation" aria-expanded="false">${stateCopies}</button>`
    : stateCopies;
  const perId = poolLayers
    ? `${poolLayers} 层槽 × ${fmt.format(pages.main)} B`
    : `${mainLayers} × ${fmt.format(pages.main)} B（MLA / KDA） + ${mainLayers} × ${fmt.format(pages.index)} B（Indexer）${pages.draft ? ` + ${draftLayers} × ${fmt.format(pages.draft)} B（Draft）` : ''}`;
  return `<div class="allocation-formula">
    <div><strong>${historyIds + stateGroups * stateCopies + draftIds} 个池块 ID</strong><small>${historyIds} MLA 历史 + ${stateGroups} × ${copies} KDA 状态${draftIds ? ` + ${draftIds} GQA Draft` : ''}</small></div>
    <span aria-hidden="true">×</span>
    <div><strong>${fmt.format(bytesPerId)} B / ID / 卡</strong><small>${perId}</small></div>
    <span aria-hidden="true">=</span>
    <strong class="formula-total">${memory(bytes)} / 卡</strong>
  </div>`;
}

function renderLayout() {
  const value = calculate(inputs[selected], current);
  const gqa = Boolean(value.draftInfo?.ids);
  const mlaDraft = value.draftInfo?.kind === 'MLA';
  const separateDraftIds = value.draftInfo?.ids ?? 0;
  const groups = Array.from({ length: stateGroups }, (_, index) => index);
  const { pages, regions } = value;
  const stateCount = `${value.stateCopies} 块`;
  const blockMarks = (count) => `<span class="block-marks" aria-hidden="true">${'<i></i>'.repeat(Math.min(count, 12))}${count > 12 ? '<b>…</b>' : ''}</span>`;
  const mainSlots = Array.from({ length: mainLayers }, (_, slot) => {
    const empty = slot === layersPerStateGroup;
    const collapse = slot === 3
      ? '<button id="collapse-middle-slots" type="button" class="slot-collapse" data-layout-panel="middle-slots" aria-controls="middle-slots" aria-expanded="true" aria-label="收起第 3–22 行">收起</button>'
      : '';
    return `<tr><th scope="row">第 ${slot} 行${collapse}</th><td class="mla">MLA 第 ${mlaLayer(slot)} 层</td>${groups.map((group) => empty
      ? '<td class="empty">空槽</td>' : `<td class="kda">KDA 第 ${kdaLayer(slot, group)} 层</td>`).join('')}${gqa
      ? (slot < value.draftInfo.layers ? `<td class="draft">GQA Draft 第 ${slot} 层</td>` : '<td class="empty">空槽</td>') : ''}</tr>`;
  });
  const sideRow = (name, layers, kind, text) => `<tr><th scope="row">${name} 区 × ${layers} 层</th><td class="${kind}">${text}</td>${groups.map(() => '<td class="empty">预留空页</td>').join('')}</tr>`;
  byId('results-evidence').innerHTML = evidenceMarkup(value) + '。';
  layoutCaption.innerHTML = `${lengths[selected]} · ${topology(current)} · ${describe(current, value)} · 生效页大小 ${renderBlockSize(value)}`;
  const scope = byId('layout-scope');
  scope.textContent = lengthEvidence(selected);
  scope.hidden = !scope.textContent;
  byId('layout-summary').innerHTML = `
    ${allocationFormula(value, true)}
    ${value.stateCopies > 1 ? `<p id="kda-state-explanation" class="caption inline-explanation" hidden>RecoverSSM 关闭时，KDA 为每个投机 token 位置保留一份候选状态，验证后按实际接受的长度选取下一轮解码所需的状态。当前 <code>spec_num = ${current.spec}</code>，所以每组是 <strong>1 块运行状态 + ${current.spec} 块投机状态 = ${stateCount}</strong>。三组共 ${value.stateIds} 块。</p>` : ''}`;
  renderLayoutVisual(value);
  renderTrace();
  layout.innerHTML = `
    <div class="axis-key" id="layout-reading-guide">
      <div>第 0 个存储行用不同的池块 ID 分别存模型第 3 层的 MLA，以及第 0、1、2 层的 KDA。</div>
    </div>
    <div class="scroll"><table class="pool-map" aria-describedby="layout-reading-guide">
      <caption>${current.modelId === 'k3' ? 'Kimi-K3 的物理存储行' : 'MLA / KDA 共用区，以及独立的 Indexer / Draft 区'}</caption>
      <thead><tr><th scope="col">存储行（L）</th><th scope="col">MLA 历史 · ${value.historyIds} 块${blockMarks(value.historyIds)}</th>${groups.map((group) => `<th scope="col">KDA ${group + 1} 组 · ${value.stateCopies} 块${blockMarks(value.stateCopies)}</th>`).join('')}${gqa ? `<th scope="col">GQA Draft · ${separateDraftIds} 块${blockMarks(separateDraftIds)}</th>` : ''}</tr></thead>
      <tbody>${mainSlots.slice(0, 3).join('')}<tr id="middle-slots-placeholder" ${showMiddleSlots ? 'hidden' : ''}><th scope="row">第 3–22 行</th><td colspan="${gqa ? 5 : 4}" class="empty"><button id="expand-middle-slots" type="button" class="layout-toggle" data-layout-panel="middle-slots" aria-controls="middle-slots" aria-expanded="false">展开第 3–22 行</button></td></tr></tbody>
      <tbody id="middle-slots" ${showMiddleSlots ? '' : 'hidden'}>${mainSlots.slice(3, -1).join('')}</tbody>
      <tbody>${mainSlots.at(-1)}</tbody>
      ${pages.index ? `<tbody>${sideRow('Indexer', mainLayers, 'indexer', `${current.indexDtype} K${current.indexDtype === 'BF16' ? '' : ' + scale'}`)}${pages.draft ? sideRow('Draft', draftLayers, 'draft', `${current.draftDtype} MLA KV`) : ''}</tbody>` : ''}
      ${mlaDraft ? `<tbody>${Array.from({ length: value.draftInfo.layers }, (_, i) => `<tr><th scope="row">第 ${mainLayers + i} 行</th><td class="draft">MLA Draft 第 ${i} 层</td>${groups.map(() => '<td class="empty">空槽</td>').join('')}</tr>`).join('')}</tbody>` : ''}
    </table></div>
    <h3>一个池 block 的显存怎么算</h3>
    <div class="scroll"><table class="pool-map" aria-label="单池 block 显存分项"><thead><tr><th scope="col">区域</th><th scope="col">物理页数 × 每页字节</th><th scope="col">每个池 ID</th></tr></thead><tbody>
      <tr><th scope="row">${gqa ? 'MLA / KDA / GQA Draft 共用池' : 'MLA / KDA'}</th><td>${mainLayers} × ${fmt.format(pages.main)} B</td><td>${memory(regions.main)}</td></tr>
      ${pages.index ? `<tr><th scope="row">Indexer</th><td>${mainLayers} × ${fmt.format(pages.index)} B</td><td>${memory(regions.index)}</td></tr>` : ''}
      ${pages.index || regions.draft ? `<tr><th scope="row">Draft</th><td>${pages.draft ? (value.draftInfo?.layers ?? draftLayers) : 0} × ${fmt.format(pages.draft)} B</td><td>${memory(regions.draft)}</td></tr>` : ''}
    </tbody></table></div>
    <p class="caption">MLA 历史块数按逻辑覆盖取整，投机预留受本档 <code>max-model-len</code> 限制。KDA 每组常驻 1 个运行状态块，RecoverSSM 关闭时再计 <code>spec_num</code> 个投机状态块。</p>`;
}

function sampleBlockIds(count, active = 0) {
  const ids = [...new Set([0, 1, active, count - 1].filter(id => id >= 0 && id < count))].sort((left, right) => left - right);
  return ids.flatMap((id, index) => index > 0 && id - ids[index - 1] > 1 ? [null, id] : [id]);
}

function renderLayoutVisual(value, focusBlock = false) {
  const { historyIds, stateCopies, stateIds, pages, limit, coverage, draftInfo } = value;
  const k3 = current.modelId === 'k3';
  const previousScroll = layoutVisual.querySelector('.visual-block-scroll')?.scrollLeft ?? 0;
  const pageDetails = focusBlock ? layoutVisual.querySelector('.visual-block-detail') : null;
  const scrollLeft = selectedMlaId < historyIds ? previousScroll : 0;
  if (selectedMlaId >= historyIds) selectedMlaId = 0;
  const selectedPoolId = examplePoolId(selectedMlaId);
  const samples = sampleBlockIds(historyIds, selectedMlaId);
  const gap = '<span class="visual-gap" aria-hidden="true">…</span>';
  const pools = [{ label: 'MLA 历史', count: historyIds, start: 0 }];
  for (let group = 0; group < stateGroups; group++) {
    pools.push({ label: `KDA 第 ${group + 1} 组`, count: stateCopies, start: historyIds + group * stateCopies });
  }
  const kdaPools = pools.slice(1);
  const gqaPool = draftInfo?.ids ? { label: 'GQA Draft', count: draftInfo.ids, start: historyIds + stateIds } : null;
  if (gqaPool) pools.push(gqaPool);
  const regions = [{ label: 'MLA / KDA', kind: 'mla', bytes: pages.main, layers: mainLayers, start: 0 }];
  if (pages.index) regions.push({ label: 'Indexer', kind: 'indexer', bytes: pages.index, layers: mainLayers, start: 0 });
  if (pages.draft && !gqaPool) regions.push({ label: k3 ? 'MLA Draft' : 'Draft', kind: 'draft', bytes: pages.draft,
    layers: draftInfo?.layers ?? draftLayers, start: draftInfo?.startRow ?? 0 });

  const strip = (pool, kind, label) => `<div class="visual-block-group"><small>${label}</small>
    <div class="visual-block-strip" role="img" aria-label="${label}，${pool.count} 个池 block${kind === 'kda' ? `，1 个运行状态 block${pool.count > 1 ? `，${pool.count - 1} 个投机状态 block` : ''}` : ''}">
    ${(pool.start === 0 ? samples : sampleBlockIds(pool.count)).map((index) => {
      if (index === null) return gap;
      const id = pool.start + index;
      const usage = kind === 'kda' ? (index ? '投机状态' : '运行状态') : label;
      const scratch = kind === 'kda' && index ? ' speculative' : '';
      const mlaId = pool.start === 0 ? ` data-mla-block="${id}"${id === selectedMlaId ? ' data-matched' : ''}` : '';
      return `<span class="visual-block ${kind}${scratch}"${mlaId} title="池 block ${examplePoolId(id)} · ${usage}" aria-hidden="true">${examplePoolId(id)}</span>`;
    }).join('')}</div></div>`;

  const renderRow = (region, slot) => {
    const kdaSlot = region.kind === 'mla' && slot < layersPerStateGroup;
    const historyLabel = region.kind === 'draft' ? (k3 ? `Draft MLA 第 ${slot} 层` : 'Draft MLA KV') : `模型第 ${mlaLayer(slot)} 层`;
    const kdaCells = kdaPools.map((pool, group) => `<td>${kdaSlot
      ? strip(pool, 'kda', `模型第 ${kdaLayer(slot, group)} 层`)
      : strip(pool, 'empty', k3 ? '空层槽' : '预留空页')}</td>`).join('');
    const draftCell = gqaPool ? `<td>${slot < draftInfo.layers
      ? strip(gqaPool, 'draft', `GQA Draft 第 ${slot} 层`)
      : strip(gqaPool, 'empty', '空层槽')}</td>` : '';
    return `<tr><th scope="row">${region.label} · 行 ${region.start + slot}<small>共 ${region.layers} 行 · ${fmt.format(region.bytes)} 字节 / block</small></th>
      <td>${strip(pools[0], region.kind, historyLabel)}</td>${kdaCells}${draftCell}</tr>`;
  };

  const activeTokens = Math.min(coverage, limit - selectedMlaId * coverage);
  const unusedTokens = historyIds * coverage - limit;
  layoutVisual.innerHTML = `
    <h3>从请求到单卡物理页</h3>
    <div class="block-path" id="visual-block-guide">
      <div><small>请求长度上限</small><strong>${fmt.format(limit)} token</strong></div>
      <div><small>MLA 逻辑块</small><strong>${fmt.format(historyIds)} 块</strong><small>每块覆盖 ${fmt.format(coverage)} token</small></div>
      <div><small>每层的单卡物理页</small><strong>${fmt.format(value.localBlock)} token</strong><small>${current.dcp} 卡共同覆盖一个逻辑块</small></div>
    </div>
    <div class="mla-block-lookup">
      <form id="block-search" class="block-search">
        <label>请求块号 <input id="logical-block-input" type="number" min="0" max="${historyIds - 1}" step="1" value="${selectedMlaId}" required aria-describedby="block-input-help"></label>
        <button type="submit">定位</button>
        ${historyIds > 1 ? '<button type="button" data-last-block>尾块</button>' : ''}
        <small id="block-input-help">0–${fmt.format(historyIds - 1)}</small>
      </form>
      <strong aria-live="polite">请求块 ${selectedMlaId} → 池块 ID ${selectedPoolId}（示例）</strong>
    </div>
    <p class="caption">块表（<code>block_table</code>）记录请求块号到池块 ID 的映射。各 MLA 层用同一个 ID 在自己的存储行读写。</p>
    ${selectedMlaId === historyIds - 1 ? `<div class="tail-usage">
      <div class="tail-token-track" role="img" aria-label="请求尾块已填 ${fmt.format(activeTokens)} token，未填 ${fmt.format(unusedTokens)} token" style="grid-template-columns:${activeTokens}fr ${unusedTokens}fr"><span></span><span></span></div>
      <p class="caption"><strong>尾块已填 ${fmt.format(activeTokens)} token</strong>${unusedTokens ? ` · 未填 ${fmt.format(unusedTokens)} token` : ''}${current.dcp > 1 ? '（各卡合计）' : ''}。显存按分配的整页计算。</p>
    </div>` : ''}
    ${k3 ? (draftInfo ? `<p class="caption"><strong>Draft 放在哪。</strong> ${gqaPool
        ? `GQA 有独立块表，另占 ${draftInfo.ids} 个历史池块 ID，在同池使用前 ${draftInfo.layers} 行，其余 ${value.poolLayers - draftInfo.layers} 行是空层槽。`
        : `MLA Draft 共用历史池块 ID，草稿页接在第 ${draftInfo.startRow}–${draftInfo.startRow + draftInfo.layers - 1} 行。每个池块都为这 ${draftInfo.layers} 行留空间，KDA 的 ID 在这些行是空槽。`}</p>` : '')
      : `<p class="caption">Indexer 独立存放用于筛选历史 token 的索引 K，沿用目标 MLA 的历史池块 ID。${pages.draft ? '草稿 MLA 也用同一 ID 在自己的存储区寻址。' : ''}</p>`}
    <div class="visual-block-nav" role="group" aria-label="跳到缓存组">${pools.map((pool, group) => `<button type="button" data-block-group="${group}">${pool.label} <span>· ${pool.count} 个 block</span></button>`).join('')}</div>
    <p class="caption">同一存储行内，不同组的池块可存不同模型层。每格是一层在一个池块 ID 下的单卡页。</p>
    <div class="scroll visual-block-scroll" tabindex="0" role="region" aria-label="缓存 block 图，可横向滚动">
      <table class="visual-block-map" aria-describedby="visual-block-guide">
        <thead><tr><th scope="col">存储行（L）</th>
          <th scope="col">${pools[0].label}<small title="请求 token 数 ÷ 每个逻辑 block 覆盖的 token 数，向上取整">⌈${fmt.format(limit)} ÷ ${fmt.format(coverage)}⌉ = ${historyIds} 个 block</small>
            <small>逻辑 block 编号 · 点击查看查表结果</small>
            <div class="visual-block-strip" role="group" aria-label="MLA 逻辑 block 编号，可用左右方向键切换">
              ${samples.map(id => id === null ? gap : `<button type="button" class="visual-block visual-block-id" data-mla-id="${id}" tabindex="${id === selectedMlaId ? 0 : -1}" aria-pressed="${id === selectedMlaId}" aria-label="请求块 ${id}，对应示例池块 ${examplePoolId(id)}">${id}</button>`).join('')}
            </div>
          </th>
          ${pools.slice(1).map((pool) => `<th scope="col">${pool.label}<small>${pool.count} 个 block</small></th>`).join('')}
        </tr></thead>
        ${regions.map((region) => `<tbody>${(region.kind === 'mla' ? [0, 1, layersPerStateGroup] : [0, 1])
          .map((slot) => renderRow(region, slot)).join('')}</tbody>`).join('')}
      </table>
    </div>
    <p class="caption">省略号表示中间块。示例池块号可以跳号。斜纹表示空槽，虚线表示投机状态。</p>
    <div class="space-guide">
      <div><small>请求尾块未填</small><strong>${fmt.format(unusedTokens)} token</strong><small>按整个逻辑块计</small></div>
      <div><small>KDA 页内对齐填充</small><strong>${fmt.format(pages.padding)} B</strong><small>每层每页</small></div>
      <div><small>KDA 池块的空层槽</small><strong>${(value.poolLayers ?? mainLayers) - layersPerStateGroup} 个</strong><small>每个 KDA 池块 ID 的共用区${pages.index ? '，侧区另计' : ''}</small></div>
    </div>
    <div class="physical-block-order" role="group" aria-label="MLA / KDA 共用区整池物理顺序">
      <strong>固定存储行 L</strong>
      <div class="visual-block-strip" aria-label="实际地址顺序的相邻池块编号示例" role="group">
        <span class="visual-block">${selectedPoolId - 1}</span>
        <span class="visual-block physical-order-focus">${selectedPoolId}</span>
        <span class="visual-block">${selectedPoolId + 1}</span>
      </div>
      <strong>池块 ID 与地址递增 →</strong><span>相邻页起点相差 ${fmt.format(pages.main)} B</span>
    </div>
    ${pageDetails ? '' : `<details class="visual-block-detail" open><summary>单卡页内字节比例</summary>${renderBlockZoom(value)}</details>`}`;
  if (pageDetails) layoutVisual.append(pageDetails);
  const scroller = layoutVisual.querySelector('.visual-block-scroll');
  scroller.scrollLeft = scrollLeft;
  if (focusBlock) {
    const activeButton = layoutVisual.querySelector(`[data-mla-id="${selectedMlaId}"]`);
    activeButton.focus({ preventScroll: true });
    activeButton.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const buttonRect = activeButton.getBoundingClientRect();
    const pinnedRight = scroller.querySelector('thead th').getBoundingClientRect().right;
    const visibleRight = scroller.getBoundingClientRect().right;
    const center = pinnedRight + (visibleRight - pinnedRight - buttonRect.width) / 2;
    scroller.scrollLeft += buttonRect.left - center;
  }
}

function renderBlockZoom({ pages, localBlock, draftInfo }) {
  const tiles = localBlock / mlaKernelBlock;
  const tileBytes = pages.main / tiles;
  const mlaParts = Array.from({ length: tiles }, (_, index) => ({
    label: `Kernel 小块 ${index + 1}`, bytes: tileBytes, css: 'mla', text: index + 1,
  }));
  const kdaParts = [
    { label: '卷积历史（Conv）', bytes: pages.conv, css: 'conv' },
    { label: '递推状态（Recurrent）', bytes: pages.recurrent, css: 'kda' },
    { label: '对齐填充（Padding）', bytes: pages.padding, css: 'empty' },
  ].filter(part => part.bytes > 0);
  const draftParts = draftInfo ? [
    { label: draftInfo.kind === 'GQA' ? 'K / V 有效载荷' : 'MLA 有效载荷', bytes: draftInfo.payload, css: 'draft' },
    ...(draftInfo.padding ? [{ label: 'Draft 页内 padding', bytes: draftInfo.padding, css: 'empty' }] : []),
  ] : [];
  const track = (name, parts, withLegend = false) => {
    let offset = 0;
    const segments = parts.map(({ label, bytes, css, text }) => {
      const start = offset;
      offset += bytes;
      const caption = `${label} · ${fmt.format(bytes)} B · 偏移 [${fmt.format(start)}, ${fmt.format(offset)}) B`;
      return { bytes, caption, css, label, text };
    });
    const buttons = segments.map(({ caption, css, text }) => `<button type="button" class="byte-segment ${css}" data-byte-caption="${caption}" aria-label="${name} ${caption}" title="${caption}">${text ?? ''}</button>`);
    const columns = parts.map(({ bytes }) => `minmax(0, ${bytes}fr)`).join(' ');
    const legend = withLegend ? `<div class="byte-legend">${segments.map(({ bytes, caption, css, label }) => `<button type="button" data-byte-caption="${caption}"><i class="${css}" aria-hidden="true"></i>${label} ${fmt.format(bytes)} B</button>`).join('')}</div>` : '';
    return `<div class="byte-track" role="group" aria-label="${name}" style="grid-template-columns:${columns}">${buttons.join('')}</div>${legend}`;
  };
  const sidePage = (label, bytes, dtype, css) => `<div class="side-page">
    <div class="byte-label"><strong>${label}</strong><span>${dtype} · 独立标尺</span></div>
    <div class="byte-scale"><span>0 B</span><span>${fmt.format(bytes)} B</span></div>
    ${track(label, [{ label, bytes, css }])}
  </div>`;
  return `<div class="block-zoom">
    <p class="caption"><code>L → B → N → H → C</code> 依次是存储行、池块 ID、本卡页内 token、KV 头和通道。KDA 状态按 MLA 的单页字节宽度对齐，共用以下标尺。</p>
    <div class="byte-scale" aria-label="单卡物理页内字节偏移"><span>0 B</span><span>${fmt.format(pages.main)} B</span></div>
    <div class="byte-label"><strong>MLA 物理页</strong><span>${current.mlaDtype} · ${tiles} 小块 × ${mlaKernelBlock} token，每小块 ${fmt.format(tileBytes)} B</span></div>
    ${track('MLA 物理页', mlaParts)}
    <div class="byte-label"><strong>KDA 对齐页</strong><span>${fmt.format(pages.state)} B 状态 + ${fmt.format(pages.padding)} B padding</span></div>
    ${track('KDA 对齐页', kdaParts, true)}
    ${draftInfo ? `<div class="byte-label"><strong>${draftInfo.kind} Draft 页</strong><span>${fmt.format(draftInfo.payload)} B 有效数据 · ${fmt.format(draftInfo.padding)} B padding</span></div>
      ${track(`${draftInfo.kind} Draft 页`, draftParts, Boolean(draftInfo.padding))}` : ''}
    ${pages.index ? sidePage('Indexer 页', pages.index, current.indexDtype, 'indexer') : ''}
    ${pages.index && pages.draft ? sidePage('Draft MLA 页', pages.draft, current.draftDtype, 'draft') : ''}
    <p data-byte-readout class="caption" aria-live="polite">点击任意分段查看页内字节偏移。</p>
  </div>`;
}

function showByteOffset(target) {
  const segment = target.closest('[data-byte-caption]');
  if (segment) {
    segment.closest('.block-zoom').querySelector('[data-byte-readout]').textContent = segment.dataset.byteCaption;
  }
  return Boolean(segment);
}

function renderTrace() {
  const pinnedAscend = 'https://github.com/Dawn952/vllm-ascend/blob/6fcafe90e9fe4091685935d162a7a3869aa204ee/vllm_ascend';
  const paired = 'https://github.com/Dawn952/vllm-ascend/blob/3634a7eae94897d1a7b14d00807323e5af6b4a04/vllm_ascend';
  const k3 = current.modelId === 'k3';
  byId('trace-content').innerHTML = `
    <ol class="trace-steps">
      <li><strong>调度器分配池块 ID</strong><p>按请求进度为各缓存组分配块，把请求块号和池块 ID 写进映射表。</p><a href="https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_manager.py#L550-L580"><code>allocate_slots</code></a></li>
      <li><strong>把这一步的 KV 写进页</strong><p>执行器用写入槽位（<code>slot_mapping</code>）定位当前 token 在本卡页内的位置。</p><a href="${k3 ? pinnedAscend + '/attention/mla_v1.py#L3298-L3356' : paired + '/attention/sfa_v1.py#L1765-L1820'}">Ascend KV 写入路径</a></li>
      <li><strong>注意力读取历史页</strong><p>读取时按块表（<code>block_table</code>）找到历史页，结合本步的序列长度执行注意力。${k3 ? '' : 'Indexer 先筛选参与稀疏注意力的历史位置。'}</p><a href="${k3 ? pinnedAscend + '/attention/mla_v1.py#L2740-L2986' : paired + '/attention/sfa_v1.py#L1723-L1764'}">Ascend 注意力读取路径</a></li>
    </ol>
    <p class="trace-state"><strong>KDA 的状态分支</strong> 读取卷积历史与上一份递推状态，计算并更新本步状态。<a href="${pinnedAscend}/ops/kimi_kda.py#L532-L628">KDA 状态读写</a></p>
    <p class="caption">链接展示对应源码分支。实际 kernel 由当前模型、dtype 和运行配置决定。</p>`;
}

byId('cache-layout').addEventListener('click', (event) => {
  const button = event.target.closest('.explanation-toggle');
  if (!button) return;
  const explanation = byId(button.getAttribute('aria-controls'));
  explanation.hidden = !explanation.hidden;
  button.setAttribute('aria-expanded', !explanation.hidden);
});

function selectMlaId(id) {
  const value = calculate(inputs[selected], current);
  selectedMlaId = Math.max(0, Math.min(id, value.historyIds - 1));
  renderLayoutVisual(value, true);
}

layoutVisual.addEventListener('keydown', (event) => {
  const button = event.target.closest('[data-mla-id]');
  if (!button) return;
  const id = Number(button.dataset.mlaId);
  const targets = {
    ArrowLeft: id - 1,
    ArrowRight: id + 1,
    Home: 0,
    End: Number(byId('logical-block-input').max),
  };
  if (!Object.hasOwn(targets, event.key)) return;
  event.preventDefault();
  selectMlaId(targets[event.key]);
});

layoutVisual.addEventListener('submit', (event) => {
  if (event.target.id !== 'block-search') return;
  event.preventDefault();
  selectMlaId(byId('logical-block-input').valueAsNumber);
});

layoutVisual.addEventListener('click', (event) => {
  if (showByteOffset(event.target)) return;
  if (event.target.closest('[data-last-block]')) {
    selectMlaId(Number(byId('logical-block-input').max));
    return;
  }
  const id = event.target.closest('[data-mla-id]');
  if (id) {
    selectMlaId(Number(id.dataset.mlaId));
    return;
  }
  const jump = event.target.closest('[data-block-group]');
  if (!jump) return;
  const scroller = layoutVisual.querySelector('.visual-block-scroll');
  const table = scroller.querySelector('table');
  const column = Number(jump.dataset.blockGroup) + 1;
  for (const cell of table.querySelectorAll('.visual-column-selected')) cell.classList.remove('visual-column-selected');
  for (const row of table.rows) row.cells[column].classList.add('visual-column-selected');
  const headers = table.tHead.rows[0].cells;
  scroller.scrollLeft = headers[column].offsetLeft - headers[0].offsetWidth;
});

layout.addEventListener('click', (event) => {
  if (!event.target.closest('[data-layout-panel="middle-slots"]')) return;
  showMiddleSlots = !showMiddleSlots;
  byId('middle-slots').hidden = !showMiddleSlots;
  byId('middle-slots-placeholder').hidden = showMiddleSlots;
  byId(showMiddleSlots ? 'collapse-middle-slots' : 'expand-middle-slots').focus();
});

function openDetail(row) {
  const config = current;
  const value = calculate(inputs[row], config);
  byId('detail-title').textContent = `${memory(value.bytes)} / 卡`;
  const draftLabel = value.pages.draft ? ` · 草稿 KV ${config.draftDtype}` : '';
  const indexLabel = value.pages.index ? ` · Indexer ${config.indexDtype}` : '';
  byId('detail-caption').textContent = `${modelChoice.selectedOptions[0].textContent} · ${hardware.selectedOptions[0].textContent} · ${lengths[row]} · ${topology(config)} · ${weightLabel(config.weight)} · ${describe(config, value)} · block ${blockInputPrefix(value)}${value.localBlock} token / 卡 · MLA ${config.mlaDtype}${indexLabel}${draftLabel}`;
  byId('detail-rows').innerHTML = `<p class="caption">单卡占用按模型估算。 ${evidenceMarkup(value)}。 ${lengthEvidence(row)}</p>
    ${allocationFormula(value)}
    <p class="caption">MLA 历史块数 = ⌈${fmt.format(value.limit)} token ÷ ${fmt.format(value.coverage)} token / 块⌉ = ${value.historyIds}。另外有 ${value.stateIds} 个池块 ID 存 KDA 状态${value.draftInfo?.ids ? `，${value.draftInfo.ids} 个存 GQA 草稿历史` : ''}。每个 ID 都预留一组层页，分项中列出的空层槽和空页也计入占用。</p>
    ${value.rows.map((entry) => `<details class="breakdown"><summary><span>${entry.name}</span><strong>${memory(entry.bytes)}</strong></summary><p>${entry.shape}</p><p><code>${entry.formula} = ${fmt.format(entry.bytes)} B</code></p><p class="tags">${entry.tags.map((tag) => `<span>${tag}</span>`).join('')}</p></details>`).join('')}`;
  detail.showModal();
}

results.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-row]');
  if (!button) return;
  selected = Number(button.dataset.row);
  updateSelection();
  if (button.classList.contains('value')) openDetail(selected);
});

mlaDraftLayers.replaceChildren(...k3Preset.mlaDraftLayerOptions.map(count => new Option(`${count} 层`, count)));
customDp.addEventListener('input', render);
customTp.addEventListener('change', () => {
  updateCustomDcp();
  render();
});
customDcp.addEventListener('change', render);
byId('copy-preset-topology').addEventListener('click', () => {
  customDp.value = '4';
  customTp.value = String(platforms[hardware.value].tp);
  updateCustomDcp();
  customDcp.value = String(presetDcp);
  render();
});
byId('copy-parallel-link').addEventListener('click', async () => {
  const url = new URL(location.href);
  url.hash = '';
  url.search = new URLSearchParams({
    model: current.modelId, platform: current.platform, weight: current.weight, scenario: 'custom',
    dp: current.dp, tp: current.tp, dcp: current.dcp, input: inputs[selected],
    block: current.requestedBlock, mla: current.mlaDtype, index: current.indexDtype, draftdtype: current.draftDtype,
    draft: Number(draft.checked), recover: Number(recover.checked), spec: spec.value,
    kind: current.draftKind, layers: current.mlaDraftLayerCount,
  }).toString();
  const link = byId('parallel-link-url');
  try {
    await navigator.clipboard.writeText(url.href);
    byId('parallel-link-status').textContent = '已复制';
  } catch {
    link.value = url.href;
    byId('parallel-link-fallback').hidden = false;
    link.select();
  }
});
byId('gqa-draft-layer-count').textContent = `${k3Preset.gqaDraftLayers} 层（固定）`;
for (const control of [weight, scenario, recover, draft, spec, blockSize, mlaDtype, indexDtype, draftDtype, mlaDraftLayers]) control.addEventListener('change', render);
for (const input of draftKinds) input.addEventListener('change', () => {
  draftSpecByKind[current.draftKind] = spec.value;
  spec.value = draftSpecByKind[input.value] ?? spec.value;
  render();
});
hardware.addEventListener('change', selectPlatform);
modelChoice.addEventListener('change', () => selectModel());
byId('reset').addEventListener('click', () => {
  selected = 3;
  selectedMlaId = 0;
  const scroller = layoutVisual.querySelector('.visual-block-scroll');
  if (scroller) scroller.scrollLeft = 0;
  selectModel();
});

const query = new URLSearchParams(location.search);
if (query.get('model') === 'k3') modelChoice.value = 'k3';
selectModel(query.get('platform'));
if (query.get('scenario') === 'custom') applySharedParallel(query);
