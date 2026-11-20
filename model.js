(function (root) {
  'use strict';

  const Ki = 1024;
  const inputs = [4 * Ki, 16 * Ki, 64 * Ki, 128 * Ki, Ki * Ki - Ki];
  const localBlock = 1536;
  const mainLayers = 24;
  const stateLayers = 69;
  const stateGroups = 3;
  const layersPerStateGroup = stateLayers / stateGroups;
  const draftLayers = 5;
  const heads = 12;
  const dim = 128;

  function calculate(input, { dcp, recover, spec }) {
    if (![1, 8].includes(dcp) || !Number.isInteger(spec) || spec < 0 || spec > 7) {
      throw new RangeError('DCP 须为 1 或 8，numspec 须为 0 至 7 的整数');
    }
    const limit = input + Ki;
    const coverage = localBlock * dcp;
    const historyIds = Math.ceil((limit + spec) / coverage);
    const stateCopies = 2 + (recover ? 0 : spec);
    const stateIds = stateCopies * stateGroups;
    const mainPage = localBlock * 656;
    const indexPage = localBlock * 132;
    const draftPage = spec ? localBlock * 1152 : 0;
    const mainRegion = mainLayers * mainPage;
    const indexRegion = mainLayers * indexPage;
    const draftRegion = draftLayers * draftPage;
    const bytesPerId = mainRegion + indexRegion + draftRegion;
    const recurrent = heads * dim * dim * 4;
    const conv = (3 + spec) * 3 * heads * dim * 2;
    const state = recurrent + conv;
    const rows = [
      { name: '主模型 Attention KV', bytes: historyIds * mainRegion,
        shape: '512 × FP8 + 64 × BF16 + 4 × FP32 scale = 656 B / token / 层',
        formula: `${historyIds} 历史 ID × 24 层 × ${localBlock} 本地 token × 656 B`, tags: ['DCP', 'DSpark'] },
      { name: '主模型 Indexer K', bytes: historyIds * indexRegion,
        shape: '128 × FP8 + 1 × FP32 scale = 132 B / token / 层',
        formula: `${historyIds} 历史 ID × 24 层 × ${localBlock} 本地 token × 132 B`, tags: ['DCP', 'DSpark'] },
      { name: 'DSpark draft MLA KV', bytes: historyIds * draftRegion,
        shape: spec ? '(512 latent + 64 RoPE) × BF16 = 1152 B / token / 层' : 'DSpark 关闭，草稿缓存区域为 0 B',
        formula: `${historyIds} 历史 ID × ${spec ? 5 : 0} 层 × ${localBlock} 本地 token × 1152 B`, tags: ['DCP', 'DSpark'] },
      { name: 'KDA 状态内容', bytes: stateLayers * stateCopies * state,
        shape: `[12,128,128] FP32 + [${3 + spec},3,12,128] BF16 = ${state} B / 层 / 份`,
        formula: `69 层 × ${stateCopies} 份 × (${recurrent} B recurrent + ${conv} B conv)`, tags: ['TP8', 'DSpark', 'RecoverSSM'] },
      { name: 'KDA 页内 padding', bytes: stateLayers * stateCopies * (mainPage - state),
        shape: `主 MLA 物理页 ${mainPage} B，KDA 原始状态 ${state} B`,
        formula: `69 层 × ${stateCopies} 份 × (${mainPage} − ${state}) B`, tags: ['DSpark', 'RecoverSSM'] },
      { name: 'KDA ID 的空 main 槽', bytes: stateIds * mainPage,
        shape: '69 层分成三组，每组 23 层。每个对应 ID 的 main 区有 24 槽',
        formula: `${stateIds} 状态 ID × (24 − 23) 槽 × ${mainPage} B` , tags: ['RecoverSSM'] },
      { name: 'KDA ID 的空 Indexer / draft 区', bytes: stateIds * (indexRegion + draftRegion),
        shape: '同一个池 ID 在独立的 Indexer 区与 draft 区也有物理槽位',
        formula: `${stateIds} 状态 ID × (${indexRegion} B Indexer + ${draftRegion} B draft)`, tags: ['DSpark', 'RecoverSSM'] },
    ];
    const bytes = (historyIds + stateIds) * bytesPerId;
    if (rows.reduce((sum, row) => sum + row.bytes, 0) !== bytes) throw new Error('分项与整页计费不一致');
    const pages = { main: mainPage, index: indexPage, draft: draftPage, conv, recurrent, state, padding: mainPage - state };
    const regions = { main: mainRegion, index: indexRegion, draft: draftRegion };
    return { input, limit, coverage, historyIds, stateCopies, stateIds, bytesPerId, pages, regions, rows, bytes };
  }

  const model = { Ki, inputs, localBlock, mainLayers, stateGroups, layersPerStateGroup, draftLayers, calculate };
  if (typeof module !== 'undefined') module.exports = model;
  else root.KVCache = model;
})(globalThis);
