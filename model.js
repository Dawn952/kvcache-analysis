(function (root) {
  'use strict';

  const Ki = 1024;
  const inputs = [4 * Ki, 16 * Ki, 64 * Ki, 128 * Ki, Ki * Ki - Ki];
  const defaultBlock = 1536;
  const platforms = {
    a5: { name: 'Ascend 950DT', weight: 'W4A8_MXFP', tp: 8, dcp: 8, recover: true,
      blockOptions: [640, 1024, 1408, defaultBlock], defaultBlock, cacheDtype: 'FP8', indexScaleDtype: 'FP32', indexBytes: 132 },
    a3: { name: 'Atlas 800I A3', weight: 'W4A8', tp: 16, dcp: 16, recover: false,
      blockOptions: [768], defaultBlock: 768, cacheDtype: 'INT8', indexScaleDtype: 'FP16', indexBytes: 130 },
  };
  const mlaKernelBlock = 128;
  const upstreamMlaTokenBytes = 576;
  const packedMlaTokenBytes = 656;
  const mainLayers = 24;
  const stateLayers = 69;
  const stateGroups = 3;
  const layersPerStateGroup = stateLayers / stateGroups;
  const draftLayers = 5;
  const dim = 128;

  function calculate(input, { platform = 'a5', dcp, recover, spec, requestedBlock = platforms[platform]?.defaultBlock }) {
    const preset = platforms[platform];
    if (!preset || (dcp !== preset.dcp && !(platform === 'a5' && dcp === 1))) {
      throw new RangeError('不支持此机型和 DCP 组合');
    }
    if (!Number.isInteger(spec) || spec < 0 || spec > 7) throw new RangeError('numspec 须为 0 至 7 的整数');
    if (!preset.blockOptions.includes(requestedBlock)) throw new RangeError('不支持此 block-size 档位');

    const heads = 96 / preset.tp;
    const recurrent = heads * dim * dim * 4;
    const conv = (3 + spec) * 3 * heads * dim * 2;
    const state = recurrent + conv;
    // These presets are multiples of the 128-token backend block and the KDA checkpoint alignment.
    // Upstream sizes the manager block first. Ascend then uses the packed MLA width for its page stride.
    const upstreamQuantumBytes = requestedBlock * upstreamMlaTokenBytes;
    const alignmentMultiple = Math.ceil(state / upstreamQuantumBytes);
    const localBlock = requestedBlock * alignmentMultiple;
    const limit = input + Ki;
    const coverage = localBlock * dcp;
    const historyIds = Math.ceil((limit + spec) / coverage);
    const stateCopies = 2 + (recover ? 0 : spec);
    const stateIds = stateCopies * stateGroups;
    const mainPage = localBlock * packedMlaTokenBytes;
    const indexPage = localBlock * preset.indexBytes;
    const draftPage = spec ? localBlock * 1152 : 0;
    const mainRegion = mainLayers * mainPage;
    const indexRegion = mainLayers * indexPage;
    const draftRegion = draftLayers * draftPage;
    const bytesPerId = mainRegion + indexRegion + draftRegion;
    const rows = [
      { name: '主模型 Attention KV', bytes: historyIds * mainRegion,
        shape: `512 × ${preset.cacheDtype} + 64 × BF16 + 4 × FP32 scale = 656 B / token / 层`,
        formula: `${historyIds} 历史 ID × 24 层 × ${localBlock} 本地 token × 656 B`, tags: ['DCP', 'DSpark'] },
      { name: '主模型 Indexer K', bytes: historyIds * indexRegion,
        shape: `128 × ${preset.cacheDtype} + 1 × ${preset.indexScaleDtype} scale = ${preset.indexBytes} B / token / 层`,
        formula: `${historyIds} 历史 ID × 24 层 × ${localBlock} 本地 token × ${preset.indexBytes} B`, tags: ['DCP', 'DSpark'] },
      { name: 'DSpark draft MLA KV', bytes: historyIds * draftRegion,
        shape: spec ? '(512 latent + 64 RoPE) × BF16 = 1152 B / token / 层' : 'DSpark 关闭，草稿缓存区域为 0 B',
        formula: `${historyIds} 历史 ID × ${spec ? 5 : 0} 层 × ${localBlock} 本地 token × 1152 B`, tags: ['DCP', 'DSpark'] },
      { name: 'KDA 状态内容', bytes: stateLayers * stateCopies * state,
        shape: `[${heads},128,128] FP32 + [${3 + spec},3,${heads},128] BF16 = ${state} B / 层 / 份`,
        formula: `69 层 × ${stateCopies} 份 × (${recurrent} B recurrent + ${conv} B conv)`, tags: [`TP${preset.tp}`, 'DSpark', 'RecoverSSM'] },
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
    return { input, limit, coverage, historyIds, stateCopies, stateIds, bytesPerId, pages, regions, rows, bytes,
      requestedBlock, localBlock, upstreamQuantumBytes, alignmentMultiple };
  }

  const model = { Ki, inputs, defaultBlock, platforms, mlaKernelBlock, mainLayers, stateGroups, layersPerStateGroup, draftLayers, calculate };
  if (typeof module !== 'undefined') module.exports = model;
  else root.KVCache = model;
})(globalThis);
