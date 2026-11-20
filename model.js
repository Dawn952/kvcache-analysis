(function (root) {
  'use strict';

  const Ki = 1024;
  const inputs = [4 * Ki, 16 * Ki, 64 * Ki, 128 * Ki, Ki * Ki - Ki];
  const defaultBlock = 1536;
  const platforms = {
    a5: { name: 'Ascend 950DT', weights: ['W4A8_MXFP', 'native'], tp: 8, dcp: 8, recover: true,
      blockOptions: [640, 1024, 1408, defaultBlock], defaultBlock, cacheDtype: 'FP8',
      dsaDtypes: ['FP8', 'INT8', 'BF16'], draftDtypes: ['BF16', 'FP8'] },
    a3: { name: 'Atlas 800I A3', weights: ['W4A8'], tp: 16, dcp: 16, recover: false,
      blockOptions: [768], defaultBlock: 768, cacheDtype: 'INT8',
      dsaDtypes: ['INT8', 'BF16'], draftDtypes: ['BF16', 'INT8'] },
  };
  const mlaKernelBlock = 128;
  const mlaFormats = {
    BF16: { selectBytes: 1152, tokenBytes: 1152, shape: '(512 latent + 64 RoPE) × BF16' },
    FP8: { selectBytes: 576, tokenBytes: 656, shape: '512 × FP8 + 64 × BF16 + 4 × FP32 scale' },
    INT8: { selectBytes: 576, tokenBytes: 656, shape: '512 × INT8 + 64 × BF16 + 4 × FP32 scale' },
  };
  const indexFormats = {
    BF16: { tokenBytes: 256, shape: '128 × BF16' },
    FP8: { tokenBytes: 132, shape: '128 × FP8 + 1 × FP32 scale' },
    INT8: { tokenBytes: 130, shape: '128 × INT8 + 1 × FP16 scale' },
  };
  const mainLayers = 24;
  const stateLayers = 69;
  const stateGroups = 3;
  const layersPerStateGroup = stateLayers / stateGroups;
  const draftLayers = 5;
  const dim = 128;

  function calculate(input, { platform = 'a5', weight = platforms[platform]?.weights[0], dcp, recover, spec,
    requestedBlock = platforms[platform]?.defaultBlock, mlaDtype = platforms[platform]?.cacheDtype,
    indexDtype = platforms[platform]?.cacheDtype, draftDtype = 'BF16' }) {
    const preset = platforms[platform];
    if (!preset || (dcp !== preset.dcp && !(platform === 'a5' && dcp === 1))) {
      throw new RangeError('不支持此机型和 DCP 组合');
    }
    if (!preset.weights.includes(weight)) throw new RangeError('不支持此机型和权重组合');
    if (!preset.dsaDtypes.includes(mlaDtype) || !preset.dsaDtypes.includes(indexDtype) || !preset.draftDtypes.includes(draftDtype)) {
      throw new RangeError('不支持此机型和 dtype 组合');
    }
    if (!Number.isInteger(spec) || spec < 0 || spec > 7) throw new RangeError('spec_num 须为 0 至 7 的整数');
    if (recover && spec === 0) throw new RangeError('RecoverSSM 需要开启 DSpark');
    if (!preset.blockOptions.includes(requestedBlock)) throw new RangeError('不支持此 block-size 档位');

    const heads = 96 / preset.tp;
    const recurrent = heads * dim * dim * 4;
    const conv = (3 + spec) * 3 * heads * dim * 2;
    const state = recurrent + conv;
    const mla = mlaFormats[mlaDtype];
    const indexer = indexFormats[indexDtype];
    const draftTokenBytes = draftDtype === 'BF16' ? 1152 : 576;
    // These presets are multiples of the 128-token backend block and the KDA checkpoint alignment.
    // Upstream sizes the manager block first. Ascend then uses the packed MLA width for its page stride.
    const upstreamQuantumBytes = requestedBlock * mla.selectBytes;
    const alignmentMultiple = Math.ceil(state / upstreamQuantumBytes);
    const localBlock = requestedBlock * alignmentMultiple;
    const limit = input + Ki;
    const coverage = localBlock * dcp;
    // Each row sets the engine max_model_len to input + output. allocate_slots
    // caps main tokens plus lookahead at that limit before full-attention paging.
    const historyIds = Math.ceil(limit / coverage);
    const stateCopies = 2 + (recover ? 0 : spec);
    const stateIds = stateCopies * stateGroups;
    const mainPage = localBlock * mla.tokenBytes;
    const indexPage = localBlock * indexer.tokenBytes;
    const draftPage = spec ? localBlock * draftTokenBytes : 0;
    const mainRegion = mainLayers * mainPage;
    const indexRegion = mainLayers * indexPage;
    const draftRegion = draftLayers * draftPage;
    const bytesPerId = mainRegion + indexRegion + draftRegion;
    const rows = [
      { name: 'DSA MLA KV', bytes: historyIds * mainRegion,
        shape: `${mla.shape} = ${mla.tokenBytes} B / token / 层`,
        formula: `${historyIds} 历史 ID × 24 层 × ${localBlock} 本地 token × ${mla.tokenBytes} B`, tags: ['DCP', 'DSpark'] },
      { name: 'DSA Indexer K', bytes: historyIds * indexRegion,
        shape: `${indexer.shape} = ${indexer.tokenBytes} B / token / 层`,
        formula: `${historyIds} 历史 ID × 24 层 × ${localBlock} 本地 token × ${indexer.tokenBytes} B`, tags: ['DCP', 'DSpark'] },
      { name: 'DSpark draft MLA KV', bytes: historyIds * draftRegion,
        shape: spec ? `(512 latent + 64 RoPE) × ${draftDtype} = ${draftTokenBytes} B / token / 层${draftDtype === 'BF16' ? '' : '（C8 容量估算，静态 scale 另计）'}` : 'DSpark 关闭，草稿缓存区域为 0 B',
        formula: `${historyIds} 历史 ID × ${spec ? 5 : 0} 层 × ${localBlock} 本地 token × ${draftTokenBytes} B`, tags: ['DCP', 'DSpark'] },
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
