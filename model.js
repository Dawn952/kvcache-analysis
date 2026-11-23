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
  const k3Preset = { mlaDraftLayerOptions: [3, 5], gqaDraftLayers: 5, gqaKvHeads: 16, gqaHeadDim: 64 };
  const k3Platforms = {
    a5: { blockSize: 768, dcpOptions: [1], spec: 3, mlaDraftLayers: 3, maxMlaSpec: 5 },
    a3: { blockSize: 384, dcpOptions: [16, 1], spec: 7, mlaDraftLayers: 5, maxMlaSpec: 7 },
  };
  const customTpOptions = (platform) => [1, 2, 4, 8, 16].filter(tp => tp <= (platforms[platform]?.tp ?? 0));
  const customDcpOptions = (platform, tp) => customTpOptions(platform).filter(dcp => tp > 0 && tp % dcp === 0);
  const validCustomParallel = (platform, dp, tp, dcp) => Number.isSafeInteger(dp) && dp > 0
    && customTpOptions(platform).includes(tp) && customDcpOptions(platform, tp).includes(dcp);
  const gqaFitsK3Page = (tp) => 2 * k3Preset.gqaKvHeads / tp * k3Preset.gqaHeadDim * 2 <= mlaFormats.BF16.tokenBytes;

  function kdaGeometry(tp, spec) {
    const heads = 96 / tp;
    const recurrent = heads * dim * dim * 4;
    const conv = (3 + spec) * 3 * heads * dim * 2;
    return { heads, recurrent, conv, state: recurrent + conv };
  }

  function alignBlock(requestedBlock, selectBytes, stateBytes) {
    // Each requested block is a multiple of the 128-token backend alignment.
    const upstreamQuantumBytes = requestedBlock * selectBytes;
    const alignmentMultiple = Math.ceil(stateBytes / upstreamQuantumBytes);
    return { localBlock: requestedBlock * alignmentMultiple, upstreamQuantumBytes, alignmentMultiple };
  }

  function calculate(input, { platform = 'a5', weight = platforms[platform]?.weights[0], dcp, recover, spec,
    customParallel = false, dp = 4, tp = platforms[platform]?.tp,
    requestedBlock = platforms[platform]?.defaultBlock, mlaDtype = platforms[platform]?.cacheDtype,
    indexDtype = platforms[platform]?.cacheDtype, draftDtype = 'BF16' }) {
    const preset = platforms[platform];
    const validParallel = customParallel
      ? validCustomParallel(platform, dp, tp, dcp)
      : preset && tp === preset.tp && (dcp === preset.dcp || (platform === 'a5' && dcp === 1));
    if (!validParallel) {
      throw new RangeError('不支持此机型和 DCP 组合');
    }
    if (!preset.weights.includes(weight)) throw new RangeError('不支持此机型和权重组合');
    if (!preset.dsaDtypes.includes(mlaDtype) || !preset.dsaDtypes.includes(indexDtype) || !preset.draftDtypes.includes(draftDtype)) {
      throw new RangeError('不支持此机型和 dtype 组合');
    }
    if (!Number.isInteger(spec) || spec < 0 || spec > 7) throw new RangeError('spec_num 须为 0 至 7 的整数');
    if (recover && spec === 0) throw new RangeError('RecoverSSM 需要开启 DSpark');
    if (!preset.blockOptions.includes(requestedBlock)) throw new RangeError('不支持此 block-size 档位');

    const { heads, recurrent, conv, state } = kdaGeometry(tp, spec);
    const mla = mlaFormats[mlaDtype];
    const indexer = indexFormats[indexDtype];
    const draftTokenBytes = draftDtype === 'BF16' ? 1152 : 576;
    // Upstream sizes the manager block first. Ascend then uses the packed MLA width for its page stride.
    const block = alignBlock(requestedBlock, mla.selectBytes, state);
    const { localBlock } = block;
    const limit = input + Ki;
    const coverage = localBlock * dcp;
    // Each row sets the engine max_model_len to input + output. allocate_slots
    // caps main tokens plus lookahead at that limit before full-attention paging.
    const historyIds = Math.ceil(limit / coverage);
    // Decode steady state holds one running state per group. Cross-block overlap is transient.
    const stateCopies = 1 + (recover ? 0 : spec);
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
        formula: `69 层 × ${stateCopies} 份 × (${recurrent} B recurrent + ${conv} B conv)`, tags: [`TP${tp}`, 'DSpark', 'RecoverSSM'] },
      { name: 'KDA 页内 padding', bytes: stateLayers * stateCopies * (mainPage - state),
        shape: `主 MLA 物理页 ${mainPage} B，KDA 原始状态 ${state} B`,
        formula: `69 层 × ${stateCopies} 份 × (${mainPage} − ${state}) B`, tags: ['DSpark', 'RecoverSSM'] },
      { name: 'KDA 池块在共用区的空层槽', bytes: stateIds * mainPage,
        shape: 'MLA / KDA 共用区为每个池块 ID 分配 24 层页。每个 KDA 组只使用其中 23 层，其余 1 层计入分配占用。',
        formula: `${stateIds} 状态 ID × (24 − 23) 槽 × ${mainPage} B` , tags: ['RecoverSSM'] },
      { name: spec ? 'KDA 池块的空 Indexer / Draft 页' : 'KDA 池块的空 Indexer 页', bytes: stateIds * (indexRegion + draftRegion),
        shape: `KDA 的池块 ID 在独立的 Indexer${spec ? ' 与 Draft' : ''} 区也有预留页。它们没有写入内容，仍计入分配占用。`,
        formula: `${stateIds} 状态 ID × (${indexRegion} B Indexer + ${draftRegion} B draft)`, tags: ['DSpark', 'RecoverSSM'] },
    ];
    const bytes = (historyIds + stateIds) * bytesPerId;
    if (rows.reduce((sum, row) => sum + row.bytes, 0) !== bytes) throw new Error('分项与整页计费不一致');
    const pages = { main: mainPage, index: indexPage, draft: draftPage, conv, recurrent, state, padding: mainPage - state };
    const regions = { main: mainRegion, index: indexRegion, draft: draftRegion };
    return { input, limit, coverage, historyIds, stateCopies, stateIds, bytesPerId, pages, regions, rows, bytes,
      requestedBlock, ...block };
  }

  function calculateK3(input, { platform = 'a5', recover, spec, draftKind = 'mla', customParallel = false,
    dp = 4, tp = platforms[platform]?.tp,
    dcp = k3Platforms[platform]?.dcpOptions[0], mlaDraftLayerCount = k3Platforms[platform]?.mlaDraftLayers }) {
    const preset = k3Platforms[platform];
    const validParallel = customParallel ? validCustomParallel(platform, dp, tp, dcp)
      : preset && tp === platforms[platform].tp && preset.dcpOptions.includes(dcp);
    if (!validParallel) throw new RangeError('不支持此 Kimi-K3 机型和并行组合');
    if (draftKind !== 'mla' && draftKind !== 'gqa') throw new RangeError('不支持此 Draft 类型');
    const gqa = draftKind === 'gqa';
    if (!gqa && !k3Preset.mlaDraftLayerOptions.includes(mlaDraftLayerCount)) throw new RangeError('MLA Draft 层数可选 3 或 5');
    if (!Number.isInteger(spec) || spec < 0 || spec > (gqa ? 7 : preset.maxMlaSpec) || (recover && !spec)) {
      throw new RangeError('不支持此 Kimi-K3 投机配置');
    }
    if (gqa && spec && !gqaFitsK3Page(tp)) throw new RangeError('GQA Draft 页超过 MLA 共用页宽');
    if (gqa && spec && customParallel && dcp !== 1) throw new RangeError('Kimi-K3 GQA Draft 的自定义并行要求 DCP=1');

    const requestedBlock = preset.blockSize;
    const { heads, conv, recurrent, state } = kdaGeometry(tp, spec);
    const tokenBytes = mlaFormats.BF16.tokenBytes;
    const block = alignBlock(requestedBlock, tokenBytes, state);
    const { localBlock } = block;
    const limit = input + Ki;
    const coverage = localBlock * dcp;
    const historyIds = Math.ceil(limit / coverage);
    const stateCopies = 1 + (recover ? 0 : spec);
    const stateIds = stateCopies * stateGroups;
    const mainPage = localBlock * tokenBytes;
    const padding = mainPage - state;
    const draftLayerCount = spec ? (gqa ? k3Preset.gqaDraftLayers : mlaDraftLayerCount) : 0;
    const gqaHeads = k3Preset.gqaKvHeads / tp;
    const draftTokenBytes = gqa ? 2 * gqaHeads * k3Preset.gqaHeadDim * 2 : tokenBytes;
    const draftPayload = localBlock * draftTokenBytes;
    const draftPadding = mainPage - draftPayload;
    const sharedDraftLayers = gqa ? 0 : draftLayerCount;
    const poolLayers = mainLayers + sharedDraftLayers;
    const draftIds = gqa && draftLayerCount ? historyIds : 0;
    const bytesPerId = poolLayers * mainPage;
    const draftInfo = draftLayerCount ? {
      kind: gqa ? 'GQA' : 'MLA', layers: draftLayerCount, ids: draftIds, payload: draftPayload,
      padding: draftPadding, startRow: gqa ? 0 : mainLayers,
    } : null;

    const rows = [
      { name: 'MLA 历史 KV', bytes: historyIds * mainLayers * mainPage,
        shape: '每 token 存 512 维 latent 与 64 维位置分量（BF16）',
        formula: `${historyIds} ID × ${mainLayers} 层 × ${localBlock} token × ${tokenBytes} B`, tags: dcp > 1 ? ['LBNHC', 'DCP'] : ['LBNHC'] },
      { name: 'KDA 状态内容', bytes: stateLayers * stateCopies * state,
        shape: `[${heads},128,128] FP32 + [${3 + spec},3,${heads},128] BF16`,
        formula: `${stateLayers} 层 × ${stateCopies} 份 × (${recurrent} + ${conv}) B`, tags: [`TP${tp}`, 'RecoverSSM'] },
      { name: 'KDA 页内 padding', bytes: stateLayers * stateCopies * padding,
        shape: 'KDA 页尾补齐到 MLA 的单层页宽',
        formula: `${stateLayers} 层 × ${stateCopies} 份 × (${mainPage} − ${state}) B`, tags: ['LBNHC'] },
      { name: 'KDA 池块在共用区的空层槽', bytes: stateIds * (poolLayers - layersPerStateGroup) * mainPage,
        shape: `每个 KDA block ID 用 ${layersPerStateGroup} 个槽，池中给它留 ${poolLayers} 个槽`,
        formula: `${stateIds} ID × (${poolLayers} − ${layersPerStateGroup}) 槽 × ${mainPage} B`, tags: ['LBNHC'] },
    ];
    if (draftInfo) {
      const ids = draftIds || historyIds;
      rows.push({ name: `DSpark ${draftInfo.kind} Draft 有效数据`, bytes: ids * draftLayerCount * draftPayload,
        shape: gqa ? `K / V 各存 ${gqaHeads} × ${k3Preset.gqaHeadDim} BF16 / token` : '每 token 存 512 维 latent 与 64 维位置分量（BF16）',
        formula: `${ids} ID × ${draftLayerCount} 层 × ${localBlock} token × ${draftTokenBytes} B`, tags: ['DSpark'] });
    }
    if (draftIds) {
      rows.push(
        { name: 'GQA Draft 页内 padding', bytes: draftIds * draftLayerCount * draftPadding,
          shape: 'K / V 有效数据之后补齐到公共页步长',
          formula: `${draftIds} ID × ${draftLayerCount} 层 × (${mainPage} − ${draftPayload}) B`, tags: ['推算'] },
        { name: 'GQA Draft ID 的空层槽', bytes: draftIds * (poolLayers - draftLayerCount) * mainPage,
          shape: `GQA Draft 组用前 ${draftLayerCount} 个槽，其余槽预留`,
          formula: `${draftIds} ID × (${poolLayers} − ${draftLayerCount}) 槽 × ${mainPage} B`, tags: ['推算'] },
      );
    }
    const bytes = (historyIds + stateIds + draftIds) * bytesPerId;
    if (rows.reduce((sum, row) => sum + row.bytes, 0) !== bytes) throw new Error('Kimi-K3 分项与整页计费不一致');
    const pages = { main: mainPage, index: 0, draft: draftInfo ? mainPage : 0, state, conv, recurrent, padding };
    return { input, limit, coverage, historyIds, stateCopies, stateIds, poolLayers, draftInfo, bytesPerId,
      pages, regions: { main: mainLayers * mainPage, index: 0, draft: sharedDraftLayers * mainPage }, rows, bytes,
      requestedBlock, ...block };
  }

  const model = { Ki, inputs, defaultBlock, platforms, mlaKernelBlock, mainLayers, stateGroups, layersPerStateGroup,
    draftLayers, k3Preset, k3Platforms, customTpOptions, customDcpOptions, validCustomParallel, gqaFitsK3Page, calculate, calculateK3 };
  if (typeof module !== 'undefined') module.exports = model;
  else root.KVCache = model;
})(globalThis);
