# KVCache 对比台

直接在浏览器打开 `index.html`。静态页面无需安装依赖，计算在浏览器中完成。

Kimi-K3.1 Pro 在 Ascend 950DT 上可选 `DP4 / DCP8 / TP8` 或 `DP4 / TP8`。首页展示当前配置的五档估算，生成预算均为 1024 token。可选项按 KDA、DSA、DSpark 分组，可切换 RecoverSSM、MLA dtype、Indexer dtype、DSpark、`spec_num` 和 draft dtype。非默认 dtype 组合按相同分组做容量推算。点表里的占用数字查看分项与公式，点长度档位切换布局图。

机型选择器包含 Atlas 800I A3 的 W4A8 场景。A3 采用公开 1M Decode 模板的 `DP4 / DCP16 / TP16`、`block-size 768` 和 DSpark3，主 MLA 与 Indexer 使用 INT8，draft MLA 使用 BF16。RecoverSSM 默认关闭，打开后的单卡占用按源码推算。

Kimi-K3 按 `LBNHC` 展示 BF16 KV。950DT 场景采用 DP4 / TP8、768 token / 卡，默认 3 层 MLA Draft。Atlas 800I A3 场景采用公开 1M Decode 模板的 DP4 / DCP16 / TP16、384 token / 卡和 5 层 MLA Draft、DSpark7，另可切到 DP4 / TP16。两种机型均可切换 3 / 5 层作容量对照。GQA Draft 采用 RadixArk 的 5 层配置，按上游独立缓存组推算。图中标出 KDA 和 GQA 的页内 padding、各组未使用的层槽与 MLA Draft 的存储行。GQA 的 Ascend 布局待实测。

950DT 的 `--block-size` 可在 640、1024、1408 和 1536 之间切换，默认传入 1536。页面按固定版上游选块规则推算生效值，再按 Ascend 打包后的页宽计算显存。1536 token 的 DCP8 页几何有 A5 Atlas 历史日志。A3 的 paired 页几何按公开模板及源码复核。

每档以输入与生成预算之和设置 `max_model_len`。投机 lookahead 受 `KVCacheManager.allocate_slots` 的模型上限截断。KDA 估算单请求稳态 Decode 的驻留，每组计 1 个运行状态块，RecoverSSM 关闭时再加 `spec_num` 个投机状态块。跨块切换时的临时重叠、额外 Prefill checkpoint、历史驻留、全池 null ID、图内存、权重和引擎级 RecoverSSM records 另计。

950DT 可切换到 Moonshot 原生 MXFP4 checkpoint。原生档在沿用同一 KV dtype 和布局的假设下计算单请求 KVCache，实际加载及总 HBM 待验证。
