# KVCache 对比台

直接在浏览器打开 `index.html`。静态页面无需安装依赖，计算在浏览器中完成。

[Kimi K3.1的KVCache布局原理计算](https://dawn952.github.io/kvcache-analysis/lbnhc.html)介绍 Kimi K3.1 的历史与状态缓存如何共用块池，含配图和离线复算附件。

Kimi-K3.1 Pro 在 Ascend 950DT 上可选 `DP4 / DCP8 / TP8` 或 `DP4 / TP8`。「单请求显存」工具展示当前配置的五档估算，输出长度均为 1024 token。可选项按 KDA、DSA、DSpark 分组，可切换 RecoverSSM、MLA dtype、Indexer dtype、DSpark、`spec_num` 和 draft dtype。非默认 dtype 组合按相同分组做容量推算。点表里的占用数字查看分项与公式，点长度档位切换布局图。

机型选择器包含 Atlas 800I A3 的 W4A8 场景。A3 采用公开 1M Decode 模板的 `DP4 / DCP16 / TP16`、`block-size 768` 和 DSpark3，主 MLA 与 Indexer 使用 INT8，draft MLA 使用 BF16。RecoverSSM 默认关闭，打开后的单卡占用按源码推算。

Kimi-K3 按 `LBNHC` 展示 BF16 KV。950DT 场景采用 DP4 / TP8、768 token / 卡，默认 3 层 MLA Draft。Atlas 800I A3 场景采用公开 1M Decode 模板的 DP4 / DCP16 / TP16、384 token / 卡和 5 层 MLA Draft、DSpark7，另可切到 DP4 / TP16。两种机型均可切换 3 / 5 层作容量对照。GQA Draft 采用 RadixArk 的 5 层配置，按上游独立缓存组推算。图中标出 KDA 和 GQA 的页内 padding、各组未使用的层槽与 MLA Draft 的存储行。

两个模型的所有机型和权重都可选择自定义 Decode 并行。一个实例按 4–8 台机器计算，950DT 每机 8 卡，A3 每机 16 卡。TP 可选 1、2、4、8、16、32，DP × TP 不能超过实例卡数，DCP 须整除 TP。K3 的 GQA Draft 自定义估算要求 TP≥4，TP32 可选 DCP1 或 DCP2，其余只支持 DCP1。可从预设填入并行值，或复制配置链接。自定义场景用于容量推算。

「单请求显存」与「请求容量」是两个独立的计算窗口，共用部署配置与当前请求长度。可填入 1–1,048,576 token 的总长度和其中的输出长度。容量工具输入每卡 KVCache、共享前缀比例和不同前缀数，给出每副本与整个实例的理论同时驻留条数。只有完整历史块计入共享，KDA 状态和独立 GQA Draft 块按请求计。

`--block-size` 默认传入 384，可选 768、1536 或自定义（128 的倍数，上限 1M token）。页面按固定版上游选块规则推算生效值，再按 Ascend 打包后的页宽计算显存。1536 token 的 DCP8 页几何有 A5 Atlas 历史日志。A3 的 paired 页几何按公开模板及源码复核。

每档以输入与输出长度之和设置 `max_model_len`。投机 lookahead 受 `KVCacheManager.allocate_slots` 的模型上限截断。KDA 估算单请求稳态 Decode 的驻留，每组计 1 个运行状态块，RecoverSSM 关闭时再加 `spec_num` 个投机状态块。跨块切换时的临时重叠、额外 Prefill checkpoint、历史驻留、全池 null ID、图内存、权重和引擎级 RecoverSSM records 另计。

950DT 可切换到 Moonshot 原生 MXFP4 checkpoint。原生档在沿用同一 KV dtype 和布局的假设下计算单请求 KVCache，实际加载及总 HBM 待验证。
