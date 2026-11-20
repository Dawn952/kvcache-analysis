# KVCache 对比台

直接在浏览器打开 `index.html`。静态页面无需安装依赖，计算在浏览器中完成。

首版包含 Kimi-K3.1 Pro 在 Ascend 950DT 上的 `DP4 / DCP8 / TP8` 与 `DP4 / TP8` 对照。五档请求使用相同的 1024 token 生成预算。配置控件支持 RecoverSSM、DSpark 及 `numspec` 即时重算。点表里的占用数字查看分项与公式，点输入档位切换布局图。

机型选择器包含 Atlas 800I A3 的 W4A8 场景。A3 采用公开 1M Decode 模板的 `DP4 / DCP16 / TP16`、`block-size 768` 和 DSpark3，主 MLA 与 Indexer 使用 INT8，draft MLA 使用 BF16。RecoverSSM 默认关闭，打开后的单卡占用按源码推算。

950DT 的 `--block-size` 可在 640、1024、1408 和 1536 之间切换，默认传入 1536。页面按固定版上游选块规则推算生效值，再按 Ascend 打包后的页宽计算显存。1536 token 的 DCP8 页几何有 A5 Atlas 历史日志。A3 的 paired 页几何按公开模板及源码复核。

每档以输入与生成预算之和设置 `max_model_len`。投机 lookahead 受 `KVCacheManager.allocate_slots` 的模型上限截断。KDA 采用两个基础状态页加投机副本的规划口径。额外 Prefill checkpoint、全池 null ID、图内存、权重和引擎级 RecoverSSM records 另计。跨机型列使用另一机型整套默认配置，差值以当前选项为基准。
