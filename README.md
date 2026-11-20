# KVCache 对比台

直接在浏览器打开 `index.html`。静态页面无需安装依赖，计算在浏览器中完成。

首版包含 Kimi-K3.1 Pro 在 Ascend 950DT 上的 `DP4 / DCP8 / TP8` 与 `DP4 / TP8` 对照。五档请求使用相同的 1024 token 生成预算。配置控件支持 RecoverSSM、DSpark 及 `numspec` 即时重算。点表里的占用数字查看分项与公式，点输入档位切换布局图。

1536 token 本地页沿用 A5 Atlas 的历史场景。DP4 / TP8 使用相同页几何和无 DCP 计算，尚待部署校准。表中使用 KDA 两份基础状态页的规划上界，单请求按整池 ID 计费。具体口径和实现依据见页面下方「对齐原理与计算口径」。不包含全池 null ID、图内存、权重和引擎级 RecoverSSM records。
