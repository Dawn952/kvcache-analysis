# KVCache 对比台

直接在浏览器打开 `index.html`。静态页面无需安装依赖，计算在浏览器中完成。

首版包含 Kimi-K3.1 Pro 在 Ascend 950DT 上的 `DP4 / DCP8 / TP8` 与 `DP4 / TP8` 对照。五档请求使用相同的 1024 token 生成预算。配置控件支持 RecoverSSM、DSpark 及 `numspec` 即时重算。点表里的占用数字查看分项与公式，点输入档位切换布局图。

`--block-size` 可在 640、1024、1408 和 1536 之间切换，默认传入 1536。页面按固定版上游选块规则推算生效值，再按 Ascend 打包后的页宽计算显存。1536 token 的 DCP8 页几何有 A5 Atlas 历史日志，其他页几何和无 DCP 对照按源码推算。表中使用 KDA 两份基础状态页的规划上界，单请求按整池 ID 计费。全池 null ID、图内存、权重和引擎级 RecoverSSM records 另计。
