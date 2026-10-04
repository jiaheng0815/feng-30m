# feng-30m v1（stageB 最终检查点 · 对话微调前）

这是 v1 流水线里 **stageB 结束后、对话微调之前**的权重（Qwen3 架构：8 层 / hidden 448 /
7 头（7 个 KV 头，MHA）/ FFN 896 / tied embedding / 32k 词表，30.75M 参数）。

它主要用于对比和续训起点，**不是**推荐的对话模型：

> 权重下载：见 [Releases](https://github.com/jiaheng0815/feng-30m/releases)。当前 Release 发布 **PC v3.14 + 板端 v3.16-embed**（v3.17 引擎）；
> v1 作为历史对照保留在文档与 `student/` 的训练记录里，未随包发布。

- 推荐使用 `student/feng-30m-chat/`（对话微调后，非打包 SFT，效果更稳）
- 32k 输入版见 `student/feng-30m-32k/`
- 评测、体积、速度等数据见项目根目录 `CHANGELOG.md` 的 v1 小节

对话格式同 v1 其它版本：`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n…<|im_end|>`。
