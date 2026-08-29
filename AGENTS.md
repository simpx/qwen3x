# qwen3x 协作约定

本文件适用于整个仓库。项目目标和目录结构见 `README.md`。

## 操作边界

- 除非用户明确要求修改，否则只解读和 review，不修改代码。
- 只有用户明确要求时才创建 commit。
- 修改后运行与风险相称的测试；基础 C++ 改动至少运行 `make test`，render 边界运行
  `make render-test`，HTTP 数据流运行 `make http-test`。
- 用户明确要求“直接 push main”时，先在已验证的 feature branch 上准备并检查提交，再将
  该提交 fast-forward 到本地 `main` 并推送 `origin/main`。若远端 main 无法 fast-forward，
  停止并向用户说明。

## 代码风格

- 使用 C-oriented、exception-free C++17：允许 namespace、class、RAII、`auto` 和简单
  lambda，不使用异常、RTTI、复杂模板或隐藏数据流的抽象。
- 模型计算优先使用显式数组、指针、循环和 shape；只做不影响阅读的性能优化。
- 可恢复的输入错误通过返回值报告；内部不变量失败时先输出上下文，再 assert/abort。
- Python 只用于 `scripts/`、`reference/` 和 `eval/`，不进入部署或推理数据流。

## 解读方式

- 使用中文，先回答当前问题，再补必要背景。
- 解释模型计算时标注 `H、I、D、T、V` 等 shape，并检查乘法维度能否相消。
- 明确区分数学矩阵 shape 与 checkpoint 的实际存储布局。
- review 新模块时优先说明：目的与数据流、shape/公式、对应 C++ 实现。
