# Coding agent 固定评测

这里的 `inspect`、`review`、`bugfix` fixture 用同一套
`scripts/agent.py` 工具和消息循环比较模型或后端。runner 每次把 fixture 复制到临时 Git
仓库，因此 review 和 bugfix 不会修改 qwen3x 工作区。

启动 qwen35 后运行：

```sh
python3 scripts/agent_eval.py all
python3 scripts/agent_eval.py inspect --url http://127.0.0.1:8002/v1/chat/completions \
  --model qwen3.5-0.8b-reference --backend transformers \
  --artifact build/models/Qwen3.5-0.8B --source Qwen/Qwen3.5-0.8B
```

也可以用 `make -C eval agent-eval`；`AGENT_SCENE`、`SERVER`、`EVAL_MODEL` 和
`AGENT_OUTPUT` 可覆盖默认值。结果默认写到被 Git 忽略的
`eval/results/agent/<UTC>/`：

- `manifest.json`：同次运行的配置、三个场景 verdict 和汇总指标。
- `<scene>/result.json`：最终回答、diff、测试结果、usage、TTFT 和错误。
- `<scene>/trace.json`：每轮完整 messages、原始响应、finish reason、tool call/result 和时延。

trace 包含 prompt、fixture 代码和工具输出，只能用于本地开发评测，不应写入生产日志或提交。

## pi 本地模型配置

pi 从 `~/.pi/agent/models.json` 读取 OpenAI-compatible 自定义模型。qwen35 的最小配置是：

```json
{
  "providers": {
    "qwen3x": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "qwen3.5-4b",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 40960,
          "maxTokens": 4096
        },
        {
          "id": "qwen3.5-9b",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 40960,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

`apiKey` 是让 pi 启用本地 provider 的占位值。如果 qwen35 通过 `QWEN_API_KEY` 启动，二者
应使用相同的真实值。固定 agent eval 仍显式关闭 thinking，避免能力比较混入不同推理预算；
pi 日常使用则按模型默认行为打开 thinking。
