# 私有模型对比聊天

服务器运行 `scripts/model_chat_server.py`，页面为 `scripts/model_chat.html`。仅监听服务器 `127.0.0.1:17860`，通过 SSH 隧道访问，不使用公网共享链接。使用现有 Python 环境和标准库 HTTP 服务，不安装额外 Web 依赖。该轻量服务仅面向单用户本机访问，不适合公网/多用户生产部署。

服务器项目根目录启动：

```bash
screen -L -Logfile outputs/logs/model-chat.log -dmS model-chat .venv/bin/python scripts/model_chat_server.py
```

本机 PowerShell 运行 `scripts/connect_model_chat.ps1`，保持终端连接，然后访问 http://127.0.0.1:17860/ 。若端口已被本次建立的隧道占用，不要重复启动。本机断网或关闭 SSH 后页面会失联；服务器服务仍在，重建隧道即可。

支持 Base、CPT 5%、SFT、DPO、GRPO，各自独立对话历史。选“全部五个模型”逐个回答相同问题，默认最多生成 192 token，可改为 384/512。上下文超过 2048 token 会明确提示，不静默截断。想做相同条件的单轮比较，先清空全部对话，并保持相同系统提示和输出长度。模型第一次访问时加载，五模型显存实测约 5.5GB。

聊天使用 BF16 基座及 adapter、贪心解码、repetition_penalty=1.05，并在 EOS 或 im_end 处停止；这与离线评测停止条件可能不同。Base 未经过指令微调；其他模型实测存在重复、乱码和偏题，不作后处理美化。

聊天内容只保留在页面内存，刷新会丢失，不持久化用户聊天。服务不修改训练数据、模型和 checkpoint；服务日志记录模型加载及异常，不记录 HTTP 请求内容。不要把这个端口改为公网监听。

2026-09-18：五模型各完成一次 32-token HTTP 推理测试，均成功返回；静态检查通过。已知页面为整条回复展示，不是逐 token 流式输出。
