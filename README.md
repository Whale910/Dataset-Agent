# 🤖 Dataset-Agent:维护数据集Agent 

### 🚀 项目简介

**Dataset-Agent** 是一个基于大模型（LLM）驱动的自动化数据集调研工具。它能自动检索 **arXiv** 和 **Hugging Face**，并检索对应 **GitHub** 仓库，最终为你生成一份包含数据集规格（条数、时长、开源协议）的结构化 JSON 报告。

------

### 📦 环境准备

1. **克隆项目**

   ```
   git clone https://github.com/Whale910/Dataset-Agent.git
   ```

2. **安装依赖**

   ```
   pip install -r requirements.txt
   ```

3. **配置密钥** 新建一个 `.env` 文件，并填入你的 API 密钥：

   ```
   OPENAI_API_KEY=your_deepseek_key
   OPENAI_BASE_URL=https://api.deepseek.com/v1
   HF_TOKEN=your_huggingface_token
   GITHUB_TOKEN=your_github_token
   ```

------

### 🛠️ 如何使用

只需一行命令，即可启动当天的调研流程：

```
python main.py
```

可以手动设置 max_results 和 limit 大小来调整每日需要往前检索多少篇文章。

**程序执行逻辑：**

1. 加载 `processed_ids.json`，跳过已读内容。
2. 抓取最新的 arXiv 论文和 HF 数据集。
3. 对每项新内容进行“GitHub 搜寻 -> README 读取 -> LLM 判定”。
4. 在控制台实时显示判定结果（✅ 接受 / ❌ 拒绝）。
5. 更新历史记录。

注：仓库中的示例代码主要适用于human motion和humanoid motion的数据集，其余数据集的维护可以通过修改提示词模版及部分代码来实现。

------

### 📊 输出结果

程序会根据当前日期自动生成 JSON 报告，例如 `2026-04-04.json`。：

```
{
    "name": "EgoNav",
    "is_open_source": "根据协议而定",
    "human_motion_count": "/",
    "humanoid_motion_count": "/",
    "video_duration": "5 hours",
    "data_origin": "动捕(MoCap)",
    "link": "https://egonav.weizhuowang.com",
    "is_locomotion": true,
    "reason": "接受依据：摘要明确提到系统使用5小时的人类行走数据来训练人形机器人导航，涉及全身运动如行走、穿越环境，且未涉及抓取、手部动作或物体交互等排除项。",
    "tags": []
}
```

