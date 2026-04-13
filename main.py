import os
import json
import time
import arxiv
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from huggingface_hub import list_datasets, hf_hub_download
import datetime
from typing import List, Optional
import re
import requests

# 1. 环境配置
load_dotenv() # 将.env文件加入运行时的环境变量中

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

HISTORY_FILE = "processed_ids.json" # 已检索过的文章/数据集
RESULTS_DIR = "results"

# 2. 数据结构定义
class MotionDataReport(BaseModel):
    name: str = Field(description="数据集名称")
    is_open_source: str = Field(default="/", description="开/闭源情况")
    human_motion_count: str = Field(default="/", description="human motion条数")
    humanoid_motion_count: str = Field(default="/", description="humanoid motion条数")
    video_duration: str = Field(default="/", description="视频总时长")
    data_origin: str = Field(default="/", description="数据来源(如动捕、视频抽取、整合已有)")
    link: str = Field(description="链接地址")
    is_locomotion: bool = Field(description="是否属于全身运动")
    reason: str = Field(description="中文理由")
    tags: List[str] = Field(default_factory=list) # 标签，如human motion，SMPL等


# 3. 辅助功能
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                return set(content) # 变成集合查找更快
            except:
                return set()
    return set()


def save_history(processed_ids):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_ids), f, ensure_ascii=False, indent=4)

def get_hf_readme(repo_id: str):
    """获取 HF 数据集的 README 前 2000 字"""
    try:
        # 下载 README.md 文件到本地缓存，在终端输入 huggingface-cli delete-cache，可以扫描并选择删除不用的缓存
        readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
            token=os.getenv("HF_TOKEN") # 在 .env 填入 HF_TOKEN 提高限制
        )
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read(2000)
            # 去掉过多的换行，方便 LLM 阅读
            return content.replace('\n', ' ').strip() + "..."
    except Exception:
        return "No README content available."

def get_github_readme(github_url: str):
    """提取 GitHub 链接并抓取其 README 内容"""
    try:
        # 提取user和repo
        clean_url = github_url.split('#')[0].rstrip('/')
        path_parts = clean_url.replace("https://github.com/", "").split('/')
        if len(path_parts) < 2: return ""

        user, repo = path_parts[0], path_parts[1]

        # 尝试常见的 README 分支 (main 或 master)
        for branch in ['main', 'master']:
            raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/README.md"
            resp = requests.get(raw_url, timeout=10)
            if resp.status_code == 200:
                return f"\n[GitHub README 选段]:\n{resp.text[:2000]}"
        return ""
    except Exception as e:
        print(f"⚠️ GitHub 读取失败: {e}")
        return ""

def extract_github_link(text: str):
    pattern = r'https?://github\.com/[a-zA-Z0-9\-_.]+/[a-zA-Z0-9\-_.]+'
    match = re.search(pattern, text)
    return match.group(0) if match else None


def search_github_by_title(title: str):
    """通过论文标题在 GitHub 搜索对应的仓库"""
    try:
        # 清洗标题，去掉特殊字符，保留核心词
        clean_title = re.sub(r'[^\w\s]', '', title)
        # 加上 GitHub Token 提高频率限制
        headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"} if os.getenv('GITHUB_TOKEN') else {}

        search_url = "https://api.github.com/search/repositories"
        # 搜索参数：按相关性排序
        params = {"q": clean_title, "sort": "stars", "order": "desc"}

        resp = requests.get(search_url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data['total_count'] > 0:
                # 如果标题里的核心词在仓库名或描述里，获取第一个搜索结果（通常是最相关的）
                top_repo = data['items'][0]
                return top_repo['html_url']
        return None
    except Exception as e:
        print(f"⚠️ GitHub 搜索失败: {e}")
        return None

# 4. 爬取数据
def fetch_arxiv_papers(query='(humanoid OR "human motion") AND dataset', max_results=5):
    results = []
    arxiv_client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3) # 每3s爬取一次，防止被封禁ip地址
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)

    try:
        for result in arxiv_client.results(search):
            results.append({
                "id": result.entry_id,
                "title": result.title,
                "summary": result.summary,
                "link": result.entry_id,
                "source": "arXiv"
            })
    except Exception as e:
        print(f"❌ arXiv 获取失败: {e}")
    return results


def fetch_hf_datasets(limit=5):
    keywords = ['human motion', 'humanoid motion']
    unique_datasets = {}

    try:
        for kw in keywords:
            batch = list_datasets(
                search=kw,
                limit=limit,
                sort="lastModified"
            )

            for ds in batch:
                if ds.id not in unique_datasets:
                    unique_datasets[ds.id] = ds

        results = []
        for ds_id, ds in unique_datasets.items():
            readme_text = get_hf_readme(ds_id)

            results.append({
                "id": ds_id,
                "title": ds_id,
                "summary": readme_text,
                "link": f"https://huggingface.co/datasets/{ds_id}",
                "source": "Hugging Face"
            })
        return results

    except Exception as e:
        print(f"❌ Hugging Face 获取失败: {e}")
        return []

# 5. 决策模块
def filter_data_with_llm(raw_data):
    prompt = f"""
    你是一个机器人学专家。请分析以下内容（包含论文摘要及可能的 GitHub README 选段）并输出 JSON。

    判定规则：
    1. 判断是否属于 Locomotion (全身运动如走、跑、跳、平衡)。
    2. **排除项 (Manipulation)**：如果涉及抓取、手部动作、物体交互，is_locomotion 必须为 false。
    3. **禁止捏造数据**：如果摘要中没有明确提到条数、时长或来源，对应的字段必须填 "/"。
    4. **语言要求**：输出的 "reason" 字段必须使用中文编写。
    5. **接受要求**：如果 is_locomotion 为 true，请简洁地说明接受依据。
    6. **拒绝理由**：如果 is_locomotion 为 false，你必须在 "reason" 字段中用中文明确说明拒绝原因。
    
    特别注意：
    - 如果内容中包含 [GitHub README 选段]，请优先从 README 中寻找数据集的条数、时长和开源协议，那里的信息通常比论文摘要更准确。
    - 如果通过 README 发现该仓库其实只是个“占位符”（即还没有上传代码或数据），请在理由中说明并考虑拒绝。

    JSON 字段提取要求：
    - name: 数据集简称或标题
    - is_open_source: 开源、闭源、或根据协议而定
    - human_motion_count: 找到具体的条数或样本量（如 11k, 3911等）
    - humanoid_motion_count: 针对人形机器人的数据条数
    - video_duration: 视频总时长（如 40h, 11.2s等）
    - data_origin: 动捕(MoCap)、视频抽取、数据整合等
    - link: 直接使用原数据提供的链接
    - is_locomotion: true/false
    - reason: 判定理由或拒绝理由（中文）

    待分析内容：
    标题: {raw_data['title']}
    摘要: {raw_data['summary']}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 格式的科研助理。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}  # 强制 DeepSeek 返回 JSON
        )

        content = response.choices[0].message.content
        # 使用 Pydantic 的 model_validate_json 将字符串转为对象
        return MotionDataReport.model_validate_json(content)

    except Exception as e:
        print(f"⚠️ 判定失败: {e}")
        return None


import subprocess


def git_push_results(report_path):
    """自动化提交并推送结果到 GitHub"""
    try:
        subprocess.run(["git", "add", HISTORY_FILE, report_path], check=True)

        file_name = os.path.basename(report_path)
        commit_message = f"Auto-update: {file_name} report"

        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)

        print(f"🚀 结果已成功推送至 GitHub: {commit_message}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git 推送失败: {e}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    processed_ids = load_history()
    print(f"🔄 已加载历史记录：{len(processed_ids)} 条")

    print("🔎 正在从 arXiv 和 Hugging Face 搜寻新内容...")
    raw_list = fetch_arxiv_papers(max_results=10) + fetch_hf_datasets(limit=10)

    if not raw_list:
        print("📭 未获取到新数据。")
        return

    today_analysis_results = []

    for item in raw_list:
        if item['id'] in processed_ids: continue
        processed_ids.add(item['id'])

        if item['source'] == "arXiv":
            # 先看摘要里有没有
            gh_link = extract_github_link(item['summary'])

            # 如果摘要没有，主动去搜标题
            if not gh_link:
                gh_link = search_github_by_title(item['title'])

            if gh_link:
                print(f"✨ 关联到仓库: {gh_link}")
                gh_readme = get_github_readme(gh_link)
                if gh_readme:
                    item['summary'] = f"[论文摘要]: {item['summary']}\n{gh_readme}"
                    item['link'] = gh_link  # 优先保存 GitHub 链接

        result = filter_data_with_llm(item)

        if result:
            # 无论 True 还是 False，都存入今日结果集，以便查看理由
            today_analysis_results.append(result)

            if result.is_locomotion:
                print(f"✅ [接受] {result.name} | 原因: {result.reason[:]}")
            else:
                print(f"❌ [拒绝] {result.name} | 原因: {result.reason[:]}")

    save_history(processed_ids)

    # 保存报告逻辑
    if today_analysis_results:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        report_filename = f"{date_str}.json"
        report_path = os.path.join(RESULTS_DIR, report_filename)

        report_data = [m.model_dump() for m in today_analysis_results]
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)

        print(f"\n📊 今日报告已生成: {report_path}")

        print("📤 正在同步至远端仓库...")
        git_push_results(report_path)
    else:
        print("\n📭 今日无新增内容，跳过推送。")

if __name__ == "__main__":
    main()
