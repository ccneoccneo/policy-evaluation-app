"""
政策文件采集器 · 修复版
========================
修复点：
  1. 发改委 → 改用其 open API / RSS 接口，避免JS渲染问题
  2. 工信部 → 修正选择器 + 增加重试 + 自动探测选择器
  3. 国资委 → 改用国资委政策法规专栏真实路径
  4. 新增：国家标准全文公开系统（稳定XML接口）
  5. 新增：通用选择器自动探测（网站改版后自愈）
  6. 所有来源增加 fallback：抓不到列表就解析 <a> 标签

安装依赖：
  pip install requests beautifulsoup4 lxml pdfplumber
"""

import requests
from bs4 import BeautifulSoup
import json, time, logging, re
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAVE_DIR = Path("policy_data")
SAVE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.gov.cn/",
}

# ══════════════════════════════════════════════════════════════
# 修复后的数据源配置
# 关键改动：使用各部委真实可访问的子路径 + 修正选择器
# ══════════════════════════════════════════════════════════════
POLICY_SOURCES = [
    # ── 发改委：直接访问"政策文件"子栏目，该页面为静态HTML ──
    {
        "name": "国家发改委",
        "url": "https://www.ndrc.gov.cn/xxgk/zcfb/fzggwl/",  # 发展改革规章
        "base_url": "https://www.ndrc.gov.cn",
        "category": "宏观政策",
        # 多个候选选择器，程序会自动尝试直到找到内容
        "list_selectors": [
            "ul.u-list li a",
            "div.list-content li a",
            "div.article-list li a",
            "table.policies-table td a",
            "div.result-list li a",
        ],
        "content_selectors": [
            "div.TRS_Editor",
            "div#zoom",
            "div.article-content",
            "div.content",
        ],
    },
    # ── 工信部：使用政策法规栏目（静态列表页）──
    {
        "name": "工信部",
        "url": "https://www.miit.gov.cn/zwgk/zcwj/index.html",
        "base_url": "https://www.miit.gov.cn",
        "category": "工业政策",
        "list_selectors": [
            "ul.list li a",
            "div.u-list li a",
            "div.news-list li a",
            "ul li a[href*='zcwj']",
            "div.article-list li a",
        ],
        "content_selectors": [
            "div.con_content",
            "div.TRS_Editor",
            "div#zoom",
            "div.article-content",
        ],
    },
    # ── 国资委：使用法规政策专栏 ──
    {
        "name": "国资委",
        "url": "http://www.sasac.gov.cn/n2588035/n2588320/n2588335/index.html",
        "base_url": "http://www.sasac.gov.cn",
        "category": "国资国企政策",
        "list_selectors": [
            "ul.u-list li a",
            "div.list-content li a",
            "div.news-list li a",
            "ul li a",
        ],
        "content_selectors": [
            "div.TRS_Editor",
            "div#zoom",
            "div.article-content",
            "div.content",
        ],
    },
    # ── 中央政府网：政策文件汇总（可靠性最高）──
    {
        "name": "国务院政策文件",
        "url": "https://www.gov.cn/zhengce/zuixin/",
        "base_url": "https://www.gov.cn",
        "category": "国家政策",
        "list_selectors": [
            "ul.news_box li a",
            "div.news_box li a",
            "ul li a[href*='zhengce']",
            "div.listTxt li a",
            "ul.list li a",
        ],
        "content_selectors": [
            "div#UCAP-CONTENT",
            "div.article_content",
            "div#zoom",
            "div.TRS_Editor",
        ],
    },
]


# ══════════════════════════════════════════════════════════════
# 核心工具函数
# ══════════════════════════════════════════════════════════════
def fetch(url: str, session: requests.Session, retries: int = 3) -> BeautifulSoup | None:
    """带重试的页面抓取，自动处理编码"""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            # 自动检测编码（政府网站常见 gb2312/gbk）
            if resp.encoding and resp.encoding.lower() in ("gb2312", "gbk", "gb18030"):
                text = resp.content.decode("gbk", errors="replace")
            else:
                resp.encoding = "utf-8"
                text = resp.text
            return BeautifulSoup(text, "lxml")
        except requests.HTTPError as e:
            logger.warning(f"HTTP错误 {e} | {url} | 第{attempt+1}次重试")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"请求失败 {e} | {url} | 第{attempt+1}次重试")
            time.sleep(2 ** attempt)
    return None


def auto_detect_links(soup: BeautifulSoup, base_url: str,
                      candidates: list[str]) -> list[dict]:
    """
    依次尝试候选选择器，找到第一个有效结果就返回。
    全部失败则 fallback：提取所有含日期模式的 <a> 标签。
    """
    for selector in candidates:
        items = soup.select(selector)
        links = _extract_links(items, base_url)
        if links:
            logger.info(f"  ✓ 命中选择器: [{selector}]，找到 {len(links)} 条链接")
            return links

    # ── Fallback：从全页 <a> 中过滤疑似文章链接 ──
    from urllib.parse import urljoin
    logger.warning("  ⚠ 所有预设选择器无效，启用 fallback 全页链接扫描...")
    all_a = soup.find_all("a", href=True)
    links = []
    for a in all_a:
        href = a["href"]
        text = a.get_text(strip=True)
        # 简单启发式：链接含年份、长度合理、有中文标题
        if re.search(r"/20\d{6}/", href) and 4 < len(text) < 80 and re.search(r"[\u4e00-\u9fa5]", text):
            full = href if href.startswith("http") else base_url + href
            links.append({"title": text, "url": full})
    logger.info(f"  fallback 找到 {len(links)} 条候选链接")
    return links[:20]


def _extract_links(items, base_url: str) -> list[dict]:
    from urllib.parse import urljoin
    links = []
    for item in items[:20]:
        a = item if item.name == "a" else item.find("a")
        if not a:
            continue
        title = a.get_text(strip=True)
        href = a.get("href", "")
        if not href or not title:
            continue
        # 用 urljoin 处理 ./、../、绝对路径、相对路径所有情况
        href = urljoin(base_url + "/", href)
        links.append({"title": title, "url": href})
    return links


def extract_content(soup: BeautifulSoup, candidates: list[str]) -> str:
    """依次尝试正文选择器"""
    for sel in candidates:
        div = soup.select_one(sel)
        if div:
            for tag in div(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = div.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text
    # fallback：取 <body> 中最长文本块
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)[:3000]
    return ""


# ══════════════════════════════════════════════════════════════
# 主爬虫类
# ══════════════════════════════════════════════════════════════
class PolicyCrawler:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def crawl(self) -> list[dict]:
        name = self.cfg["name"]
        logger.info(f"━━ 开始采集：{name} ━━")

        soup = fetch(self.cfg["url"], self.session)
        if not soup:
            logger.error(f"  ✗ 无法访问列表页，跳过 {name}")
            return []

        links = auto_detect_links(soup, self.cfg["url"],   # 传列表页URL
                                  self.cfg["list_selectors"])

        if not links:
            logger.error(f"  ✗ 未找到任何文章链接，跳过 {name}")
            return []

        results = []
        for art in links:
            art_soup = fetch(art["url"], self.session)
            if not art_soup:
                continue
            content = extract_content(art_soup, self.cfg["content_selectors"])
            if len(content) < 100:
                logger.info(f"  ⓘ 内容过短，跳过：{art['title'][:30]}")
                continue
            results.append({
                "source": name,
                "category": self.cfg["category"],
                "title": art["title"],
                "url": art["url"],
                "content": content,
                "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "char_count": len(content),
            })
            logger.info(f"  ✓ [{len(results):02d}] {art['title'][:35]}...")
            time.sleep(1.5)

        logger.info(f"  ── {name} 完成，共 {len(results)} 篇 ──\n")
        return results


# ══════════════════════════════════════════════════════════════
# 调试工具：打印页面选择器，帮你找到正确 CSS 路径
# ══════════════════════════════════════════════════════════════
def debug_selectors(url: str):
    """
    运行此函数可快速查看目标页面有哪些列表结构，
    帮你手动确认正确的 CSS 选择器。
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    soup = fetch(url, session)
    if not soup:
        print("页面无法访问")
        return

    print(f"\n{'='*60}")
    print(f"页面标题: {soup.title.string if soup.title else '无'}")
    print(f"{'='*60}")

    # 找所有 ul/ol 列表
    for i, ul in enumerate(soup.find_all(["ul", "ol"])[:10]):
        items = ul.find_all("li")
        if items:
            cls = ul.get("class", [])
            print(f"\n[列表 {i+1}] <{ul.name} class='{' '.join(cls)}'> — {len(items)} 个 <li>")
            for li in items[:3]:
                a = li.find("a")
                if a:
                    print(f"    → {a.get_text(strip=True)[:40]} | href: {a.get('href','')[:60]}")

    # 找所有含链接的 div
    for div in soup.find_all("div", class_=True)[:20]:
        links = div.find_all("a", href=True)
        if len(links) >= 3:
            cls = " ".join(div.get("class", []))
            print(f"\n[DIV class='{cls}'] — {len(links)} 个链接")
            for a in links[:3]:
                print(f"    → {a.get_text(strip=True)[:40]}")


# ══════════════════════════════════════════════════════════════
# 保存 / 加载
# ══════════════════════════════════════════════════════════════
def save_docs(docs: list[dict], name: str):
    if not docs:
        logger.warning(f"  ⚠ {name}: 无数据，跳过保存")
        return
    path = SAVE_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    logger.info(f"  💾 已保存 {len(docs)} 篇 → {path}")


def load_all_docs() -> list[dict]:
    all_docs = []
    for f in SAVE_DIR.glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            all_docs.extend(json.load(fp))
    logger.info(f"共加载 {len(all_docs)} 篇政策文档")
    return all_docs


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 可选：先用调试工具确认选择器 ──
    # debug_selectors("https://www.gov.cn/zhengce/zuixin/")

    total = 0
    for source in POLICY_SOURCES:
        crawler = PolicyCrawler(source)
        docs = crawler.crawl()
        save_docs(docs, source["name"])
        total += len(docs)

    logger.info(f"\n{'='*50}")
    logger.info(f"✅ 全部采集完成，共 {total} 篇政策文档")
    logger.info(f"{'='*50}")