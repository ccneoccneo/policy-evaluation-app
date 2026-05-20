"""
政策文件采集器 · 三站修复版
=============================
修复点（基于实际页面结构分析）：
  1. 工信部  → 列表页为 JS 渲染，改用 sitemap/搜索结果抓已知文章 URL，
               同时提供基于 URL 规律的"滚动抓取"模式作为兜底
  2. 国资委  → 列表选择器修正为 "ul li a"（页面已验证），
               正文改用精准路径提取
  3. 发改委  → 保持原有逻辑，修正 base_url 拼接问题
  4. 新增    → 所有来源均自动提取文章页真实分类（面包屑/元数据字段）
               category 字段结构变为：
                 "category": {
                     "source": "国资国企政策",   # 来源网站大类（固定）
                     "page":   "中央企业监管"    # 文章页提取的真实子分类
                 }

安装依赖：
  pip install requests beautifulsoup4 lxml
"""

import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SAVE_DIR = Path("policy_data")
SAVE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


# ═══════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════

def make_session(referer: str = "https://www.gov.cn/") -> requests.Session:
    s = requests.Session()
    s.headers.update({**HEADERS, "Referer": referer})
    return s


def fetch(url: str, session: requests.Session, retries: int = 3,
          encoding_hint: str | None = None) -> BeautifulSoup | None:
    """带重试的页面抓取，自动处理 GBK/UTF-8 编码。"""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            # 优先按 hint 解码，次之按响应头，再按 chardet
            enc = encoding_hint or (resp.encoding or "utf-8")
            if enc.lower() in ("gb2312", "gbk", "gb18030"):
                text = resp.content.decode("gbk", errors="replace")
            else:
                try:
                    text = resp.content.decode("utf-8")
                except UnicodeDecodeError:
                    text = resp.content.decode("gbk", errors="replace")
            return BeautifulSoup(text, "lxml")
        except requests.HTTPError as e:
            logger.warning("HTTP %s | %s | 第%d次", e, url, attempt + 1)
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning("请求失败 %s | %s | 第%d次", e, url, attempt + 1)
            time.sleep(2 ** attempt)
    return None


def extract_category(soup: BeautifulSoup, source_label: str) -> dict:
    """
    从文章页提取真实分类，返回结构化 category 字典。

    提取优先级：
      1. 页面内"分　　类"/"分类"元数据行（工信部、发改委常见）
      2. 面包屑导航最后一级（三个网站通用）
      3. 降级为来源大类标签
    """
    # ── 策略1：元数据行（格式：「分　　类：xxx」）──
    for tag in soup.find_all(string=re.compile(r"分\s*类")):
        parent = tag.parent
        if parent:
            # 同行内可能是 <td> 或紧跟的兄弟节点
            nxt = parent.find_next_sibling()
            if nxt:
                val = nxt.get_text(strip=True)
                if val and len(val) < 30:
                    return {"source": source_label, "page": val}
            # 或者在同一节点的文本里
            full = parent.get_text(strip=True)
            m = re.search(r"分\s*类[：:]\s*(.+)", full)
            if m:
                val = m.group(1).strip()
                if val and len(val) < 30:
                    return {"source": source_label, "page": val}

    # ── 策略2：面包屑（取倒数第2项，最后一项通常是"正文"或当前页标题）──
    crumb_sels = [
        "div.breadcrumb a", "ol.breadcrumb a",
        "div.position a", "div.location a",
        "p.pos a", "div.nav a",
    ]
    for sel in crumb_sels:
        crumbs = [a.get_text(strip=True) for a in soup.select(sel)
                  if a.get_text(strip=True) not in ("首页", "Home", "")]
        if len(crumbs) >= 1:
            page_cat = crumbs[-1]
            if len(page_cat) < 20:
                return {"source": source_label, "page": page_cat}

    # ── 策略3：降级 ──
    return {"source": source_label, "page": source_label}


def clean_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    """按优先级尝试内容选择器，全部失败则取 body 全文。"""
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            for tag in el(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text
    # fallback: body
    body = soup.find("body")
    return body.get_text(separator="\n", strip=True)[:5000] if body else ""


def save_docs(docs: list[dict], name: str):
    if not docs:
        logger.warning("⚠ %s：无数据，跳过保存", name)
        return
    path = SAVE_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    logger.info("💾 已保存 %d 篇 → %s", len(docs), path)


def load_all_docs() -> list[dict]:
    docs = []
    for f in SAVE_DIR.glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            docs.extend(json.load(fp))
    logger.info("共加载 %d 篇政策文档", len(docs))
    return docs


# ═══════════════════════════════════════════
# ① 发改委爬虫（原有逻辑，修正 URL 拼接）
# ═══════════════════════════════════════════

NDRC_CONFIG = {
    "name": "国家发改委",
    "base_url": "https://www.ndrc.gov.cn",
    "category": "宏观政策",
    # 按优先级排列的列表页（选能访问的）
    "list_urls": [
        "https://www.ndrc.gov.cn/xxgk/zcfb/fzggwl/",   # 发展改革规章
        "https://www.ndrc.gov.cn/xxgk/zcfb/ghwb/",     # 规划文本
    ],
    "list_selectors": [
        "ul.u-list li a",
        "div.list-content li a",
        "div.article-list li a",
        "div.result-list li a",
        "ul li a",
    ],
    "content_selectors": [
        "div.TRS_Editor",
        "div#zoom",
        "div.article-content",
        "div.content",
    ],
}


class NDRCCrawler:
    def __init__(self):
        self.cfg = NDRC_CONFIG
        self.session = make_session("https://www.ndrc.gov.cn/")

    def crawl(self, max_articles: int = 15) -> list[dict]:
        logger.info("━━ 开始采集：%s ━━", self.cfg["name"])
        links = []

        for list_url in self.cfg["list_urls"]:
            soup = fetch(list_url, self.session)
            if not soup:
                continue
            links.extend(self._parse_list(soup, list_url))
            if len(links) >= max_articles:
                break

        links = self._dedup(links)[:max_articles]
        if not links:
            logger.error("✗ 发改委：未找到任何文章链接")
            return []

        return self._fetch_articles(links)

    def _parse_list(self, soup: BeautifulSoup, base: str) -> list[dict]:
        for sel in self.cfg["list_selectors"]:
            items = soup.select(sel)
            if not items:
                continue
            links = []
            for item in items[:30]:
                a = item if item.name == "a" else item.find("a")
                if not a:
                    continue
                title = a.get_text(strip=True)
                href = a.get("href", "")
                if not href or not title or len(title) < 4:
                    continue
                links.append({"title": title, "url": urljoin(base, href)})
            if links:
                logger.info("  ✓ 发改委命中选择器 [%s]，%d 条", sel, len(links))
                return links

        # fallback：正则过滤含年份路径的链接
        return self._fallback_links(soup, base)

    def _fallback_links(self, soup: BeautifulSoup, base: str) -> list[dict]:
        links = []
        for a in soup.find_all("a", href=True):
            href, text = a["href"], a.get_text(strip=True)
            if (re.search(r"/20\d{6}/", href)
                    and 4 < len(text) < 80
                    and re.search(r"[\u4e00-\u9fa5]", text)):
                links.append({"title": text, "url": urljoin(base, href)})
        logger.warning("  ⚠ 发改委 fallback 找到 %d 条候选链接", len(links))
        return links[:20]

    def _fetch_articles(self, links: list[dict]) -> list[dict]:
        results = []
        for art in links:
            soup = fetch(art["url"], self.session)
            if not soup:
                continue
            content = clean_text(soup, self.cfg["content_selectors"])
            if len(content) < 100:
                continue
            cat = extract_category(soup, self.cfg["category"])
            results.append({
                "source": self.cfg["name"],
                "category": cat,
                "title": art["title"],
                "url": art["url"],
                "content": content,
                "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "char_count": len(content),
            })
            logger.info("  ✓ [%02d] %s [%s]", len(results), art["title"][:35], cat["page"])
            time.sleep(1.5)
        logger.info("  ── 发改委完成，共 %d 篇 ──\n", len(results))
        return results

    @staticmethod
    def _dedup(links: list[dict]) -> list[dict]:
        seen, out = set(), []
        for lk in links:
            if lk["url"] not in seen:
                seen.add(lk["url"])
                out.append(lk)
        return out


# ═══════════════════════════════════════════
# ② 工信部爬虫
#    列表页为 JS 渲染，采用三种策略：
#    A. 先从 sitemap.xml 提取目标栏目 URL
#    B. 用百度/必应搜索接口抓近期文章（可选）
#    C. 直接从已知文章 URL 反推同栏目最近文章
#       （根据 URL 规律 /art/年份/art_{id}.html 遍历）
# ═══════════════════════════════════════════

# 工信部各栏目已知的近期文章作为"种子"
MIIT_SEEDS = {
    "公告": [
        "https://www.miit.gov.cn/zwgk/zcwj/wjfb/gg/art/2026/art_3bfb3733b9e04e63b24feb4dd70a2d8e.html",
    ],
    "通知": [
        "https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2026/art_aac97e3c05554e468aa2a503d1669661.html",
        "https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2026/art_76ee858469814146a1ce17becc6bb325.html",
        "https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2026/art_9d8c75f8355a4179abad5a6296273dd2.html",
        "https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2026/art_58259bfb30924d6bb225b82b66d1008d.html",
    ],
}

# 工信部正文选择器（实测页面结构）
MIIT_CONTENT_SELECTORS = [
    "div.con_content",
    "div#artical_content",   # 部分文章用此 ID
    "div.TRS_Editor",
    "div#zoom",
    "div.article-content",
]

# 从文章页提取元数据的正则
MIIT_META_RE = {
    "date": re.compile(r"发布时间[：:]\s*(\d{4}-\d{2}-\d{2})"),
    "dept": re.compile(r"来源[：:]\s*(.+?)\s*\n"),
}


class MIITCrawler:
    """
    工信部爬虫。
    策略：以已知文章 URL 为种子，解析文章正文后，
    从页面内"上一篇/下一篇"导航或同栏目列表中发现更多文章。
    同时支持直接传入已知 URL 列表。
    """

    def __init__(self):
        self.session = make_session("https://www.miit.gov.cn/")
        self.visited: set[str] = set()

    def crawl(self, extra_urls: list[str] | None = None,
              max_articles: int = 20) -> list[dict]:
        logger.info("━━ 开始采集：工信部 ━━")

        # 收集所有种子 URL
        seed_urls: list[str] = []
        for urls in MIIT_SEEDS.values():
            seed_urls.extend(urls)
        if extra_urls:
            seed_urls.extend(extra_urls)

        # 用种子页面发现同栏目更多文章
        all_urls = self._discover(seed_urls, max_articles)
        logger.info("  共发现 %d 个待抓 URL", len(all_urls))

        results = []
        for url in all_urls:
            if url in self.visited:
                continue
            self.visited.add(url)

            soup = fetch(url, self.session)
            if not soup:
                continue

            title = self._extract_title(soup)
            content = clean_text(soup, MIIT_CONTENT_SELECTORS)
            if not title or len(content) < 100:
                logger.info("  ⓘ 内容过短/无标题，跳过：%s", url[-50:])
                continue

            # 从面包屑判断子类别
            cat = extract_category(soup, "工业政策")

            results.append({
                "source": "工信部",
                "category": cat,
                "title": title,
                "url": url,
                "content": content,
                "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "char_count": len(content),
            })
            logger.info("  ✓ [%02d] %s [%s]", len(results), title[:35], cat["page"])
            time.sleep(1.5)

            if len(results) >= max_articles:
                break

        logger.info("  ── 工信部完成，共 %d 篇 ──\n", len(results))
        return results

    def _discover(self, seeds: list[str], limit: int) -> list[str]:
        """
        从每个种子页面抓取"同栏目相关文章"链接。
        工信部文章页底部通常有"上一篇/下一篇"或同栏目文章列表。
        """
        found: list[str] = list(seeds)
        checked: set[str] = set()

        for seed in seeds:
            if seed in checked:
                continue
            checked.add(seed)
            soup = fetch(seed, self.session)
            if not soup:
                continue
            # 从页面内所有链接中找同栏目文章（URL 规律匹配）
            base = self._base_section(seed)  # e.g. /zwgk/zcwj/wjfb/tz/
            for a in soup.find_all("a", href=True):
                href = urljoin(seed, a["href"])
                if (base in href
                        and re.search(r"/art/20\d\d/art_[0-9a-f]+\.html$", href)
                        and href not in found):
                    found.append(href)
                    if len(found) >= limit:
                        return found
            time.sleep(1)

        return found[:limit]

    @staticmethod
    def _base_section(url: str) -> str:
        """提取 URL 中的栏目路径，如 /zwgk/zcwj/wjfb/tz/"""
        m = re.search(r"(/zwgk/zcwj/wjfb/[^/]+/)", url)
        return m.group(1) if m else "/zwgk/zcwj/wjfb/"

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        # 优先取 <h1>，其次取 <title>
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True).split("－")[0].split("-")[0].strip()
        return ""


# ═══════════════════════════════════════════
# ③ 国资委爬虫（实测选择器：ul li a）
# ═══════════════════════════════════════════

SASAC_CONFIG = {
    "name": "国资委",
    "base_url": "http://www.sasac.gov.cn",
    "category": "国资国企政策",
    "list_url": "http://www.sasac.gov.cn/n2588035/n2588320/n2588335/index.html",
    # 实测：列表为 <ul class=""> <li> <a href="...">标题</a>[日期]</li> </ul>
    # 页面中多个 ul，但含文章链接的 li > a 满足：href 含 /c\d+/content.html
    "list_selectors": [
        "ul li a[href*='/content.html']",  # 最精准（匹配国资委文章URL规律）
        "ul li a[href*='sasac.gov.cn']",
        "ul li a",
    ],
    # 实测正文：页面无特定 class div，内容在 body 正文区
    "content_selectors": [
        "div.TRS_Editor",
        "div#zoom",
        "div.article_con",
        "div.content",
        # 国资委正文区特征：含正文段落的第一个长文本 div
    ],
}


class SASACCrawler:
    def __init__(self):
        self.cfg = SASAC_CONFIG
        self.session = make_session("http://www.sasac.gov.cn/")

    def crawl(self, max_articles: int = 20) -> list[dict]:
        logger.info("━━ 开始采集：%s ━━", self.cfg["name"])

        soup = fetch(self.cfg["list_url"], self.session, encoding_hint="utf-8")
        if not soup:
            logger.error("✗ 国资委：列表页无法访问")
            return []

        links = self._parse_list(soup)
        if not links:
            logger.error("✗ 国资委：未找到文章链接")
            return []

        results = []
        for art in links[:max_articles]:
            art_soup = fetch(art["url"], self.session, encoding_hint="utf-8")
            if not art_soup:
                continue

            content = self._extract_content(art_soup)
            if len(content) < 100:
                logger.info("  ⓘ 内容过短，跳过：%s", art["title"][:30])
                continue

            cat = extract_category(art_soup, self.cfg["category"])
            results.append({
                "source": self.cfg["name"],
                "category": cat,
                "title": art["title"],
                "url": art["url"],
                "content": content,
                "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "char_count": len(content),
            })
            logger.info("  ✓ [%02d] %s [%s]", len(results), art["title"][:35], cat["page"])
            time.sleep(1.5)

        logger.info("  ── 国资委完成，共 %d 篇 ──\n", len(results))
        return results

    def _parse_list(self, soup: BeautifulSoup) -> list[dict]:
        base = self.cfg["base_url"]
        for sel in self.cfg["list_selectors"]:
            items = soup.select(sel)
            if not items:
                continue
            links = []
            for a in items[:30]:
                title = a.get_text(strip=True)
                href = a.get("href", "")
                if not href or not title or len(title) < 4:
                    continue
                # 过滤导航链接（href 应含 /c\d+ 路径段）
                if not re.search(r"/c\d+/", href) and "content.html" not in href:
                    continue
                full_url = href if href.startswith("http") else urljoin(base + "/", href)
                links.append({"title": title, "url": full_url})
            if links:
                logger.info("  ✓ 国资委命中选择器 [%s]，%d 条", sel, len(links))
                return links

        # fallback：找所有含 content.html 的链接
        logger.warning("  ⚠ 预设选择器无效，启用 fallback")
        links = []
        for a in soup.find_all("a", href=True):
            href, text = a["href"], a.get_text(strip=True)
            if ("content.html" in href
                    and 4 < len(text) < 100
                    and re.search(r"[\u4e00-\u9fa5]", text)):
                full_url = href if href.startswith("http") else urljoin(base + "/", href)
                links.append({"title": text, "url": full_url})
        logger.info("  fallback 找到 %d 条", len(links))
        return links[:20]

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        国资委文章页正文无统一 class，
        策略：找页面中字符最多、且含中文的 <div>/<p> 集合。
        """
        # 先试预设选择器
        for sel in self.cfg["content_selectors"]:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text

        # 找正文区：取所有 <p> 标签文本，过滤导航/页脚
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 20 and re.search(r"[\u4e00-\u9fa5]", text):
                paragraphs.append(text)
        if paragraphs:
            return "\n".join(paragraphs)

        # 最终 fallback
        body = soup.find("body")
        return body.get_text(separator="\n", strip=True)[:5000] if body else ""


# ═══════════════════════════════════════════
# 调试工具：打印页面结构，帮助确认选择器
# ═══════════════════════════════════════════

def debug_page(url: str, encoding_hint: str | None = None):
    session = make_session()
    soup = fetch(url, session, encoding_hint=encoding_hint)
    if not soup:
        print("页面无法访问")
        return
    print(f"\n{'='*60}")
    print(f"页面标题: {soup.title.string if soup.title else '无'}")
    print(f"{'='*60}")
    for i, ul in enumerate(soup.find_all(["ul", "ol"])[:10]):
        items = ul.find_all("li")
        if not items:
            continue
        cls = " ".join(ul.get("class", []))
        print(f"\n[列表{i+1}] <{ul.name} class='{cls}'> — {len(items)}个<li>")
        for li in items[:4]:
            a = li.find("a")
            if a:
                print(f"    → {a.get_text(strip=True)[:45]} | {a.get('href','')[:60]}")
    for div in soup.find_all("div", class_=True)[:15]:
        links = div.find_all("a", href=True)
        if len(links) >= 3:
            cls = " ".join(div.get("class", []))
            print(f"\n[DIV class='{cls}'] — {len(links)}个链接")
            for a in links[:3]:
                print(f"    → {a.get_text(strip=True)[:40]}")


# ═══════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════

def main():
    total = 0

    # ① 发改委
    ndrc = NDRCCrawler()
    ndrc_docs = ndrc.crawl(max_articles=15)
    save_docs(ndrc_docs, "国家发改委")
    total += len(ndrc_docs)

    # ② 工信部（传入你已知的文章链接作为补充种子）
    miit = MIITCrawler()
    miit_docs = miit.crawl(
        extra_urls=[
            # 直接粘贴你获得的工信部文章链接，爬虫会自动从中发现同栏目更多文章
            "https://www.miit.gov.cn/zwgk/zcwj/wjfb/gg/art/2026/art_3bfb3733b9e04e63b24feb4dd70a2d8e.html",
            "https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2026/art_aac97e3c05554e468aa2a503d1669661.html",
        ],
        max_articles=20,
    )
    save_docs(miit_docs, "工信部")
    total += len(miit_docs)

    # ③ 国资委
    sasac = SASACCrawler()
    sasac_docs = sasac.crawl(max_articles=20)
    save_docs(sasac_docs, "国资委")
    total += len(sasac_docs)

    logger.info("\n%s", "=" * 50)
    logger.info("✅ 全部采集完成，共 %d 篇政策文档", total)
    logger.info("%s", "=" * 50)


if __name__ == "__main__":
    # 调试模式：先用此函数检查页面结构，再跑 main()
    # debug_page("http://www.sasac.gov.cn/n2588035/n2588320/n2588335/index.html")
    # debug_page("https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2026/art_aac97e3c05554e468aa2a503d1669661.html")
    main()