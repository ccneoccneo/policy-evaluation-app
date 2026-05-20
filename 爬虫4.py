import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_ndrc_policies(start_year=2005, end_year=2025):
    """爬取国家发改委政策文件（示例URL模式需根据实际调整）"""
    results = []
    for year in range(start_year, end_year+1):
        # 假设搜索页URL（实际需适配官网结构）
        url = f"https://www.ndrc.gov.cn/xxgk/zcfb/tz/ {year}/"
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for link in soup.select('a[href*="/xxgk/zcfb/"]'):
                title = link.get_text(strip=True)
                if any(kw in title for kw in ['装备制造', '工业', '产业', '制造']):
                    results.append({'year': year, 'title': title, 'url': link['href']})
            time.sleep(1)
        except:
            continue
    pd.DataFrame(results).to_csv('policies_raw.csv', index=False)
    return results