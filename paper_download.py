#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: paper_download.py
@description: 从抓取的论文csv文件中批量打包下载论文
@date: 2024-12-3
@author: Sokachii
@version: 1.0
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import zipfile
import chardet

# 从arXiv API查找论文PDF链接
def search_arxiv(title):
    query = f"http://export.arxiv.org/api/query?search_query=all:{title}&start=0&max_results=1"
    response = requests.get(query)
    soup = BeautifulSoup(response.content, 'xml')
    
    entry = soup.find('entry')
    if entry:
        pdf_link = entry.find('link', title='pdf').get('href')
        return pdf_link
    else:
        return None

# 1. 读取CSV文件，提取标题和作者
with open(r"your_file", 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# 使用检测到的编码读取CSV
df = pd.read_csv(r"your_file", encoding=encoding)

# 保存PDF的文件夹
if not os.path.exists('downloaded_papers'):
    os.makedirs('downloaded_papers')

# 2. 设置起始索引
start_index = 1  # 修改部分：指定从第1个论文开始查找

# 3. 循环遍历CSV中的每篇论文，搜索并下载PDF
for index, row in df.iterrows():
    # 修改部分：只从指定的起始索引开始查找
    if index < start_index:
        continue

    title = row['title']  # 假设列名为 'title'
    print(f"正在查找论文: {title}")
    
    # 通过arXiv搜索论文
    pdf_link = search_arxiv(title)
    
    if pdf_link:
        print(f"找到PDF链接: {pdf_link}")
        pdf_response = requests.get(pdf_link)
        
        # 4. 下载PDF文件
        pdf_filename = f"paper_{index + 1}.pdf"
        pdf_path = os.path.join('downloaded_papers', pdf_filename)
        with open(pdf_path, 'wb') as f:
            f.write(pdf_response.content)
        print(f"下载完成: {pdf_filename}")
    else:
        print(f"未找到论文: {title} 的PDF链接")

# 5. 打包所有PDF文件到ZIP
zip_filename = 'papers.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for pdf_file in os.listdir('downloaded_papers'):
        pdf_file_path = os.path.join('downloaded_papers', pdf_file)
        zipf.write(pdf_file_path, pdf_file)
        
print(f"所有PDF文件已打包为 {zip_filename}")
