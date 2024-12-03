# import requests
# import json
# import time
# import jieba
# from collections import Counter
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import nltk
# from nltk.corpus import stopwords
# import re

# from typing import List, Dict
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim import corpora, models
# import logging

# import urllib.parse
# import xml.etree.ElementTree as ET
# from datetime import datetime, timedelta


# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['font.family']='sans-serif'
# plt.rcParams['axes.unicode_minus'] = False # 插图中显示中文

# # 下载NLTK停用词
# nltk.download('stopwords')
# # 设置停用词
# english_stopwords = set(stopwords.words('english'))

# class ArxivScraper:
#     BASE_URL = "http://export.arxiv.org/api/query"
    
#     def __init__(self):
#         self.headers = {
#             "User-Agent": "Mozilla/5.0 (Python Academic Paper Scraper)"
#         }
    
#     def search_papers(self, query, start_year, end_year, max_results=1000):
#         papers = []
#         start = 0
        
#         # arXiv每次请求的最大结果数
#         batch_size = 500
        
#         # 构建日期范围
#         start_date = f"{start_year}0101"
#         end_date = f"{end_year}1231"
        
#         while start < max_results:
#             params = {
#                 "search_query": f"all:{query} AND submittedDate:[{start_date} TO {end_date}]",
#                 "start": start,
#                 "max_results": min(batch_size, max_results - start),
#                 "sortBy": "submittedDate",
#                 "sortOrder": "descending"
#             }
            
#             try:
#                 response = requests.get(self.BASE_URL, params=params, headers=self.headers)
                
#                 if response.status_code == 200:
#                     # 解析XML响应
#                     root = ET.fromstring(response.content)
                    
#                     # arXiv命名空间
#                     ns = {'arxiv': 'http://arxiv.org/schemas/atom'}
                    
#                     entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                    
#                     if not entries:
#                         break
                        
#                     for entry in entries:
#                         abstract = entry.find('{http://www.w3.org/2005/Atom}summary')
#                         title = entry.find('{http://www.w3.org/2005/Atom}title')
#                         published = entry.find('{http://www.w3.org/2005/Atom}published')
                        
#                         if abstract is not None and title is not None:
#                             paper = {
#                                 'title': title.text.strip(),
#                                 'abstract': abstract.text.strip(),
#                                 'published': published.text if published is not None else None,
#                                 'source': 'arXiv'
#                             }
#                             papers.append(paper)
                    
#                     print(f"arXiv: 已获取 {len(papers)} 篇论文")
                    
#                     if len(papers) >= max_results:
#                         papers = papers[:max_results]
#                         break
                    
#                     start += batch_size
                    
#                 else:
#                     print(f"arXiv API请求失败: {response.status_code}")
#                     break
                    
#             except Exception as e:
#                 print(f"处理arXiv数据时出错: {e}")
#                 break
                
#             time.sleep(3)  # arXiv API速率限制
            
#         return papers

# def save_results(papers, filename):
#     """保存结果到JSON文件"""
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(papers, f, ensure_ascii=False, indent=2)

# class TextAnalyzer:
#     def __init__(self, chinese_font_path='simhei.ttf'):
#         self.chinese_font_path = chinese_font_path
        
#         # 扩展停用词列表
#         self.english_stopwords = set(stopwords.words('english'))
#         self.english_stopwords.update(['also', 'using', 'used', 'based', 'study', 'paper', 'research'])
    
#         # 设置关键词集合
#         self.topic_keywords = {
#             'english': {
#                 'hot_topics': [
#                     'model', 'simulation', 'network', 'dynamics', 'behavior',
#                     'evolution', 'optimization', 'prediction', 'control', 'learning',
#                     'intelligence', 'adaptation', 'emergence', 'coordination'
#                 ],
#                 'research_methods': [
#                     'analysis', 'method', 'approach', 'algorithm', 'framework',
#                     'technique', 'methodology', 'strategy', 'implementation',
#                     'evaluation', 'experiment', 'validation'
#                 ]
#             },
#         }
        
#         # 设置日志
#         logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#     def clean_text(self, text: str) -> List[str]:
#         """清洗文本"""
        
#         # 去除非字母字符
#         text = re.sub(r'[^A-Za-z\s]', ' ', text)
#         # 转换为小写并分词
#         tokens = text.lower().split()
#         # 去除停用词和短词
#         tokens = [
#             word for word in tokens 
#             if word not in self.english_stopwords 
#             and len(word) > 2
#         ]
#         return tokens
    
#     def extract_keywords(self, papers: List[Dict], language: str) -> Counter:
#         """提取关键词"""
#         all_tokens = []
#         for paper in papers:
#             # use abstract or title 
#             tokens = self.clean_text(paper['abstract'], language)
#             tokens = self.clean_text(paper['title'], language)

#             all_tokens.extend(tokens)
#         return Counter(all_tokens)
    
#     def calculate_tfidf(self, papers: List[Dict]) -> Dict:
#         """计算TF-IDF权重"""
#         abstracts = [paper['abstract'] for paper in papers]
#         vectorizer = TfidfVectorizer(
#             stop_words='english',
#             max_features=1000,
#             token_pattern=r'[A-Za-z]{3,}'
#         )
  
#         tfidf_matrix = vectorizer.fit_transform(abstracts)
#         feature_names = vectorizer.get_feature_names_out()
        
#         # 计算每个词的平均TF-IDF值
#         tfidf_means = np.array(tfidf_matrix.mean(axis=0)).flatten()
        
#         # 创建词-权重字典
#         return dict(zip(feature_names, tfidf_means))
    
#     def apply_lda(self, papers: List[Dict],num_topics: int = 5) -> List[tuple]:
#         """应用LDA主题模型"""
#         # 准备文档集
#         docs = []
#         for paper in papers:
#             tokens = self.clean_text(paper['abstract'])
#             docs.append(tokens)
        
#         # 创建词典和语料库
#         dictionary = corpora.Dictionary(docs)
#         corpus = [dictionary.doc2bow(doc) for doc in docs]
        
#         # 训练LDA模型
#         lda_model = models.LdaModel(
#             corpus,
#             num_topics=num_topics,
#             id2word=dictionary,
#             passes=15
#         )
        
#         return lda_model.show_topics(formatted=False)
    
#     def generate_wordcloud(self, counter: Counter, title: str):
#         """生成词云图"""
#         wc = WordCloud(
#             width=1200,
#             height=800,
#             background_color='white',
#             font_path=self.chinese_font_path,
#             max_words=100,
#             colormap='viridis'
#         )
        
#         # 生成词云
#         wc.generate_from_frequencies(counter)
        
#         # 创建图形
#         plt.figure(figsize=(15, 10))
#         plt.imshow(wc, interpolation='bilinear')
#         plt.axis('off')
#         plt.title(title, fontsize=20, pad=20)
#         plt.tight_layout(pad=0)
#         plt.show()
    
#     def analyze_papers(self, papers: List[Dict]):
#         """分析论文数据"""
#         english_papers = papers
#         # 提取关键词
#         print("\n提取关键词...")
#         english_keywords = self.extract_keywords(english_papers, 'english')
        
#         # 计算TF-IDF
#         print("\n计算TF-IDF权重...")
#         english_tfidf = self.calculate_tfidf(english_papers, 'english')
#         # 应用LDA主题模型
#         print("\n应用LDA主题模型...")
#         english_topics = self.apply_lda(english_papers, 'english')
    
#         # 生成可视化
#         print("\n生成词云图...")
        
#         # 合并中英文关键词
#         combined_keywords = Counter()
#         combined_keywords.update(english_keywords)
#         self.generate_wordcloud(combined_keywords, "综合关键词云图")

#         print(english_topics)
#         # combined_topics = Counter(' '.join(english_topics))
#         # 初始化一个 Counter 对象
#         counter = Counter()

#         # 遍历 english_topics，累加权重
#         for _, word_list in english_topics:
#             for word, weight in word_list:
#                 counter[word] += weight  # 累加权重

#         # 打印结果
#         print(counter)
#         # combined_topics.update(english_topics)
#         self.generate_wordcloud(counter, "关键主题词云图")

#         # 可以根据需要添加更多可视化...
        
#         return {
#             'english_keywords': english_keywords,
#             'english_tfidf': english_tfidf,
#             'english_topics': english_topics,
#         }

# def main():
#     # 搜索参数
#     keywords = ["complex systems","complex network"]
#     start_year = 2020
#     end_year = 2024
#     max_results_per_source = 1000

#     # 初始化爬虫
#     arxiv_scraper = ArxivScraper()

#     all_papers = []

#     for keyword in keywords:
#         print(f"\n搜索关键词: {keyword}")
        
#         # 获取arXiv论文
#         print("\n从arXiv获取论文...")
#         arxiv_papers = arxiv_scraper.search_papers(
#             keyword, 
#             start_year, 
#             end_year, 
#             max_results_per_source
#         )
#         all_papers.extend(arxiv_papers)
        
#     print(f"\n总共收集到 {len(all_papers)} 篇论文")

#     # 保存结果
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_results(all_papers, f'papers_{timestamp}.json')
#     print(f"结果已保存到 papers_{timestamp}.json")

#     # 读取之前保存的论文数据
#     with open(f'./papers_{timestamp}.json', 'r', encoding='utf-8') as f:
#         papers = json.load(f)
    
#     # 创建分析器实例
#     analyzer = TextAnalyzer(chinese_font_path='C:\windows\Fonts\simhei.ttf')
    
#     # 分析论文
#     results = analyzer.analyze_papers(papers)
    
#     # 保存分析结果
#     # with open('analysis_results.json', 'w', encoding='utf-8') as f:
#     #     json.dump(results, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: arxiv_paper_scraper.py
@description: 从arXiv抓取论文，并对其进行文本分析，包括关键词提取、TF-IDF计算、LDA主题建模和词云生成。
@date: 2024-04-27
@author: MAE
@version: 1.0
"""

import requests
import json
import time
from collections import Counter
from typing import List, Dict
import re
import logging
from datetime import datetime

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models

import nltk
from nltk.corpus import stopwords

import xml.etree.ElementTree as ET

# 配置Matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 在图表中显示负号

# 下载NLTK停用词
nltk.download('stopwords')

# 设置日志配置
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def save_results(papers: List[Dict], filename: str):
    """
    将论文数据保存到JSON文件。

    :param papers: 论文列表
    :param filename: 保存的文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    logging.info(f"结果已保存到 {filename}")


class ArxivScraper:
    """
    用于从arXiv API抓取论文的爬虫类。
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Python Academic Paper Scraper)"
        }

    def search_papers(self, query: str, start_year: int, end_year: int, max_results: int = 1000) -> List[Dict]:
        """
        根据查询关键词和年份范围搜索论文。

        :param query: 搜索关键词
        :param start_year: 起始年份
        :param end_year: 结束年份
        :param max_results: 最大结果数量
        :return: 论文列表
        """
        papers = []
        start = 0
        batch_size = 500  # 每次请求的最大结果数
        start_date = f"{start_year}0101"
        end_date = f"{end_year}1231"

        while start < max_results:
            params = {
                "search_query": f"all:{query} AND submittedDate:[{start_date} TO {end_date}]",
                "start": start,
                "max_results": min(batch_size, max_results - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }

            try:
                response = requests.get(self.BASE_URL, params=params, headers=self.headers)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    entries = root.findall('{http://www.w3.org/2005/Atom}entry')

                    if not entries:
                        logging.info("无更多论文数据，停止抓取。")
                        break

                    for entry in entries:
                        abstract = entry.find('{http://www.w3.org/2005/Atom}summary')
                        title = entry.find('{http://www.w3.org/2005/Atom}title')
                        published = entry.find('{http://www.w3.org/2005/Atom}published')

                        if abstract is not None and title is not None:
                            paper = {
                                'title': title.text.strip(),
                                'abstract': abstract.text.strip(),
                                'published': published.text if published is not None else None,
                                'source': 'arXiv'
                            }
                            papers.append(paper)

                    logging.info(f"arXiv: 已获取 {len(papers)} 篇论文")

                    if len(papers) >= max_results:
                        papers = papers[:max_results]
                        break

                    start += batch_size

                else:
                    logging.error(f"arXiv API请求失败: {response.status_code}")
                    break

            except Exception as e:
                logging.error(f"处理arXiv数据时出错: {e}")
                break

            time.sleep(3)  # 遵守arXiv API速率限制

        return papers


class TextAnalyzer:
    """
    用于分析论文文本的类，包括关键词提取、TF-IDF计算、LDA主题建模和词云生成。
    """

    def __init__(self, chinese_font_path: str = 'simhei.ttf'):
        """
        初始化文本分析器。

        :param chinese_font_path: 中文字体文件路径
        """
        self.chinese_font_path = chinese_font_path
        self.english_stopwords = set(stopwords.words('english'))
        self.english_stopwords.update(['also', 'using', 'used', 'based', 'study', 'paper', 'research'])

        # 设置关键词集合（可扩展）TODO
        self.topic_keywords = {
            'english': {
                'hot_topics': [
                    'model', 'simulation', 'network', 'dynamics', 'behavior',
                    'evolution', 'optimization', 'prediction', 'control', 'learning',
                    'intelligence', 'adaptation', 'emergence', 'coordination'
                ],
                'research_methods': [
                    'analysis', 'method', 'approach', 'algorithm', 'framework',
                    'technique', 'methodology', 'strategy', 'implementation',
                    'evaluation', 'experiment', 'validation'
                ]
            },
        }

    def clean_text(self, text: str) -> List[str]:
        """
        清洗文本数据，包括去除非字母字符、转换为小写、分词以及去除停用词和短词。

        :param text: 原始文本
        :return: 清洗后的词列表
        """
        # 去除非字母字符
        text = re.sub(r'[^A-Za-z\s]', ' ', text)
        # 转换为小写并分词
        tokens = text.lower().split()
        # 去除停用词和短词
        tokens = [
            word for word in tokens
            if word not in self.english_stopwords
            and len(word) > 2
        ]
        return tokens

    def extract_keywords(self, papers: List[Dict]) -> Counter:
        """
        从论文中提取关键词。

        :param papers: 论文列表
        :return: 关键词计数器
        """
        all_tokens = []
        for paper in papers:
            # 使用摘要和标题进行清洗和分词
            abstract_tokens = self.clean_text(paper.get('abstract', ''))
            title_tokens = self.clean_text(paper.get('title', ''))
            all_tokens.extend(abstract_tokens)
            all_tokens.extend(title_tokens)
        return Counter(all_tokens)

    def calculate_tfidf(self, papers: List[Dict]) -> Dict[str, float]:
        """
        计算每个词的TF-IDF权重。

        :param papers: 论文列表
        :return: 词-权重字典
        """
        abstracts = [paper['abstract'] for paper in papers if 'abstract' in paper]
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            token_pattern=r'[A-Za-z]{3,}'
        )

        tfidf_matrix = vectorizer.fit_transform(abstracts)
        feature_names = vectorizer.get_feature_names_out()

        # 计算每个词的平均TF-IDF值
        tfidf_means = np.array(tfidf_matrix.mean(axis=0)).flatten()

        # 创建词-权重字典
        tfidf_dict = dict(zip(feature_names, tfidf_means))
        logging.info("TF-IDF计算完成。")
        return tfidf_dict

    def apply_lda(self, papers: List[Dict], num_topics: int = 5) -> List[tuple]:
        """
        应用LDA主题模型进行主题建模。

        :param papers: 论文列表
        :param num_topics: 主题数量
        :return: 主题列表，每个主题包含词及其权重
        """
        docs = []
        for paper in papers:
            tokens = self.clean_text(paper.get('abstract', ''))
            docs.append(tokens)

        # 创建词典和语料库
        dictionary = corpora.Dictionary(docs)
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        # 训练LDA模型
        lda_model = models.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=15,
            random_state=42
        )

        topics = lda_model.show_topics(formatted=False)
        logging.info("LDA主题建模完成。")
        return topics

    def generate_wordcloud(self, counter: Counter, title: str):
        """
        根据词频生成词云图。

        :param counter: 词频计数器
        :param title: 词云标题
        """
        wc = WordCloud(
            width=1200,
            height=800,
            background_color='white',
            font_path=self.chinese_font_path,
            max_words=100,
            colormap='viridis'
        )

        # 生成词云
        wc.generate_from_frequencies(counter)

        # 创建图形
        plt.figure(figsize=(15, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, pad=20)
        plt.tight_layout(pad=0)
        plt.show()
        logging.info(f"{title}词云图生成完成。")

    def analyze_papers(self, papers: List[Dict]) -> Dict[str, any]:
        """
        分析论文数据，包括关键词提取、TF-IDF计算、LDA主题建模和生成词云。

        :param papers: 论文列表
        :return: 分析结果字典
        """
        logging.info("开始分析论文数据...")

        # 提取关键词
        logging.info("提取关键词...")
        english_keywords = self.extract_keywords(papers)

        # 计算TF-IDF
        logging.info("计算TF-IDF权重...")
        english_tfidf = self.calculate_tfidf(papers)

        # 应用LDA主题模型
        logging.info("应用LDA主题模型...")
        english_topics = self.apply_lda(papers, num_topics=5)

        # 生成关键词词云
        logging.info("生成综合关键词词云图...")
        self.generate_wordcloud(english_keywords, "综合关键词云图")

        # 处理LDA主题，累加权重
        counter = Counter()
        for _, word_list in english_topics:
            for word, weight in word_list:
                counter[word] += weight

        # 生成主题词云
        logging.info("生成关键主题词云图...")
        self.generate_wordcloud(counter, "关键主题词云图")

        logging.info("论文数据分析完成。")

        return {
            'english_keywords': english_keywords,
            'english_tfidf': english_tfidf,
            'english_topics': english_topics,
        }


def main():
    """
    主函数，负责执行论文抓取和分析流程。
    """
    # 搜索参数
    keywords = ["complex systems", "complex network"]
    start_year = 2020
    end_year = 2024
    max_results_per_source = 1000

    # 初始化爬虫
    arxiv_scraper = ArxivScraper()

    all_papers = []

    for keyword in keywords:
        logging.info(f"搜索关键词: {keyword}")

        # 获取arXiv论文
        logging.info("从arXiv获取论文...")
        arxiv_papers = arxiv_scraper.search_papers(
            query=keyword,
            start_year=start_year,
            end_year=end_year,
            max_results=max_results_per_source
        )
        all_papers.extend(arxiv_papers)

    logging.info(f"总共收集到 {len(all_papers)} 篇论文")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'./papers_{timestamp}.json'
    save_results(all_papers, filename)

    # 创建分析器实例，确保字体路径正确
    analyzer = TextAnalyzer(chinese_font_path=r'C:\Windows\Fonts\simhei.ttf')  # 使用原始字符串

    # 分析论文
    results = analyzer.analyze_papers(all_papers)

    # 保存分析结果（可选）
    # with open('analysis_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    # logging.info("分析结果已保存到 analysis_results.json")


if __name__ == "__main__":
    main()

