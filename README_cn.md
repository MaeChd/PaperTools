# PaperTools

本工具专为研究人员和数据分析人员设计，能够高效地从 **arXiv** 获取学术论文，并对收集到的数据进行文本分析，包括关键词提取、TF-IDF计算、LDA主题建模和词云生成等功能。

## 🌍 语言切换

- [English](README.md)
- [中文](#功能特点)

---

## 功能特点

1. **论文抓取**：基于用户定义的关键词和日期范围，从arXiv抓取论文数据。
2. **关键词提取**：从论文标题和摘要中提取高频关键词。
3. **TF-IDF计算**：通过词频-逆文档频率计算重要术语的权重。
4. **LDA主题建模**：通过LDA算法揭示论文文本中的潜在主题。
5. **词云生成**：将关键词和主题生成直观的词云图。
6. **多语言支持**：支持英文和中文文本分析（需配置中文字体）。

---

## 安装说明

### 1. 环境要求

- Python 版本：3.8 或更高
- 依赖库：
  - `requests`
  - `matplotlib`
  - `wordcloud`
  - `nltk`
  - `numpy`
  - `scikit-learn`
  - `gensim`

使用以下命令安装依赖库：

```bash
pip install requests matplotlib wordcloud nltk numpy scikit-learn gensim
```

确保下载NLTK的停用词列表：

```python
import nltk
nltk.download('stopwords')
```

---

### 2. 中文字体配置（可选）

如果需要对中文文本进行分析或生成词云，请确保安装中文字体（如 `SimHei.ttf`）。并在脚本中正确设置字体路径，例如：
```python
chinese_font_path = r"C:\Windows\Fonts\simhei.ttf"
```

---

## 使用方法

### 1. 基本用法

运行脚本：
```bash
python arxiv_paper_scraper.py
```

运行后程序将：
1. 根据设置的关键词和日期范围从arXiv抓取论文。
2. 将抓取的论文数据保存为JSON文件（如 `papers_YYYYMMDD_HHMMSS.json`）。
3. 对论文数据进行分析，包括关键词提取、TF-IDF计算、主题建模，并生成词云图。

### 2. 可配置参数

您可以通过修改 `main()` 函数中的以下参数自定义抓取和分析行为：
- `keywords`: 搜索关键词列表（如：`["complex systems", "complex network"]`）。
- `start_year`, `end_year`: 搜索时间范围的起止年份。
- `max_results_per_source`: 每个关键词抓取的最大论文数量。

---

## 未来功能（规划中）

1. **支持更多论文来源**：
   - 集成更多API（如IEEE Xplore、Springer、PubMed等）。
   
2. **高级可视化**：
   - 增加关键词共现网络图。
   - 使用交互式工具（如pyLDAvis）增强主题建模的可视化效果。

3. **多种导出格式**：
   - 支持将分析结果导出为CSV、Excel或交互式仪表盘。

4. **情感分析**：
   - 对摘要或全文进行情感分析，以发现趋势。

5. **自定义停用词**：
   - 允许用户上传自定义停用词列表，提高关键词提取的精准度。

6. **性能优化**：
   - 优化代码以支持并行抓取和分析，提升运行效率。

---

## 如何贡献

我们欢迎社区贡献来扩展和完善此工具！您可以：
- 提交新功能的建议。
- 报告工具中的Bug或性能问题。
- 提交Pull Request来改进代码。

---

## 许可证

本项目采用 **MIT许可证** 开源。您可以自由使用、修改和分发，但需保留原作者信息。

---

祝您研究顺利！
