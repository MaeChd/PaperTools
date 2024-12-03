# PaperTools

This tool is designed for researchers and data analysts to efficiently fetch academic papers from **arXiv** and perform text analysis on the collected data. It includes features like keyword extraction, TF-IDF calculation, LDA topic modeling, and word cloud generation.

## 🌍 Language Options

- [English](#features)
- [中文](README_cn.md)

## Features

- **Paper Scraping**: Fetch papers from arXiv based on user-defined keywords and date ranges.
- **Keyword Extraction**: Extract frequently occurring keywords from paper titles and abstracts.
- **TF-IDF Calculation**: Compute term frequency-inverse document frequency to identify significant terms.
- **LDA Topic Modeling**: Apply Latent Dirichlet Allocation to uncover underlying topics in the text data.
- **Word Cloud Visualization**: Generate visually appealing word clouds for keywords and topics.
- **Multi-language Support**: Optimized for both English and Chinese text (requires proper font configuration).

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Recommended libraries:
  - `requests`
  - `matplotlib`
  - `wordcloud`
  - `nltk`
  - `numpy`
  - `scikit-learn`
  - `gensim`

Install the required libraries:

```bash
pip install requests matplotlib wordcloud nltk numpy scikit-learn gensim
```

Ensure the NLTK stopword list is downloaded:

```python
import nltk
nltk.download('stopwords')
```

### Font Setup (Optional for Chinese Support)

If analyzing or visualizing Chinese text, ensure a proper Chinese font (e.g., `SimHei.ttf`) is installed. Update the font path in the script (`chinese_font_path` parameter).

---

## Usage

### 1. Basic Usage
To run the tool, execute the script:
```bash
python arxiv_paper_scraper.py
```

The script will:
1. Fetch papers based on predefined keywords and date ranges.
2. Save the collected data as a JSON file (e.g., `papers_YYYYMMDD_HHMMSS.json`).
3. Analyze the papers and generate word clouds for keywords and topics.

### 2. Configurable Parameters
Modify these parameters in the `main()` function:
- `keywords`: List of search keywords.
- `start_year`, `end_year`: Date range for paper search.
- `max_results_per_source`: Maximum number of papers to fetch.

---

## Future Features (Planned)

1. **Support for Additional Sources**:
   - Integrate other APIs (e.g., IEEE Xplore, Springer, or PubMed).
   
2. **Advanced Visualization**:
   - Add network graphs for keyword co-occurrence.
   - Enhance topic modeling results with interactive visualizations (e.g., pyLDAvis).

3. **Export Formats**:
   - Support for exporting results to CSV, Excel, or interactive dashboards.

4. **Sentiment Analysis**:
   - Analyze sentiment in abstracts or full text for trend identification.

5. **Custom Stopwords**:
   - Allow user-defined stopword lists for better keyword extraction.

6. **Performance Improvements**:
   - Parallelize scraping and analysis for faster execution.

---

## Contributing

We welcome contributions to enhance this tool! Feel free to:
- Suggest new features.
- Report bugs or performance issues.
- Submit pull requests for improvements.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this tool with attribution.

---



Happy Researching!
