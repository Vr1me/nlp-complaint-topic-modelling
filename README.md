# NLP Analysis of Consumer Complaint Narratives

Portfolio project for the IU course **Project: Data Analysis (DLBDSEDA02)**, Task 1: use NLP techniques to analyse a collection of texts.

The notebook extracts the most frequently discussed topics from free-text consumer complaints. It stands in for the use case of a municipality that receives more written complaints than its staff can read.

## Data

[Consumer Complaint Database](https://www.kaggle.com/datasets/selener/consumer-complaint-database) (CFPB, via Kaggle), 1,282,355 complaints with 18 columns. Only the column `Consumer complaint narrative` is used. After removing empty and duplicate narratives, a random sample of 30,000 texts is analysed (`random_state=42`).

The dataset is not stored in this repository. The notebook downloads it through the Kaggle API.

## Workflow

1. **Load and inspect** the data (pandas)
2. **Preprocess**: lowercase, remove URLs, numbers, punctuation, masked personal data (`XXXX`) and stop words, then lemmatise (re, NLTK)
3. **Vectorise**: Bag of Words and TF-IDF with unigrams and bigrams, 5,000 features (scikit-learn)
4. **Extract topics**: LDA on Bag of Words, NMF on TF-IDF, five topics each (scikit-learn)
5. **Visualise**: word frequencies, word cloud, topic prevalence (matplotlib, wordcloud)

## Results

| Model | Largest topics |
|---|---|
| LDA | Credit reporting and debt (34%), credit cards and phone contact (25%), loan payments (17%) |
| NMF | Credit reports (26%), mortgage and loan payments (23%), phone contact (20%), bank accounts (17%), debt collection (14%) |

NMF on TF-IDF gives the more distinct topics. The discussion is at the end of the notebook.

## How to run

**Google Colab (as developed)**

1. Open `complaint_topic_modelling.ipynb` in Colab.
2. Create a Kaggle API token (Kaggle → Settings → API → Create New Token) and upload `kaggle.json` to `/content`.
3. Run all cells.

**Locally**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook complaint_topic_modelling.ipynb
```

Place `kaggle.json` in `~/.kaggle/` and change the `/content/...` paths in the notebook to a local folder.

`kaggle.json` is a personal key and is excluded through `.gitignore`.
