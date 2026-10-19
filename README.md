# NLP Analysis of Consumer Complaint Narratives

Portfolio project for the IU course **Project: Data Analysis (DLBDSEDA02)**, Task 1: use NLP techniques to analyse a collection of texts.

The notebook extracts the most frequently discussed topics from free-text consumer complaints. It stands in for the use case of a municipality that receives more written complaints than its staff can read.

## Files

| File | Content |
|---|---|
| `Vlasov-Mykhailo_102301850_DataAnalysis_P3.ipynb` | Final notebook (phase 3) with coherence-based model selection and Product check |
| `complaint_topic_modelling.ipynb` | Earlier version from phase 2 (five topics, no coherence evaluation) |

## Data

[Consumer Complaint Database](https://www.kaggle.com/datasets/selener/consumer-complaint-database) (CFPB, via Kaggle), 1,282,355 complaints with 18 columns. The column `Consumer complaint narrative` is used for the analysis. After removing empty and duplicate narratives (366,945 left), a random sample of 30,000 texts is drawn (`random_state=42`). One text is empty after preprocessing, so 29,999 complaints are analysed.

The dataset is not stored in this repository. The notebook downloads it through the Kaggle API.

## Workflow

1. **Load and inspect** the data (pandas)
2. **Preprocess**: lowercase, remove URLs, numbers, punctuation, masked personal data (`XXXX`) and stop words, then lemmatise (re, NLTK)
3. **Vectorise**: Bag of Words and TF-IDF with unigrams and bigrams, 5,000 features (scikit-learn)
4. **Extract topics**: LDA on Bag of Words, NMF on TF-IDF, first with five topics each (scikit-learn)
5. **Select the model**: C_V coherence (Gensim) for LDA and NMF with 3 to 8 topics
6. **Final model**: best configuration retrained on all 29,999 complaints
7. **Validate**: compare the final topics with the `Product` column of the dataset

## Results

LDA reached higher coherence than NMF for every tested number of topics. The best configuration was LDA with three topics (tuning C_V = 0.7252, final C_V = 0.7222).

| Rank | Topic | Top keywords | Complaints | Share |
|---|---|---|---|---|
| 1 | Credit reporting and debt collection | credit, report, account, debt, information, credit report, reporting, collection, company, letter | 11,647 | 38.82% |
| 2 | Banking, credit cards and customer service | account, card, payment, call, bank, called, credit, time, day, received | 11,618 | 38.73% |
| 3 | Loan and mortgage payments | loan, payment, mortgage, time, home, year, month, bank, interest, amount | 6,734 | 22.45% |

The Product check supports topics 1 and 3: 93.1% of topic 1 are credit reporting or debt collection complaints, and 75.9% of topic 3 are mortgage or student loan complaints. Topic 2 is more mixed (debt collection 23.4%, credit card products 24.9%).

## How to run

**Google Colab (as developed)**

1. Open `Vlasov-Mykhailo_102301850_DataAnalysis_P3.ipynb` in Colab.
2. Create a Kaggle API key (Kaggle → Settings → API) and upload `kaggle.json` to `/content`.
3. Run all cells. The full run takes about 15 minutes, most of it in the coherence evaluation.

**Locally**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Vlasov-Mykhailo_102301850_DataAnalysis_P3.ipynb
```

Change the `/content/...` paths in the notebook to a local folder.

Never commit `kaggle.json` or any API token. Key files are excluded through `.gitignore`.
