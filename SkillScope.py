import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('adzuna_jobs.csv')

print(df.shape)
df.head()

import re
import html

def clean(text,
          lowercase=True,
          remove_numbers=True,
          remove_non_ascii=False):

    # Convert HTML escapes like &amp; to characters.
    text = html.unescape(text)

    # Remove tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)

    # Remove markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]])\]\([^\(\)]\)', r'\1', text)

    # Remove text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)

    # Remove standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)

    # Remove standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)

    # Remove periods, commas, semicolons, colons, quotes, parentheses
    text = re.sub(r'[.,;:"\'()!?]', ' ', text)

    if remove_numbers:
        text = re.sub(r'\b\d+\b', ' ', text)

    if remove_non_ascii:
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    if lowercase:
        text = text.lower()

    # Remove sequences of white spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


df['clean_description'] = df['description'].apply(clean)
df.head()

import re ###
import spacy ###
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import textacy ###

def custom_tokenizer(nlp):

    # use default patterns except the ones matched by re.search
    prefixes = [pattern for pattern in nlp.Defaults.prefixes
                if pattern not in ['-', '_', '#']]
    suffixes = [pattern for pattern in nlp.Defaults.suffixes
                if pattern not in ['_']]
    infixes  = [pattern for pattern in nlp.Defaults.infixes
                if not re.search(pattern, 'xx-xx')]

    return Tokenizer(vocab          = nlp.vocab,
                     rules          = nlp.Defaults.tokenizer_exceptions,
                     prefix_search  = compile_prefix_regex(prefixes).search,
                     suffix_search  = compile_suffix_regex(suffixes).search,
                     infix_finditer = compile_infix_regex(infixes).finditer,
                     token_match    = nlp.Defaults.token_match)

def extract_noun_phrases(doc, sep=' '):
    patterns = ["POS:ADJ POS:NOUN:+", "POS:NOUN POS:NOUN:+"]

    spans = textacy.extract.matches.token_matches(doc, patterns=patterns)

    return [str(s) for s in spans]

def extract_lemmas(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.words(doc, **kwargs)]

    return [(e.lemma_, e.label_) for e in ents]

def extract_entities(doc, include_types=None, sep='_'):
    ents = textacy.extract.entities(doc,
             include_types=include_types,
             exclude_types=None,
             drop_determiners=True,
             min_freq=1)

    #debugging...
    #for e in ents:
      #print(e.lemma_, e.label_)

    return [(e.lemma_, e.label_) for e in ents]

def extract_nlp(doc):
    return {
    'lemmas'          : extract_lemmas(doc,
                                      exclude_pos = ['PART', 'PUNCT', 'DET', 'PRON', 'SYM', 'SPACE'],
                                      filter_stops = True),
    'adjs_verbs'      : extract_lemmas(doc, include_pos = ['ADJ', 'VERB']),
    'nouns'           : extract_lemmas(doc, include_pos = ['NOUN', 'PROPN']),
    'noun_phrases'    : extract_noun_phrases(doc),
    'entities'        : extract_entities(doc, ['PERSON', 'ORG', 'GPE', 'LOC'])
    }

import nltk
nltk.download('punkt_tab')

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

stop_words_list = list(stopwords)

vectorizer = TfidfVectorizer(stop_words=stop_words_list, min_df=5, max_df=0.7)

tfidf_df = vectorizer.fit_transform(df['clean_description'])

num_features = tfidf_df.shape[1]

print(num_features)

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

# Assuming you have your custom_tokenizer, extract_noun_phrases, extract_lemmas, and extract_entities functions defined

# Load spaCy model (if not already loaded)
nlp = spacy.load('en_core_web_sm', disable=[])
nlp.tokenizer = custom_tokenizer(nlp)

# Apply the extract_nlp function to the 'final_description' column
df['nlp_features'] = df['clean_description'].apply(lambda text: extract_nlp(nlp(text)))

df.head()

# Assuming 'nlp_features' column contains dictionaries with different text features
# Join the values of desired features within each dictionary into a single string
df['text_for_tfidf'] = df['nlp_features'].apply(lambda x: ' '.join(x.get('lemmas', []) + x.get('noun_phrases', [])))

tfidf_text_vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
# Now use this new column 'text_for_tfidf' for TF-IDF vectorization
tfidf_text_dt = tfidf_text_vectorizer.fit_transform(df['text_for_tfidf'])

from sklearn.decomposition import NMF
num_topics = 10
nmf_text_model = NMF(n_components=num_topics, random_state=42)
W_text_matrix = nmf_text_model.fit_transform(tfidf_text_dt)
H_text_matrix = nmf_text_model.components_
W_text_matrix.shape, H_text_matrix.shape

def display_topics(model, features, no_top_words=5):
    for topic_id, word_loadings in enumerate(model.components_):
        total = word_loadings.sum()
        sorted_loadings = word_loadings.argsort()[::-1] # invert sort order
        print(f"\nTopic {topic_id}")
        for i in range(0, no_top_words):
          word_id = sorted_loadings[i]
          print(f"  {features[word_id]} ({abs(word_loadings[word_id]*100.0/total):2.2f})")

display_topics(nmf_text_model, tfidf_text_vectorizer.get_feature_names_out(), 10)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
enhanced_stop_words = ['aba','akkodis','ketchum', "futu", "nbc", "smbc", 'silos', 'stv', 'aec','gro','job','description','datum']
add_stopwords = list(ENGLISH_STOP_WORDS.union(enhanced_stop_words))

vectorizer = TfidfVectorizer(stop_words=add_stopwords,min_df=5,max_df=0.7)
dtm = vectorizer.fit_transform(df["text_for_tfidf"])
feature_names = vectorizer.get_feature_names_out()

num_topics = 10
nmf_text_model = NMF(n_components=num_topics, random_state=42)
W_enhanced = nmf_text_model.fit_transform(dtm)
H_enhanced = nmf_text_model.components_

def display_topics(H, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(H):
        print(f"\nTopic {topic_idx}:")
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            print(f"{feature_names[i]} ({topic[i]:.3f})")

display_topics(H_enhanced, feature_names, no_top_words=5)

topic_names = [
  'Community Justice & Social Advocacy',
  'Data Analytics & Engineering',
  'Behavioral Therapy & Autism Services',
  'Digital Content Strategy & Communications',
  'Financial Services & Global Banking',
  'Healthcare & Insurance Operations',
  'Marketing, SEO & Brand Management',
  'Risk, Investment & Technology Management',
  'Award-Winning AEC & Industry Recognition',
  'Business Analysis & Product Development'
]
def topic_distribution(topic_names, percents):
    plt.barh(topic_names, percents)
    plt.gca().invert_yaxis()
    plt.xlabel("Percentage of Documents")
    plt.show()

doc_pct = W_enhanced.sum(axis=0) / W_enhanced.sum() * 100.0

# Plot the distribution
topic_distribution(topic_names, doc_pct)


from sklearn.decomposition import TruncatedSVD

svd_para_model = TruncatedSVD(n_components = num_topics, random_state=42)
W_svd_para_matrix = svd_para_model.fit_transform(dtm)
H_svd_para_matrix = svd_para_model.components_

display_topics(H_svd_para_matrix, vectorizer.get_feature_names_out(), 5)

doc_pct = W_svd_para_matrix.sum(axis=0)/W_svd_para_matrix.sum()*100.0

topic_names = [
  'Community Justice & Civic Engagement',
  'Data & Business Analytics',
  'Behavioral & Mental Health Services',
  'Digital Content & Media Strategy',
  'Financial Services & Banking',
  'Health Marketing & Brand Strategy',
  'Healthcare & Insurance Operations',
  'Risk, Investment & Tech Management',
  'Health Risk & Market Insights',
  'Business Titles & Analyst Roles'
]

def topic_distribution(topic_names, percents):
    plt.barh(topic_names, percents)
    plt.gca().invert_yaxis()
    plt.xlabel("Percentage of Documents")
    plt.show()
doc_pct = W_enhanced.sum(axis=0) / W_enhanced.sum() * 100.0


# Plot the distribution
topic_distribution(topic_names, doc_pct)

from sklearn.feature_extraction.text import CountVectorizer
count_para_vectorizer = CountVectorizer(stop_words=list(enhanced_stop_words), min_df=5, max_df=0.7)
count_para_dt = count_para_vectorizer.fit_transform(df["text_for_tfidf"])
count_para_dt.shape

from sklearn.decomposition import LatentDirichletAllocation
lda_para_model = LatentDirichletAllocation(n_components = 10, random_state=42)
W_lda_para_matrix = lda_para_model.fit_transform(count_para_dt)
H_lda_para_matrix = lda_para_model.components_

display_topics(H_lda_para_matrix, count_para_vectorizer.get_feature_names_out(), 5)

topic_names = [
  'Global Finance & Corporate Services',
  'Data Analytics & Business Intelligence',
  'Tech-Driven Insurance & Information Management',
  'Community Justice & Civic Organizations',
  'Healthcare, Credit & Investment Insights',
  'Digital Content & Global Communications',
  'Marketing Strategy & Media Operations',
  'Health & Data Analyst Roles',
  'Product Design, UX & Research',
  'Risk Management & Behavioral Analytics'
]

def topic_distribution(topic_names, percents):
    plt.barh(topic_names, percents)
    plt.gca().invert_yaxis()
    plt.xlabel("Percentage of Documents")
    plt.show()

doc_pct = W_enhanced.sum(axis=0) / W_enhanced.sum() * 100.0


# Plot the distribution
topic_distribution(topic_names, doc_pct)

df.to_csv('final_data.csv', index=False)


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'add_stopwords', 'df', and 'topic_names' are already defined

vectorizer = TfidfVectorizer(stop_words=add_stopwords, min_df=5, max_df=0.7)
tfidf_text_dt = vectorizer.fit_transform(df['text_for_tfidf'])

# Get feature names (words)
voc = vectorizer.get_feature_names_out()

# Calculate cosine similarity between words
r = cosine_similarity(tfidf_text_dt.T, tfidf_text_dt.T)
np.fill_diagonal(r, 0)  # Set diagonal to 0 to avoid self-similarity

# Get top words across all topics based on total TF-IDF score
word_scores = tfidf_text_dt.sum(axis=0).A1  # Sum TF-IDF scores for each word
top_word_indices = np.argsort(word_scores)[-50:]  # Get indices of top 50 words
top_words = [voc[i] for i in top_word_indices]

# Filter similarity matrix for top words only
reduced_r = r[top_word_indices][:, top_word_indices]

# Plot heatmap
plt.figure(figsize=(16, 10))
plt.title('Word Similarity Heatmap (Top Words Across Topics)')
sns.heatmap(data=reduced_r, xticklabels=top_words, yticklabels=top_words,
            cbar=True, cmap="Blues")
plt.show()

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("🔍 Word Similarity Heatmap (Top TF-IDF Terms)")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words=add_stopwords, min_df=5, max_df=0.7)
tfidf_text_dt = vectorizer.fit_transform(df['text_for_tfidf'])

# Vocabulary
voc = vectorizer.get_feature_names_out()

# Cosine similarity between words
similarity_matrix = cosine_similarity(tfidf_text_dt.T, tfidf_text_dt.T)
np.fill_diagonal(similarity_matrix, 0)

# Get top 50 words by total TF-IDF score
word_scores = tfidf_text_dt.sum(axis=0).A1
top_word_indices = np.argsort(word_scores)[-50:]
top_words = [voc[i] for i in top_word_indices]

# Filter similarity matrix for top words
reduced_similarity = similarity_matrix[top_word_indices][:, top_word_indices]

# Plot heatmap
fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(data=reduced_similarity, xticklabels=top_words, yticklabels=top_words,
            cbar=True, cmap="Blues", ax=ax)
plt.xticks(rotation=90)
ax.set_title("Word Similarity Heatmap (Top 50 TF-IDF Terms)")
st.pyplot(fig)


# Example topic distributions (replace with real values from your model)
nmf_doc_pct = np.array([12.0, 12.0, 5.5, 5.6, 8.0, 10.0, 10.8, 10.0, 11.0, 17.0])
nmf_topic_names = [
    'Community Justice & Social Advocacy',
    'Data Analytics & Engineering',
    'Behavioral Therapy & Autism Services',
    'Digital Content Strategy & Communications',
    'Financial Services & Global Banking',
    'Healthcare & Insurance Operations',
    'Marketing, SEO & Brand Management',
    'Risk, Investment & Technology Management',
    'Award-Winning AEC & Industry Recognition',
    'Business Analysis & Product Development'
]

lda_doc_pct = np.array([11.1, 15.4, 8.9, 9.6, 10.2, 7.5, 10.8, 6.7, 10.0, 9.8])
lda_topic_names = [
    'Global Finance & Corporate Services',
    'Data Analytics & Business Intelligence',
    'Tech-Driven Insurance & Info Mgmt',
    'Community Justice & Civic Organizations',
    'Healthcare, Credit & Investment Insights',
    'Digital Content & Global Communications',
    'Marketing Strategy & Media Ops',
    'Health & Data Analyst Roles',
    'Product Design, UX & Research',
    'Risk Management & Behavioral Analytics'
]

svd_doc_pct = np.array([10.8, 13.2, 9.7, 8.9, 9.2, 10.5, 11.0, 7.3, 10.4, 9.0])
svd_topic_names = [
    'Community Justice & Civic Engagement',
    'Data & Business Analytics',
    'Behavioral & Mental Health Services',
    'Digital Content & Media Strategy',
    'Financial Services & Banking',
    'Health Marketing & Brand Strategy',
    'Healthcare & Insurance Operations',
    'Risk, Investment & Tech Management',
    'Health Risk & Market Insights',
    'Business Titles & Analyst Roles'
]

# Visualization for NMF
st.markdown("### NMF Topic Distribution")
fig_nmf, ax_nmf = plt.subplots(figsize=(10, 6))
ax_nmf.barh(nmf_topic_names, nmf_doc_pct)
ax_nmf.invert_yaxis()
ax_nmf.set_xlabel("Percentage of Documents")
ax_nmf.set_title("NMF Topics")
st.pyplot(fig_nmf)

# Visualization for LDA
st.markdown("### LDA Topic Distribution")
fig_lda, ax_lda = plt.subplots(figsize=(10, 6))
ax_lda.barh(lda_topic_names, lda_doc_pct)
ax_lda.invert_yaxis()
ax_lda.set_xlabel("Percentage of Documents")
ax_lda.set_title("LDA Topics")
st.pyplot(fig_lda)

# Visualization for SVD
st.markdown("### SVD Topic Distribution")
fig_svd, ax_svd = plt.subplots(figsize=(10, 6))
ax_svd.barh(svd_topic_names, svd_doc_pct)
ax_svd.invert_yaxis()
ax_svd.set_xlabel("Percentage of Documents")
ax_svd.set_title("SVD Topics")
st.pyplot(fig_svd)
