import concurrent.futures
import multiprocessing
import random
import re
import string
import time

import nltk
import pandas as pd
import wordninja
from nltk.corpus import words

stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
             "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}


def get_raw_review_data(seed: int, review_count: int):
    rating_amount = int(review_count / 10)

    # CSV dosyasını oku
    df = pd.read_csv('assets/rating_not_null.csv')

    # Karıştırma için seed ayarla
    random.seed(seed)

    # Verileri karıştır
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # print("İlk 100 veri:")
    # print(df_shuffled.head(100))

    # print("\nSon 100 veri:")
    # print(df_shuffled.tail(100))

    sampled_indices = []
    for rating in range(1, 11):
        indices = df_shuffled.index[df_shuffled['rating'] == rating].tolist()
        sampled_indices.extend(indices[:rating_amount])

    sampled_df = df_shuffled.iloc[sampled_indices].reset_index(drop=True)

    sampled_df_shuffled = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return sampled_df_shuffled


def clean_text(text, english_vocab, html_pattern):
    # HTML etiketlerini kaldırma
    text = re.sub(html_pattern, '', text)

    text_list = wordninja.split(text)
    text_string = ' '.join(text_list)

    # Noktalama işaretlerini kaldırma
    text_string = "".join([char for char in text_string if char not in string.punctuation])

    # Sayıları kaldırma
    text_string = re.sub(r'\d+', '', text_string)
    # Durak kelimelerini (stopwords) kaldırma ve küçük harfe dönüştürme
    #text_string = " ".join([word for word in text_string.lower().split() if word not in stopwords])

    # Spell checker
    text_string = " ".join([word if word in english_vocab else word for word in text_string.split()])

    # Tek boşluğa düşürme
    text_string = re.sub(r'\s+', ' ', text_string).strip()

    return text_string


def process_chunk(chunk):
    # Paralel işlemlerden gelen sonuçları toplama
    max_review_word = ""
    max_title_word = ""
    max_review_word_length = 0
    max_title_word_length = 0
    max_review_length = 0
    max_review_word_count = 0

    for index, row in chunk.iterrows():
        review_text_cleaned = clean_text(row['review'])
        title_text_cleaned = clean_text(row['title'])

        max_review_length = max(max_review_length, len(review_text_cleaned))

        review_words = review_text_cleaned.split()
        title_words = title_text_cleaned.split()

        if title_words:
            max_title_word_length_curr = max(len(word) for word in title_words)
            if max_title_word_length_curr > max_title_word_length:
                max_title_word_length = max_title_word_length_curr
                max_title_word = max(title_words, key=len)

        if review_words:
            max_review_word_length_curr = max(len(word) for word in review_words)
            if max_review_word_length_curr > max_review_word_length:
                max_review_word_length = max_review_word_length_curr
                max_review_word = max(review_words, key=len)

            review_word_count = len(review_words)
            max_review_word_count = max(max_review_word_count, review_word_count)

        #print(f"rating: {row['rating']} title: {title_text_cleaned}  review: {review_text_cleaned}")
        yield row['rating'], title_text_cleaned, review_text_cleaned

    #return max_review_word, max_title_word, max_review_word_length, max_title_word_length, max_review_length, max_review_word_count


def create_processed_review_data(seed: int):
    nltk.download('words')
    # nltk'nin ingilizce kelimeler listesi
    english_vocab = set(words.words())

    # HTML etiketlerini kaldırmak için düzenli ifade
    html_pattern = re.compile('<.*?>')

    start_time = time.time()

    # CSV dosyasını yükleme
    raw_reviews = get_raw_review_data(seed, 1_000_000)
    cleaned_reviews = pd.DataFrame(columns=raw_reviews.columns)

    max_review_word_length = 0
    max_title_word_length = 0
    max_review_length = 0
    max_review_word_count = 0

    for index, row in raw_reviews.iterrows():
        rating = row['rating']
        title_text_cleaned = clean_text(row['title'], english_vocab, html_pattern)
        review_text_cleaned = clean_text(row['review'], english_vocab, html_pattern)
        data = [rating, title_text_cleaned, review_text_cleaned]
        cleaned_reviews.loc[len(cleaned_reviews)] = data

        max_review_length = max(max_review_length, len(review_text_cleaned))

        review_words = review_text_cleaned.split()
        title_words = title_text_cleaned.split()

        if title_words:
            max_title_word_length_curr = max(len(word) for word in title_words)
            if max_title_word_length_curr > max_title_word_length:
                max_title_word_length = max_title_word_length_curr

        if review_words:
            max_review_word_length_curr = max(len(word) for word in review_words)
            if max_review_word_length_curr > max_review_word_length:
                max_review_word_length = max_review_word_length_curr

            review_word_count = len(review_words)
            max_review_word_count = max(max_review_word_count, review_word_count)

    cleaned_reviews.to_csv("assets/processed_reviews_2.csv", index=False)

    end_time = time.time()
    execution_time = end_time - start_time

    print(max_review_word_length, max_title_word_length, max_review_length, max_review_word_count)
    print(f"Preprocess data creation done time: {int(execution_time / 60)} minutes, {execution_time % 60} seconds")

    # random.seed(seed)
    # df_shuffle = chunks.sample(frac=1, random_state=seed).reset_index(drop=True).head(100)


if __name__ == '__main__':
    create_processed_review_data(0)
