import nltk 
from pprint import pprint
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle
from statistics import mean

nltk.download(["names", "stopwords", "state_union", "twitter_samples", "movie_reviews", "averaged_perceptron_tagger", "vader_lexicon", "punkt"])

words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

stopwords = nltk.corpus.stopwords.words("english")

words = [w for w in words if w.lower() not in stopwords]

fd = nltk.FreqDist(words)

print(fd["America"])

lower_fd = nltk.FreqDist([w.lower() for w in fd])

text = nltk.Text(nltk.corpus.state_union.words())
concordance_list = text.concordance_list("america", lines=2)

for entry in concordance_list:
    print(entry.line)

words: list[str] = nltk.word_tokenize("Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex.")

text = nltk.Text(words)

fd = text.vocab()
fd.tabulate(3)

words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
finder = nltk.collocations.TrigramCollocationFinder.from_words(words)

print(finder.ngram_fd.most_common(2))
print(finder.ngram_fd.tabulate(2))

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("I am positive."))
print(sia.polarity_scores("I am negative."))
print(sia.polarity_scores("I am neutral."))

tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]

def is_positive(tweet: str) -> bool:
    """True if tweet has positive componet sentiment, false otherwise."""
    return sia.polarity_scores(tweet)["compound"] >0

shuffle(tweets)
for tweet in tweets[:10]:
    print(">", is_positive(tweet), tweet)


positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids

def is_positive(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0

shuffle(all_review_ids)
correct = 0
for review_id in all_review_ids: 
    if is_positive(review_id):
        if review_id in positive_review_ids:
            correct += 1
        else:
            if review_id in negative_review_ids: 
                correct += 1

print(F"{correct / len(all_review_ids):.2%} correct")