# ============================================================
# QUESTION 5:
# Student Feedback Sentiment Analysis
# ============================================================

from transformers import pipeline

# Load pre-trained sentiment analysis model
sentiment = pipeline("sentiment-analysis")

# Sample student feedback
feedback = [
    "The course was excellent and very informative.",
    "The teaching was poor and difficult to understand.",
    "The laboratory sessions were very useful.",
    "I did not like the course."
]

print("========================================")
print("   STUDENT FEEDBACK SENTIMENT ANALYSIS")
print("========================================")

# Analyze each feedback
for comment in feedback:

    result = sentiment(comment)[0]

    print("\nFeedback:", comment)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"], 4))