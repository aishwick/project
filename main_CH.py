# Import necessary libraries
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import plotly.express as px

# Download VADER lexicon
nltk.download('vader_lexicon')

from keys import client_id, client_secret, user_agent

# reddit instance
reddit = praw.Reddit(
    client_id=client_id, # replace with  client_id, etc.
    client_secret="PLAplE15jwH8Lji4--mbswZWw-eKag",
    user_agent="script:Aish:v1.0 (by /u/Hungry_Gift)"
)

SUBREDDIT = "gaming"

# VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Define the subreddit, keywords, and minimum upvote threshold
subreddit = reddit.subreddit(SUBREDDIT) 
keywords = ["privacy", "phish", "breach", "2fa", "mfa", "ddos", "vpn", "security", "malicious", "doxx", "ip address exposure", "account theft", "account hacking", "in-game privacy", "phishing scams", "data breach", "ransomware", "malware", "virtual currency scam", "social engineering", "account security", "game client vulnerability", "metadata exposure"]
min_upvotes = 5
posts_data = []



# scrape posts by searching for each keyword
for keyword in keywords:
    for post in subreddit.search(keyword, time_filter="all"):
        # Check if post meets the upvote condition and check if bodytext is not empty
        if post.score > min_upvotes and post.selftext:
            # Run VADER sentiment analysis
            sentiment = analyzer.polarity_scores(post.selftext) # filter out link posts

            # Append post data as a dictionary, including the keyword
            post_dict = {
                "game": post.title,
                "post_text": post.selftext,
                "sentiment": sentiment['compound'],
                "keyword": keyword
            }
            posts_data.append(post_dict)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(posts_data)
df.to_csv(SUBREDDIT + '.csv') # store dataframe in csv



# new file

# load the data
df = pd.read_csv('reddit_posts.csv')

# Display the total posts and DataFrame
print(f"Total posts stored: {len(df)}")

# Group by 'keyword' and count the number of posts for each keyword
keyword_counts = df['keyword'].value_counts().reset_index()
keyword_counts.columns = ['keyword', 'post_count']  # Rename columns for clarity

# Display the table
print("Number of posts scraped for each keyword:")
print(keyword_counts)

# Count total rows with NaN in 'sentiment'
nan_count = df['sentiment'].isna().sum()
print(f"Total posts with NaN sentiment: {nan_count}")

# Histogram of sentiment scores
hist = px.histogram(df, x='Sentiment Value', title='Distribution of Sentiment Scores for Reddit Posts')
hist.show()
'''
# Additional plot
from datetime import datetime

# Given timestamp
timestamp = 1455725385.0

# Convert to datetime object
dt_object = datetime.fromtimestamp(timestamp)

# Print the datetime object
print("Datetime:", dt_object)
'''