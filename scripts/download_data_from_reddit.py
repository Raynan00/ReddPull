import praw
import datetime
import pandas as pd
import time
import logging
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedditDataCollector:
    def __init__(self):
        self.reddit = self._authenticate()
        self.earliest_date = None
        self.latest_date = None
        
    def _authenticate(self):
        """Authenticate with Reddit API"""
        try:
            reddit = praw.Reddit(
                client_id='vgo6NOPjghh3OkLdgkk8Qw',
                client_secret='9nH5dW38AKxhEQe57b6cU0Knae7meg',
                user_agent='script:reddit_sentiment_analysis:v1.0 (by /u/GLI7CH_00)'
            )
            reddit.user.me()  # Test authentication
            logger.info("✅ Successfully authenticated with Reddit API")
            return reddit
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return None

    def fetch_submissions(self, subreddit, start_date, end_date, query=None, limit=1000):
        """Fetch submissions with improved date handling"""
        if not self.reddit:
            return []

        submissions = []
        date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        logger.info(f"📅 Fetching posts from {date_range}")
        
        try:
            sub = self.reddit.subreddit(subreddit)
            results = sub.search(query, sort='new', limit=limit) if query else sub.new(limit=limit)
            
            for submission in tqdm(results, desc="Processing posts"):
                post_date = datetime.datetime.fromtimestamp(submission.created_utc)
                
                # Track date boundaries
                self.earliest_date = post_date if not self.earliest_date else min(self.earliest_date, post_date)
                self.latest_date = post_date if not self.latest_date else max(self.latest_date, post_date)
                
                if start_date <= post_date <= end_date:
                    submissions.append(self._process_submission(submission))
                elif post_date < start_date:
                    break  # Stop if we've gone past our start date

            logger.info(f"📊 Found {len(submissions)} posts in date range")
            logger.info(f"ℹ️ Earliest post found: {self.earliest_date}")
            logger.info(f"ℹ️ Latest post found: {self.latest_date}")
            
        except Exception as e:
            logger.error(f"❌ Error fetching submissions: {e}")
            
        return submissions

    def _process_submission(self, submission):
        """Process submission data with error handling"""
        try:
            return {
                'id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext.replace('\n', ' ') if submission.selftext else '',
                'url': submission.url,
                'author': str(submission.author),
                'score': submission.score,
                'created_utc': submission.created_utc,
                'num_comments': submission.num_comments,
                'permalink': submission.permalink,
                'flair': getattr(submission, 'link_flair_text', None),
                'publish_date': datetime.datetime.fromtimestamp(submission.created_utc)
            }
        except Exception as e:
            logger.error(f"⚠️ Error processing submission: {e}")
            return None

def main():
    # Configuration - try with very recent dates first
    config = {
        'subreddit': 'python',
        'query': None,  # Try with None first
        'start_date': datetime.datetime.now() - datetime.timedelta(days=7),  # Last 7 days
        'end_date': datetime.datetime.now(),
        'output_file': 'reddit_data_recent.csv'
    }
    
    logger.info("🚀 Starting Reddit data collection")
    collector = RedditDataCollector()
    
    if not collector.reddit:
        return

    submissions = collector.fetch_submissions(
        subreddit=config['subreddit'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        query=config['query']
    )
    
    if submissions:
        df = pd.DataFrame(submissions)
        df.to_csv(config['output_file'], index=False)
        logger.info(f"💾 Saved {len(df)} posts to {config['output_file']}")
        logger.info(f"📅 Actual date range saved: {df['publish_date'].min()} to {df['publish_date'].max()}")
    else:
        logger.warning("⚠️ No posts found. Possible reasons:")
        logger.warning(f"- Subreddit r/{config['subreddit']} may not have posts in {config['start_date']} to {config['end_date']}")
        logger.warning("- Try a more popular subreddit like 'askreddit' for testing")
        logger.warning("- Try a broader date range")
        
        if collector.earliest_date:
            logger.info(f"ℹ️ Earliest post available: {collector.earliest_date}")
        if collector.latest_date:
            logger.info(f"ℹ️ Latest post available: {collector.latest_date}")

if __name__ == '__main__':
    main()