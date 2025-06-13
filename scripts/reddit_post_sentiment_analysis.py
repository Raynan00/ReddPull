import pandas as pd
import flair
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import datetime
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize sentiment analyzers
try:
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
except Exception as e:
    logger.error(f"Failed to load Flair model: {e}")
    flair_sentiment = None

sid = SentimentIntensityAnalyzer()

class SentimentAnalyzer:
    def __init__(self):
        self.flair_model = flair_sentiment
        self.sid = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text):
        """Perform all sentiment analyses on a single text"""
        if not text.strip():
            return None
            
        results = {}
        
        # Flair sentiment
        if self.flair_model:
            try:
                sentence = flair.data.Sentence(text)
                self.flair_model.predict(sentence)
                label = sentence.labels[0]
                score = label.score
                results['flair'] = -score if label.value == 'NEGATIVE' else score
            except Exception as e:
                logger.warning(f"Flair analysis failed: {e}")
                results['flair'] = None
        
        # TextBlob sentiment
        try:
            tb = TextBlob(text)
            results['tb_polarity'] = tb.sentiment.polarity
            results['tb_subjectivity'] = tb.sentiment.subjectivity
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            results['tb_polarity'] = None
            results['tb_subjectivity'] = None
        
        # VADER sentiment
        try:
            vader = self.sid.polarity_scores(text)
            results['vader_pos'] = vader['pos']
            results['vader_neg'] = vader['neg']
            results['vader_neu'] = vader['neu']
            results['vader_compound'] = vader['compound']
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            results.update({
                'vader_pos': None,
                'vader_neg': None,
                'vader_neu': None,
                'vader_compound': None
            })
        
        return results

def process_batch(batch, analyzer):
    """Process a batch of texts with sentiment analysis"""
    results = []
    for _, row in batch.iterrows():
        text = row['text']
        timestamp = row.name  # Using index which should be publish_date
        
        sentiment = analyzer.analyze_text(text)
        if sentiment:
            sentiment['timestamp'] = timestamp
            sentiment['text'] = text  # Optional: include text for debugging
            results.append(sentiment)
    
    return pd.DataFrame(results)

def get_sentiment_report(input_filename, output_filename, batch_size=1000):
    """Generate sentiment report with parallel processing"""
    logger.info(f"Loading data from {input_filename}")
    try:
        df = pd.read_csv(input_filename, parse_dates=['publish_date'])
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return
    
    # Preprocess data
    df = df[['title', 'selftext', 'publish_date']].copy()
    df['text'] = (df['title'] + ' ' + df['selftext']).str.replace('\n', ' ').str.strip()
    df = df[df['text'].str.len() > 0]  # Remove empty texts
    df.set_index('publish_date', inplace=True)
    
    analyzer = SentimentAnalyzer()
    
    # Process in batches (parallel)
    batches = np.array_split(df, len(df) // batch_size + 1)
    results = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in batches:
            futures.append(executor.submit(process_batch, batch, analyzer))
        
        for future in tqdm(futures, desc="Processing batches"):
            try:
                batch_result = future.result()
                results.append(batch_result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
    
    if not results:
        logger.warning("No results generated")
        return
    
    # Combine results
    final_df = pd.concat(results)
    final_df = final_df[['timestamp', 'flair', 'tb_polarity', 'tb_subjectivity', 
                         'vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']]
    
    # Save results
    file_exists = os.path.exists(output_filename)
    final_df.to_csv(output_filename, mode='a', header=not file_exists, index=False)
    logger.info(f"Saved sentiment results to {output_filename}")

def clean_sentiment_report(input_filename, output_filename):
    """Clean and deduplicate sentiment report"""
    logger.info(f"Cleaning sentiment report {input_filename}")
    try:
        df = pd.read_csv(input_filename, parse_dates=['timestamp'])
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp')
        df.to_csv(output_filename, index=False)
        logger.info(f"Saved cleaned report to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to clean report: {e}")

def bucketize_sentiment_report(input_filename, output_filename, freq='H'):
    """Aggregate sentiment data by time buckets"""
    logger.info(f"Bucketizing sentiment data with frequency {freq}")
    try:
        df = pd.read_csv(input_filename, parse_dates=['timestamp'])
        
        # Create complete time range
        start = df['timestamp'].min().floor(freq)
        end = df['timestamp'].max().ceil(freq)
        full_range = pd.date_range(start=start, end=end, freq=freq)
        
        # Group by time bucket
        grouped = df.groupby(pd.Grouper(key='timestamp', freq=freq))
        
        # Calculate aggregated metrics
        result = grouped.agg({
            'flair': ['mean', 'count'],
            'tb_polarity': ['mean', 'count'],
            'tb_subjectivity': ['mean', 'count'],
            'vader_pos': ['mean', 'count'],
            'vader_neg': ['mean', 'count'],
            'vader_neu': ['mean', 'count'],
            'vader_compound': ['mean', 'count']
        })
        
        # Reindex to ensure all time periods are included
        result = result.reindex(full_range)
        
        # Flatten multi-index columns
        result.columns = ['_'.join(col).strip() for col in result.columns.values]
        
        # Fill NA counts with 0 and means with appropriate values
        count_cols = [col for col in result.columns if col.endswith('_count')]
        mean_cols = [col for col in result.columns if col.endswith('_mean')]
        
        result[count_cols] = result[count_cols].fillna(0)
        result[mean_cols] = result[mean_cols].fillna(0)  # Or other appropriate fill value
        
        result.to_csv(output_filename)
        logger.info(f"Saved bucketized data to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to bucketize data: {e}")

if __name__ == '__main__':
    input_filename = 'reddit_data_recent.csv'
    output_sentiment_filename = os.path.splitext(input_filename)[0] + '_sentiment.csv'
    output_bucketized_filename = os.path.splitext(output_sentiment_filename)[0] + '_bucketized.csv'
    
    # Step 1: Perform sentiment analysis
    get_sentiment_report(input_filename, output_sentiment_filename)
    
    # Step 2: Clean results
    clean_sentiment_report(output_sentiment_filename, output_sentiment_filename)
    
    # Step 3: Bucketize results
    bucketize_sentiment_report(output_sentiment_filename, output_bucketized_filename)