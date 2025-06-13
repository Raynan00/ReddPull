
# ReddPull - Reddit Sentiment Analysis Toolkit By RAYNAN WUYEP


A complete pipeline for scraping Reddit posts, performing sentiment analysis, and visualizing results through an interactive dashboard.

## Features

- **Reddit Data Collection**: Scrape posts from any subreddit with date filtering
- **Multi-Model Sentiment Analysis**: 
  - Flair NLP
  - TextBlob 
  - VADER
- **Interactive Dashboard**:
  - Temporal sentiment trends
  - Emotion classification
  - Model comparison
  - Hourly heatmaps
- **Automated Processing**: Clean and bucketize results

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reddit-sentiment-analysis.git
   cd reddit-sentiment-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Collection
```bash
python download_data_from_reddit.py
```
**Configuration**: Edit the script to set:
- `sub_reddit` - Target subreddit
- `start_date`/`end_date` - Date range
- `output_filename` - CSV output path

### 2. Sentiment Analysis
```bash
python reddit_post_sentiment_analysis.py
```
Processes `reddit_data.csv` and generates:
- `reddit_data_sentiment.csv` (raw results)
- `reddit_data_sentiment_bucketized.csv` (hourly aggregates)

### 3. Interactive Dashboard
```bash
python interactive_dashboard.py
```
Access at: `http://localhost:8050`

## File Structure
```
├── data/                    # Example data
│   ├── reddit_data.csv               
│   └── reddit_data_sentiment.csv
├── scripts/
│   ├── download_data_from_reddit.py    # Reddit scraper
│   ├── reddit_post_sentiment_analysis.py # Sentiment processor
│   └── interactive_dashboard.py        # Visualization app
├── requirements.txt         # Python dependencies
└── README.md
```

## Configuration

### Reddit API Setup
1. Create a Reddit app at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Add credentials to `.env`:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT="script:reddit-sentiment:v1.0 (by /u/yourusername)"
   ```

### Customization Options
- **Date ranges**: Adjust in each script
- **Subreddits**: Change target communities
- **Visualizations**: Modify `interactive_dashboard.py`

## Example Workflow

1. Collect 1 month of r/technology posts:
   ```python
   # In download_data_from_reddit.py
   sub_reddit = 'technology'
   start_date = datetime(2023, 1, 1)
   end_date = datetime(2023, 1, 31)
   ```

2. Analyze sentiment:
   ```bash
   python reddit_post_sentiment_analysis.py
   ```

3. Explore results:
   ```bash
   python interactive_dashboard.py
   ```

## Troubleshooting

**Common Issues**:
- `403 Forbidden`: Verify Reddit API credentials
- `ModuleNotFoundError`: Run `pip install -r requirements.txt`
- No data returned: Adjust date range or try different subreddit



### Recommended Repository Structure:
```
reddit-sentiment-analysis/
├── data/
│   ├── reddit_data.csv
│   └── reddit_data_sentiment.csv
├── scripts/
│   ├── download_data_from_reddit.py
│   ├── reddit_post_sentiment_analysis.py
│   └── interactive_dashboard.py
├── .env.example
├── requirements.txt
├── LICENSE
└── README.md
```

### Additional Recommended Files:

1. `requirements.txt`:
```
pandas>=1.3.0
praw>=7.5.0
flair>=0.11.0
textblob>=0.15.3
vaderSentiment>=3.3.2
plotly>=5.5.0
dash>=2.0.0
dash-bootstrap-components>=1.0.0
python-dotenv>=0.19.0
tqdm>=4.62.0
```

2. `.env.example`:
```
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT="script:reddit-sentiment:v1.0 (by /u/yourusername)"
```
