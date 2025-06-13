import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime as dt

# Load your data
df = pd.read_csv('data/reddit_data_recent_sentiment.csv', parse_dates=['timestamp'])

# Add emotion categories based on sentiment scores
def categorize_emotion(row):
    if row['flair'] >= 0.9:
        return 'Very Positive'
    elif row['flair'] >= 0.5:
        return 'Positive'
    elif row['flair'] <= -0.9:
        return 'Very Negative'
    elif row['flair'] <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

df['emotion'] = df.apply(categorize_emotion, axis=1)

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Reddit Sentiment Analysis Dashboard"

# Layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Reddit Sentiment Analysis Dashboard", className="text-center my-4"))),
    
    dbc.Row([
        dbc.Col([
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df['timestamp'].min(),
                max_date_allowed=df['timestamp'].max(),
                start_date=df['timestamp'].min(),
                end_date=df['timestamp'].max()
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='sentiment-trend'), lg=6),
        dbc.Col(dcc.Graph(id='emotion-distribution'), lg=6)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='heatmap'), lg=6),
        dbc.Col(dcc.Graph(id='radar-chart'), lg=6)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='model-comparison'), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(id='top-posts'), width=12)
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output('sentiment-trend', 'figure'),
     Output('emotion-distribution', 'figure'),
     Output('heatmap', 'figure'),
     Output('radar-chart', 'figure'),
     Output('model-comparison', 'figure'),
     Output('top-posts', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_dashboard(start_date, end_date):
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    # 1. Sentiment Trend
    daily_avg = filtered_df.resample('D', on='timestamp').mean(numeric_only=True).reset_index()
    
    trend_fig = px.line(daily_avg, x='timestamp', y=['flair', 'vader_compound'],
                        title='Daily Sentiment Trend',
                        labels={'value': 'Sentiment Score', 'timestamp': 'Date'},
                        color_discrete_map={'flair': '#636EFA', 'vader_compound': '#EF553B'})
    trend_fig.update_layout(hovermode='x unified')
    trend_fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    # 2. Emotion Distribution
    emotion_counts = filtered_df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    
    emotion_fig = px.pie(emotion_counts, values='count', names='emotion',
                         title='Emotion Distribution',
                         color='emotion',
                         color_discrete_map={
                             'Very Positive': '#2CA02C',
                             'Positive': '#98DF8A',
                             'Neutral': '#FF7F0E',
                             'Negative': '#D62728',
                             'Very Negative': '#8B0000'
                         })
    
    # 3. Heatmap
    filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    heatmap_data = filtered_df.pivot_table(values='flair', 
                                         index=filtered_df['timestamp'].dt.date, 
                                         columns='hour', 
                                         aggfunc='mean')
    
    heatmap_fig = px.imshow(heatmap_data,
                           labels=dict(x="Hour of Day", y="Date", color="Sentiment"),
                           title='Hourly Sentiment Heatmap',
                           color_continuous_scale='RdBu',
                           color_continuous_midpoint=0)
    heatmap_fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Date")
    
    # 4. Radar Chart (Emotion Breakdown)
    radar_data = filtered_df.groupby('emotion')[['flair', 'tb_polarity', 'vader_compound']].mean().reset_index()
    
    radar_fig = go.Figure()
    
    for emotion in radar_data['emotion']:
        radar_fig.add_trace(go.Scatterpolar(
            r=radar_data[radar_data['emotion'] == emotion][['flair', 'tb_polarity', 'vader_compound']].values[0],
            theta=['Flair', 'TextBlob', 'VADER'],
            fill='toself',
            name=emotion
        ))
    
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )),
        title='Sentiment Model Comparison by Emotion',
        showlegend=True
    )
    
    # 5. Model Comparison
    model_fig = px.scatter_matrix(filtered_df,
                                 dimensions=['flair', 'tb_polarity', 'vader_compound'],
                                 color='emotion',
                                 title='Sentiment Model Correlation',
                                 color_discrete_map={
                                     'Very Positive': '#2CA02C',
                                     'Positive': '#98DF8A',
                                     'Neutral': '#FF7F0E',
                                     'Negative': '#D62728',
                                     'Very Negative': '#8B0000'
                                 })
    
    # 6. Top Posts Table
    top_positive = filtered_df.nlargest(3, 'flair')[['timestamp', 'flair', 'emotion']]
    top_negative = filtered_df.nsmallest(3, 'flair')[['timestamp', 'flair', 'emotion']]
    
    top_posts = dbc.Card([
        dbc.CardHeader("Most Extreme Posts"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Most Positive", className="text-success"),
                    dbc.Table.from_dataframe(top_positive, striped=True, bordered=True, hover=True)
                ]),
                dbc.Col([
                    html.H5("Most Negative", className="text-danger"),
                    dbc.Table.from_dataframe(top_negative, striped=True, bordered=True, hover=True)
                ])
            ])
        ])
    ])
    
    return trend_fig, emotion_fig, heatmap_fig, radar_fig, model_fig, top_posts

if __name__ == '__main__':
    app.run(debug=True, port=8050)  # Changed from app.run_server() to app.run()