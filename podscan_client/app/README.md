# Podcast Sentiment Validation App

This FastHTML application allows you to view, edit, and validate sentiment data from podcast transcripts. It provides a user-friendly interface for reviewing sentiment analysis results and making corrections.

## Features

- View sentiment data in a sortable table
- Filter data by podcast, episode, company, ticker, and sentiment
- Double-click on a row to view the full transcript
- Search and highlight text within transcripts (company name is highlighted by default)
- Edit ground truth sentiment and add comments
- Navigate between entries in filtered view
- Save changes directly to the database

## Usage

### Running the Application

```bash
python -m podscan_client.app.data_validation_app
```

The app will be available at http://localhost:8000

### Navigating the Interface

1. **Main Table**: Displays all sentiment entries with columns for podcast, episode, company, ticker, sentiment, ground truth sentiment, and comments.

2. **Filtering**: Use the filter controls at the top to narrow down the data. You can filter by:
   - Podcast name
   - Episode title
   - Company name
   - Ticker symbol
   - Sentiment value

3. **Viewing Transcripts**: Double-click on any row to open a popup with the full transcript.

4. **Editing Data**: Inside the popup, you can:
   - Edit the ground truth sentiment (Buy, Hold, Sell)
   - Add or edit comments
   - Click "Save Changes" to update the database

5. **Transcript Navigation**: While in the popup:
   - Use the "Previous" and "Next" buttons to navigate between entries in the current filtered view
   - Use the search box to find and highlight specific text within the transcript
   - The company name is highlighted by default

## Data Structure

The app reads data from the SQLite database defined in `podscan_client.storage`. The main tables used are:

- `podcasts`: Podcast information
- `episodes`: Episode details including transcripts
- `entity_sentiments`: Sentiment analysis results for companies mentioned in episodes

Only the `gt_sentiment` and `comments` fields are editable; all other data is read-only.

## Purpose

This tool is designed to help improve the sentiment analysis pipeline by allowing manual validation and correction of the automated sentiment results. By reviewing and correcting the sentiment data, you can:

1. Identify patterns of incorrect sentiment classifications
2. Build a ground truth dataset for model evaluation
3. Collect examples for prompt improvement
4. Gain insights into how different companies are discussed in podcasts 