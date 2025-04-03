from fasthtml.common import *
import datetime
import os
import sys

# Print module paths for debugging
print(f"DEBUG: Python path: {sys.path}")
print(f"DEBUG: Current dir: {os.getcwd()}")

try:
    from podscan_client.config import DB_CONFIG
    print(f"DEBUG: Loaded DB_CONFIG: {DB_CONFIG}")
except Exception as e:
    print(f"DEBUG: Failed to import DB_CONFIG: {str(e)}")
    # Fallback DB configuration
    DB_CONFIG = {"db_path": "podcast_sentiment.db"}

# Import SQLAlchemy models and session
from podscan_client.storage import DatabaseManager, Podcast, Episode, EntitySentiment, Base
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy import create_engine, func, or_, and_, inspect
from sqlalchemy.exc import SQLAlchemyError
import json

# Try to use a direct path if the relative one isn't working
db_file = os.path.join(os.getcwd(), "podcast_sentiment.db")
if os.path.exists(db_file):
    print(f"DEBUG: Found DB file at direct path: {db_file}")
    # Initialize the database manager with direct path
    db_manager = DatabaseManager(db_file)
else:
    print(f"DEBUG: Using default DB path from DatabaseManager")
    # Initialize the database manager
    db_manager = DatabaseManager()

print(f"DEBUG: Final DB path that will be used: {db_manager.db_path}")

# Create SQLAlchemy engine and session
engine = create_engine(f"sqlite:///{db_manager.db_path}")
Session = sessionmaker(bind=engine)

# Make sure tables exist
def ensure_tables_exist():
    """Ensure all required tables exist in the database"""
    try:
        # Create all tables defined in Base if they don't exist
        print("DEBUG: Creating tables if they don't exist")
        Base.metadata.create_all(engine)
        
        # Get all table names
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        print(f"DEBUG: Existing tables: {table_names}")
        
        return True
    except Exception as e:
        print(f"DEBUG: Error ensuring tables exist: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Create FastHTML app
app, rt = fast_app(
    hdrs=(
        Script("""
            function toggleDetails(id) {
                const detailRow = document.getElementById('detail-row-' + id);
                if (detailRow) {
                    // Toggle the detail row visibility
                    if (detailRow.style.display === 'none' || detailRow.style.display === '') {
                        detailRow.style.display = 'table-row';
                        highlightCompanyName(id);
                    } else {
                        detailRow.style.display = 'none';
                    }
                }
            }
            
            function searchTranscript(id) {
                const searchText = document.getElementById('search-text-' + id).value;
                highlightText(id, searchText);
            }
            
            function highlightCompanyName(id) {
                const companyNameElement = document.getElementById('company-name-' + id);
                if (!companyNameElement) {
                    console.error('Company name element not found for id: ' + id);
                    return;
                }
                
                const companyName = companyNameElement.innerText;
                highlightText(id, companyName);
            }
            
            function highlightText(id, text) {
                if (!text) return;
                
                const transcriptEl = document.getElementById('transcript-content-' + id);
                if (!transcriptEl) {
                    console.error('Transcript element not found for id: ' + id);
                    return;
                }
                
                const transcriptText = transcriptEl.innerHTML;
                
                // Reset any previous highlighting
                transcriptEl.innerHTML = transcriptText.replace(/<mark>|<\/mark>/g, '');
                
                if (!text.trim()) return;
                
                // Create a regex that's case insensitive
                const regex = new RegExp('(' + text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
                const highlightedText = transcriptEl.innerHTML.replace(regex, '<mark>$1</mark>');
                transcriptEl.innerHTML = highlightedText;
            }
        """),
        Style("""
            .sentiment-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 1rem;
            }
            
            .sentiment-table th, .sentiment-table td {
                padding: 0.5rem;
                text-align: left;
                border: 1px solid #ccc;
            }
            
            .sentiment-table th {
                background-color: #f0f0f0;
                position: sticky;
                top: 0;
            }
            
            .sentiment-table tr:hover {
                background-color: #f8f8f8;
                cursor: pointer;
            }
            
            .detail-row {
                display: none;
            }
            
            .detail-row td {
                padding: 0;
            }
            
            .detail-container {
                padding: 1rem;
                background-color: #f9f9f9;
                border-top: none;
            }
            
            .transcript-container {
                border: 1px solid #ccc;
                padding: 1rem;
                height: 250px;
                overflow-y: auto;
                background-color: white;
                margin-bottom: 1rem;
            }
            
            .search-container {
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
            }
            
            .search-container input {
                flex-grow: 1;
                margin-right: 0.5rem;
            }
            
            mark {
                background-color: yellow;
                padding: 0;
            }
            
            .edit-form {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .edit-form-buttons {
                grid-column: 1 / -1;
                display: flex;
                justify-content: flex-end;
                gap: 0.5rem;
                margin-top: 1rem;
            }
            
            .filter-container {
                display: flex;
                flex-direction: column;
                gap: 1rem;
                margin-bottom: 1rem;
                padding: 1rem;
                background-color: #f0f0f0;
                border-radius: 4px;
            }
            
            .filter-inputs {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 0.75rem;
                width: 100%;
            }
            
            .filter-inputs > div {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
            }
            
            .filter-inputs label {
                font-weight: bold;
                font-size: 0.9rem;
            }
            
            .filter-inputs select {
                width: 100%;
                padding: 0.5rem;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            
            .filter-buttons {
                display: flex;
                gap: 0.5rem;
                margin-top: 0.5rem;
                width: 100%;
                justify-content: flex-end;
            }
            
            .button {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                border-radius: 4px;
                border: none;
                cursor: pointer;
            }
            
            .button.secondary {
                background-color: #f0f0f0;
                color: #333;
                border: 1px solid #ccc;
            }
            
            .no-data-message {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 2rem;
                margin-top: 2rem;
                text-align: center;
            }
            
            .no-data-actions {
                margin-top: 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1rem;
            }
            
            .no-data-actions button {
                width: 250px;
            }
        """)
    )
)

def get_filters(req):
    """Extract filter parameters from request"""
    filters = {
        'podcast': req.query_params.get('podcast', ''),
        'episode': req.query_params.get('episode', ''),
        'company': req.query_params.get('company', ''),
        'ticker': req.query_params.get('ticker', ''),
        'sentiment': req.query_params.get('sentiment', ''),
    }
    print(f"DEBUG: Applied filters: {filters}")
    return filters

def build_filter_query(query, filters):
    """Add filters to SQLAlchemy query"""
    # Always filter for episode-level sentiments
    query = query.filter(EntitySentiment.is_episode_level == True)
    
    if filters['podcast'] and filters['podcast'].strip():
        print(f"DEBUG: Filtering podcast: {filters['podcast']}")
        query = query.filter(Podcast.name == filters['podcast'])
    
    if filters['episode'] and filters['episode'].strip():
        print(f"DEBUG: Filtering episode: {filters['episode']}")
        query = query.filter(Episode.title == filters['episode'])
    
    if filters['company'] and filters['company'].strip():
        print(f"DEBUG: Filtering company: {filters['company']}")
        query = query.filter(EntitySentiment.company_name == filters['company'])
    
    if filters['ticker'] and filters['ticker'].strip():
        print(f"DEBUG: Filtering ticker: {filters['ticker']}")
        query = query.filter(EntitySentiment.company_ticker == filters['ticker'])
    
    if filters['sentiment'] and filters['sentiment'].strip():
        print(f"DEBUG: Filtering sentiment: {filters['sentiment']}")
        if filters['sentiment'] == 'NONE':
            # Filter for records with no sentiment or empty sentiment
            query = query.filter(or_(
                EntitySentiment.sentiment == None,
                EntitySentiment.sentiment == ''
            ))
        else:
            query = query.filter(EntitySentiment.sentiment == filters['sentiment'])
    
    return query

def get_sentiments(filters):
    """Get sentiment data from database with filters using SQLAlchemy"""
    try:
        print(f"DEBUG: Opening DB connection to {db_manager.db_path}")
        print(f"DEBUG: DB file exists: {os.path.exists(db_manager.db_path)}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        
        session = Session()
        
        try:
            # Try the join query
            query = session.query(
                EntitySentiment.id,
                Podcast.name.label('podcast'),
                Episode.title.label('episode'),
                EntitySentiment.company_name,
                EntitySentiment.company_ticker,
                EntitySentiment.sentiment,
                EntitySentiment.gt_sentiment,
                EntitySentiment.snippet_text,
                EntitySentiment.comments,
                Episode.transcript,
                EntitySentiment.is_episode_level
            ).join(
                Episode, EntitySentiment.episode_id == Episode.id
            ).join(
                Podcast, Episode.podcast_id == Podcast.id
            )
            
            # Apply filters
            query = build_filter_query(query, filters)
            
            # Order by descending ID
            query = query.order_by(EntitySentiment.id.desc())
            
            # Limit to 100 rows
            query = query.limit(100)
            
            print(f"DEBUG: Executing join query")
            rows = query.all()
            print(f"DEBUG: Join query returned {len(rows)} rows")
            
            if rows:
                # Convert SQLAlchemy row objects to dictionaries
                result = [dict(row._mapping) for row in rows]
                session.close()
                return result
                
        except Exception as e:
            print(f"DEBUG: Error executing join query: {str(e)}")
            print("DEBUG: Falling back to simpler query...")
        
        # If the join query failed or returned no rows, try a simpler query
        try:
            # This query doesn't use joins
            simple_query = session.query(
                EntitySentiment.id,
                func.literal('Unknown').label('podcast'),
                func.literal('Unknown').label('episode'),
                EntitySentiment.company_name,
                EntitySentiment.company_ticker,
                EntitySentiment.sentiment,
                EntitySentiment.gt_sentiment,
                EntitySentiment.snippet_text,
                EntitySentiment.comments,
                EntitySentiment.snippet_text.label('transcript')
            )
            
            # Filter for episode-level sentiments for the simple query too
            simple_query = simple_query.filter(EntitySentiment.is_episode_level == True)
            
            # Apply only the filters that don't require joins
            if filters['company'] and filters['company'].strip():
                simple_query = simple_query.filter(EntitySentiment.company_name == filters['company'])
            
            if filters['ticker'] and filters['ticker'].strip():
                simple_query = simple_query.filter(EntitySentiment.company_ticker == filters['ticker'])
            
            if filters['sentiment'] and filters['sentiment'].strip():
                if filters['sentiment'] == 'NONE':
                    # Filter for records with no sentiment or empty sentiment
                    simple_query = simple_query.filter(or_(
                        EntitySentiment.sentiment == None,
                        EntitySentiment.sentiment == ''
                    ))
                else:
                    simple_query = simple_query.filter(EntitySentiment.sentiment == filters['sentiment'])
            
            simple_query = simple_query.order_by(EntitySentiment.id.desc())
            
            # Limit to 100 rows
            simple_query = simple_query.limit(100)
            
            print(f"DEBUG: Executing simple query")
            rows = simple_query.all()
            print(f"DEBUG: Simple query returned {len(rows)} rows")
            
            # Convert SQLAlchemy row objects to dictionaries
            result = [dict(row._mapping) for row in rows]
            session.close()
            return result
            
        except Exception as e:
            print(f"DEBUG: Error executing simple query: {str(e)}")
        
        session.close()
        return []
    except Exception as e:
        print(f"DEBUG: Database error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def has_data():
    """Check if there is any data in the database using SQLAlchemy"""
    try:
        print(f"DEBUG: Checking for data in {db_manager.db_path}")
        session = Session()
        
        # Count entities
        count = session.query(func.count(EntitySentiment.id)).scalar()
        print(f"DEBUG: EntitySentiment count = {count}")
        
        session.close()
        return count > 0
    except Exception as e:
        print(f"DEBUG: Error checking for data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_filter_options():
    """Get distinct values for dropdown filters"""
    session = Session()
    try:
        # Get distinct podcasts
        podcast_query = session.query(Podcast.name).distinct().order_by(Podcast.name)
        podcasts = [row[0] for row in podcast_query.all()]
        
        # Get distinct episodes
        episode_query = session.query(Episode.title).distinct().order_by(Episode.title)
        episodes = [row[0] for row in episode_query.all()]
        
        # Get distinct companies
        company_query = session.query(EntitySentiment.company_name).distinct().order_by(EntitySentiment.company_name)
        companies = [row[0] for row in company_query.all()]
        
        # Get distinct tickers
        ticker_query = session.query(EntitySentiment.company_ticker).distinct().order_by(EntitySentiment.company_ticker)
        tickers = [row[0] for row in ticker_query.all()]
        
        session.close()
        return {
            'podcasts': podcasts,
            'episodes': episodes,
            'companies': companies,
            'tickers': tickers
        }
    except Exception as e:
        print(f"DEBUG: Error getting filter options: {str(e)}")
        session.close()
        return {
            'podcasts': [],
            'episodes': [],
            'companies': [],
            'tickers': []
        }

def generate_filter_controls(filters):
    """Generate filter controls HTML"""
    # Get options for dropdown filters
    options = get_filter_options()
    
    return Form(
        P("Filter Data:", cls="filter-title"),
        Div(
            Div(
                Label("Podcast:"),
                Select(
                    Option("All Podcasts", value=""),
                    *[Option(podcast, value=podcast, selected=filters['podcast'] == podcast) 
                      for podcast in options['podcasts']],
                    name="podcast"
                )
            ),
            Div(
                Label("Episode:"),
                Select(
                    Option("All Episodes", value=""),
                    *[Option(episode, value=episode, selected=filters['episode'] == episode) 
                      for episode in options['episodes']],
                    name="episode"
                )
            ),
            Div(
                Label("Company:"),
                Select(
                    Option("All Companies", value=""),
                    *[Option(company, value=company, selected=filters['company'] == company) 
                      for company in options['companies']],
                    name="company"
                )
            ),
            Div(
                Label("Ticker:"),
                Select(
                    Option("All Tickers", value=""),
                    *[Option(ticker, value=ticker, selected=filters['ticker'] == ticker) 
                      for ticker in options['tickers']],
                    name="ticker"
                )
            ),
            Div(
                Label("Sentiment:"),
                Select(
                    Option("All Sentiments", value=""),
                    Option("No Sentiment", value="NONE"),
                    Option("Buy", value="Buy", selected=filters['sentiment'] == 'Buy'),
                    Option("Hold", value="Hold", selected=filters['sentiment'] == 'Hold'),
                    Option("Sell", value="Sell", selected=filters['sentiment'] == 'Sell'),
                    name="sentiment"
                )
            ),
            cls="filter-inputs"
        ),
        Div(
            Button("Apply Filters", type="submit"),
            A("Reset", href="/", cls="button secondary"),
            cls="filter-buttons"
        ),
        action="/",
        method="get",
        cls="filter-container"
    )

def generate_no_data_message():
    """Generate a message for when there's no data"""
    return Div(
        H2("No Data Found"),
        P("There are no records in the database. This could be due to:"),
        Ul(
            Li("The database is empty - you need to run the data collection process first"),
            Li("The database path is incorrect - check the configuration")
        ),
        Div(
            P("You can generate sample data to see how the application works:"),
            Form(
                Button("Generate Sample Data", type="submit", cls="primary"),
                action="/generate-sample-data",
                method="post",
                cls="sample-data-form"
            ),
            cls="no-data-actions"
        ),
        cls="no-data-message"
    )

def generate_detail_row(row):
    """Generate an expandable detail row for a sentiment"""
    highlighted_text = highlight_text(row['transcript'], row['company_name'])
    return Tr(
        Td(
            Div(
                # Search bar
                Div(
                    P("Search Transcript:"),
                    Div(
                        Input(id=f"search-text-{row['id']}", placeholder="Enter search text..."),
                        Button("Search", onclick=f"searchTranscript({row['id']})"),
                        cls="search-container"
                    )
                ),
                
                # Transcript
                Div(
                    Div(
                        P(highlighted_text, id=f"transcript-content-{row['id']}"),
                        cls="transcript-content"
                    ),
                    cls="transcript-container"
                ),
                
                # Edit form
                Form(
                    Div(
                        Div(
                            Label("Ground Truth Sentiment:"),
                            Select(
                                Option("", value=""),
                                Option("Buy", value="Buy", selected=row['gt_sentiment'] == 'Buy'),
                                Option("Hold", value="Hold", selected=row['gt_sentiment'] == 'Hold'),
                                Option("Sell", value="Sell", selected=row['gt_sentiment'] == 'Sell'),
                                name="gt_sentiment"
                            )
                        ),
                        Div(
                            Label("Comments:"),
                            Textarea(row['comments'] or '', name="comments", rows=3)
                        ),
                        Input(type="hidden", name="id", value=row['id']),
                        cls="edit-form"
                    ),
                    Div(
                        Button("Save Changes", type="submit"),
                        cls="edit-form-buttons"
                    ),
                    action="/save-changes",
                    method="post"
                ),
                cls="detail-container"
            ),
            colspan="7"
        ),
        id=f"detail-row-{row['id']}",
        cls="detail-row"
    )

def generate_table(rows, filters):
    """Generate the sentiment table HTML"""
    # Convert row data for easier use in templates
    processed_rows = []
    for row in rows:
        processed_row = dict(row)
        
        # Ensure transcript data is available
        if not processed_row.get('transcript') or processed_row['transcript'] == '':
            processed_row['transcript'] = processed_row.get('snippet_text', 'No transcript available')
        
        # Ensure snippet text is available
        if not processed_row.get('snippet_text') or processed_row['snippet_text'] == '':
            processed_row['snippet_text'] = 'No snippet available'
        
        # Ensure all needed fields have default values
        processed_row['gt_sentiment'] = processed_row.get('gt_sentiment', '')
        processed_row['comments'] = processed_row.get('comments', '')
        
        processed_rows.append(processed_row)
    
    print(f"DEBUG: Processed {len(processed_rows)} rows for display")
    
    # Create a flat list of rows interleaved with detail rows
    table_rows = []
    for row in processed_rows:
        # Add the main data row
        table_rows.append(
            Tr(
                Td(row['podcast']),
                Td(row['episode']),
                Td(row['company_name'], id=f"company-name-{row['id']}"),
                Td(row['company_ticker']),
                Td(row['sentiment']),
                Td(row['gt_sentiment'] or '', id=f"gt-sentiment-row-{row['id']}"),
                Td(row['comments'] or '', id=f"comments-row-{row['id']}"),
                onclick=f"toggleDetails({row['id']})"
            )
        )
        
        # Add the detail row (initially hidden)
        table_rows.append(generate_detail_row(row))
    
    return (
        generate_filter_controls(filters),
        Table(
            Thead(
                Tr(
                    Th("Podcast"),
                    Th("Episode"),
                    Th("Company"),
                    Th("Ticker"),
                    Th("Sentiment"),
                    Th("GT Sentiment"),
                    Th("Comments")
                )
            ),
            Tbody(*table_rows),
            cls="sentiment-table"
        )
    )

def get_sentiment_by_id(id):
    """Get a single sentiment by ID"""
    try:
        session = Session()
        sentiment = session.query(EntitySentiment).filter(EntitySentiment.id == id).first()
        
        if sentiment:
            # Convert SQLAlchemy entity to a dictionary
            row = {
                'id': sentiment.id,
                'company_name': sentiment.company_name,
                'company_ticker': sentiment.company_ticker,
                'sentiment': sentiment.sentiment,
                'gt_sentiment': sentiment.gt_sentiment,
                'snippet_text': sentiment.snippet_text,
                'comments': sentiment.comments,
                'is_episode_level': sentiment.is_episode_level
            }
            
            # Try to get episode and podcast info
            try:
                episode = session.query(Episode).filter(Episode.id == sentiment.episode_id).first()
                if episode:
                    row['episode'] = episode.title
                    row['transcript'] = episode.transcript
                    
                    podcast = session.query(Podcast).filter(Podcast.id == episode.podcast_id).first()
                    if podcast:
                        row['podcast'] = podcast.name
                    else:
                        row['podcast'] = 'Unknown'
                else:
                    row['episode'] = 'Unknown'
                    row['transcript'] = sentiment.snippet_text or 'No transcript available'
                    row['podcast'] = 'Unknown'
            except Exception as e:
                print(f"DEBUG: Error getting related data: {str(e)}")
                row['episode'] = 'Unknown'
                row['transcript'] = sentiment.snippet_text or 'No transcript available'
                row['podcast'] = 'Unknown'
            
            # Ensure transcript data is available
            if not row.get('transcript') or row['transcript'] == '':
                row['transcript'] = row.get('snippet_text', 'No transcript available')
            
            session.close()
            return row
        
        session.close()
        return None
    except Exception as e:
        print(f"DEBUG: Error getting sentiment by ID: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def highlight_text(text, search_term):
    """Highlight search term in text"""
    if not search_term or not text:
        return text
    
    import re
    pattern = re.escape(search_term)
    highlighted = re.sub(f'({pattern})', r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return highlighted

def generate_sample_data():
    """Generate sample data for the database using SQLAlchemy"""
    print("DEBUG: Starting sample data generation")
    session = Session()
    
    # Sample podcasts
    podcasts = [
        Podcast(id="pod1", name="Investing Insights"),
        Podcast(id="pod2", name="Market Movers"),
        Podcast(id="pod3", name="Stock Talk Daily")
    ]
    
    # Sample episodes
    episodes = [
        Episode(id="ep1", podcast_id="pod1", title="Tech Sector Analysis", 
                description="Discussion about tech stocks", 
                transcript="This is a full transcript about tech stocks and various companies.", 
                date=datetime.datetime.now()),
        Episode(id="ep2", podcast_id="pod2", title="Market Outlook 2023", 
                description="Economic forecast", 
                transcript="This is a full transcript about market outlook and various companies.", 
                date=datetime.datetime.now()),
        Episode(id="ep3", podcast_id="pod3", title="Earnings Season Review", 
                description="Q3 earnings analysis", 
                transcript="This is a full transcript about earnings and various companies.", 
                date=datetime.datetime.now())
    ]
    
    # Sample sentiments - all set as episode-level sentiments
    sentiments = [
        EntitySentiment(id=1, company_name="Apple", company_ticker="AAPL", episode_id="ep1", 
                       snippet_text="Apple had a great quarter with strong iPhone sales. Revenue increased by 15%.", 
                       sentiment="Buy", confidence=0.85, comments="Positive sentiment due to strong product lineup",
                       is_episode_level=True),
        EntitySentiment(id=2, company_name="Microsoft", company_ticker="MSFT", episode_id="ep1", 
                       snippet_text="Microsoft cloud business continues to grow with Azure leading the market.", 
                       sentiment="Buy", confidence=0.82, comments="Strong cloud growth momentum",
                       is_episode_level=True),
        EntitySentiment(id=3, company_name="Google", company_ticker="GOOGL", episode_id="ep1", 
                       snippet_text="Google ad revenue showing signs of slowing down amid competition.", 
                       sentiment="Hold", confidence=0.65, comments="Concerns about ad market",
                       is_episode_level=True),
        EntitySentiment(id=4, company_name="Tesla", company_ticker="TSLA", episode_id="ep2", 
                       snippet_text="Tesla facing production challenges and increased competition.", 
                       sentiment="Sell", confidence=0.75, comments="Valuation concerns and competition",
                       is_episode_level=True),
        EntitySentiment(id=5, company_name="Amazon", company_ticker="AMZN", episode_id="ep2", 
                       snippet_text="Amazon's e-commerce growth slowing but AWS remains strong.", 
                       sentiment="Hold", confidence=0.68, comments="Mixed results across business units",
                       is_episode_level=True),
        EntitySentiment(id=6, company_name="Netflix", company_ticker="NFLX", episode_id="ep3", 
                       snippet_text="Netflix subscriber growth exceeded expectations.", 
                       sentiment="Buy", confidence=0.78, comments="Strong content pipeline",
                       is_episode_level=True),
        EntitySentiment(id=7, company_name="Facebook", company_ticker="META", episode_id="ep3", 
                       snippet_text="Facebook's investment in the metaverse creates uncertainty.", 
                       sentiment="Hold", confidence=0.60, comments="High R&D spending with uncertain returns",
                       is_episode_level=True),
        EntitySentiment(id=8, company_name="Nvidia", company_ticker="NVDA", episode_id="ep3", 
                       snippet_text="Nvidia benefiting from AI boom with strong GPU demand.", 
                       sentiment="Buy", confidence=0.90, comments="Leader in AI chip market",
                       is_episode_level=True)
    ]
    
    success_count = 0
    
    try:
        # Ensure tables exist
        ensure_tables_exist()
        
        # Clear existing data
        print("DEBUG: Clearing existing data")
        try:
            session.query(EntitySentiment).delete()
            print("DEBUG: Cleared entity_sentiments")
        except Exception as e:
            print(f"DEBUG: Error clearing entity_sentiments: {str(e)}")
        
        try:
            session.query(Episode).delete()
            print("DEBUG: Cleared episodes")
        except Exception as e:
            print(f"DEBUG: Error clearing episodes: {str(e)}")
            
        try:
            session.query(Podcast).delete()
            print("DEBUG: Cleared podcasts")
        except Exception as e:
            print(f"DEBUG: Error clearing podcasts: {str(e)}")
        
        # Insert podcasts
        print("DEBUG: Inserting podcasts")
        for podcast in podcasts:
            try:
                session.add(podcast)
                success_count += 1
                print(f"DEBUG: Added podcast {podcast.name}")
            except Exception as e:
                print(f"DEBUG: Error adding podcast {podcast.name}: {str(e)}")
        
        # Insert episodes
        print("DEBUG: Inserting episodes")
        for episode in episodes:
            try:
                session.add(episode)
                success_count += 1
                print(f"DEBUG: Added episode {episode.title}")
            except Exception as e:
                print(f"DEBUG: Error adding episode {episode.title}: {str(e)}")
        
        # Insert sentiments
        print("DEBUG: Inserting sentiments")
        for sentiment in sentiments:
            try:
                session.add(sentiment)
                success_count += 1
                print(f"DEBUG: Added sentiment for {sentiment.company_name}")
            except Exception as e:
                print(f"DEBUG: Error adding sentiment for {sentiment.company_name}: {str(e)}")
        
        session.commit()
        print(f"DEBUG: Successfully committed {success_count} records")
        session.close()
        return {"success": True, "count": len(sentiments)}
    
    except Exception as e:
        print(f"DEBUG: Error in generate_sample_data: {str(e)}")
        import traceback
        traceback.print_exc()
        session.rollback()
        session.close()
        return {"success": False, "error": str(e)}

@rt("/")
def get(req):
    """Main page route handler"""
    # Ensure tables exist
    ensure_tables_exist()
    
    if not has_data():
        return Titled(
            "Podcast Sentiment Validation - Episode Level",
            generate_no_data_message()
        )
    
    filters = get_filters(req)
    print(f"DEBUG: Processing request with filters: {filters}")
    
    rows = get_sentiments(filters)
    print(f"DEBUG: Retrieved {len(rows)} rows after filtering")
    
    if not rows:
        # No rows found with current filters
        return Titled(
            "Podcast Sentiment Validation - Episode Level",
            generate_filter_controls(filters),
            Div(
                H3("No Results Found"),
                P("No results match your current filter criteria."),
                A("Reset Filters", href="/", cls="button primary"),
                cls="no-data-message"
            )
        )
    
    return Titled(
        "Podcast Sentiment Validation - Episode Level",
        generate_table(rows, filters)
    )

@rt("/save-changes", methods=["POST"])
async def post(req):
    """Save changes to sentiment data using SQLAlchemy"""
    # Get form data
    form_data = await req.form()
    id = form_data.get('id', '')
    gt_sentiment = form_data.get('gt_sentiment', '')
    comments = form_data.get('comments', '')
    
    if not id:
        return {"success": False, "error": "Missing ID"}
    
    session = Session()
    
    try:
        # Find the sentiment by ID
        sentiment = session.query(EntitySentiment).filter(EntitySentiment.id == int(id)).first()
        
        if sentiment:
            # Update fields
            sentiment.gt_sentiment = gt_sentiment
            sentiment.comments = comments
            
            # Commit changes
            session.commit()
            success = True
        else:
            success = False
    except Exception as e:
        session.rollback()
        success = False
        print(f"DEBUG: Error saving changes: {str(e)}")
    finally:
        session.close()
    
    # Redirect back to index page
    return Redirect("/")

@rt("/generate-sample-data", methods=["POST"])
async def post(req):
    """Generate sample data for testing"""
    result = generate_sample_data()
    
    if result["success"]:
        return Redirect("/")
    else:
        return Titled(
            "Error",
            H2("Error Generating Sample Data"),
            P(f"There was an error generating sample data: {result.get('error', 'Unknown error')}"),
            A("Back to Home", href="/", cls="button")
        )

# Ensure tables exist when app starts
ensure_tables_exist()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
