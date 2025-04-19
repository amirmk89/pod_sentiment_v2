import logging
import os
import re
import sys
import streamlit as st
import pandas as pd
import datetime
import json
from sqlalchemy import create_engine, func, or_, and_
from sqlalchemy.orm import sessionmaker
from functools import lru_cache

logger = logging.getLogger(__name__)

# Add project root to path to resolve imports
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Import database models
from podscan_client.storage import DatabaseManager, Podcast, Episode, EntitySentiment, Base

# Initialize database connection
def init_db():
    """Initialize database connection"""
    # Try all possible DB locations
    possible_paths = [
        '/Users/amirmarkovitz/work/podrag/podscan_client/podcast_sentiment.db',
        # os.path.join(os.getcwd(), "podcast_sentiment.db"),                  # Current working dir 
        # os.path.join(os.path.dirname(os.getcwd()), "podcast_sentiment.db"), # One level up
        # os.path.join(root_dir, "podcast_sentiment.db"),                     # Project root
        # os.path.join(root_dir, "podscan_client", "podcast_sentiment.db")    # Inside podscan_client dir
    ]
    
    db_path = None
    for path in possible_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if db_path:
        st.success(f"Connected to database at: {db_path}")
        db_manager = DatabaseManager(db_path)
        engine = create_engine(f"sqlite:///{db_manager.db_path}")
        Session = sessionmaker(bind=engine)
        return db_manager, engine, Session
    else:
        paths_checked = "\n".join(possible_paths)
        st.error(f"Database not found. Checked these locations:\n{paths_checked}")
        return None, None, None

# Query data with filters - add caching
@st.cache_data(ttl=900)
def get_filtered_data(_session, filters_tuple):
    """Get sentiment data with filters"""
    # Convert tuple back to dict for use
    filters = dict(filters_tuple)
    
    query = _session.query(
        EntitySentiment.id,
        Podcast.name.label('podcast'),
        Episode.title.label('episode'),
        Episode.date.label('date'),
        EntitySentiment.company_name,
        EntitySentiment.company_ticker,
        EntitySentiment.sentiment,
        EntitySentiment.gt_sentiment,
        EntitySentiment.snippet_text,
        EntitySentiment.comments,
        Episode.transcript,
        EntitySentiment.confidence,
        EntitySentiment.is_episode_level,
        EntitySentiment.is_mock,
        EntitySentiment.processed_at,
        EntitySentiment.meta_data
    ).join(
        Episode, EntitySentiment.episode_id == Episode.id
    ).join(
        Podcast, Episode.podcast_id == Podcast.id
    )
    
    # Apply filters
    if filters.get('date_range'):
        start_date, end_date = filters['date_range']
        if start_date:
            query = query.filter(Episode.date >= start_date)
        if end_date:
            query = query.filter(Episode.date <= end_date)
    
    if filters.get('company'):
        query = query.filter(EntitySentiment.company_name == filters['company'])
    
    if filters.get('podcast'):
        query = query.filter(Podcast.name == filters['podcast'])
    
    if filters.get('sentiment'):
        query = query.filter(EntitySentiment.sentiment == filters['sentiment'])
        
    if filters.get('search'):
        search_term = f"%{filters['search']}%"
        query = query.filter(or_(
            Episode.title.ilike(search_term),
            EntitySentiment.company_name.ilike(search_term),
            EntitySentiment.snippet_text.ilike(search_term)
        ))
    
    # Order by date descending
    query = query.order_by(Episode.date.desc())
    
    return query.all()

# Cache filter options
@st.cache_data(ttl=1600)
def get_filter_options(_session):
    """Get distinct values for filters"""
    podcasts = [row[0] for row in _session.query(Podcast.name).distinct().order_by(Podcast.name).all()]
    companies = [row[0] for row in _session.query(EntitySentiment.company_name).distinct().order_by(EntitySentiment.company_name).all()]
    sentiments = [row[0] for row in _session.query(EntitySentiment.sentiment).distinct().filter(EntitySentiment.sentiment != None).filter(EntitySentiment.sentiment != '').order_by(EntitySentiment.sentiment).all()]
    
    return {
        'podcasts': podcasts,
        'companies': companies,
        'sentiments': sentiments
    }

# Update sentiment record
def update_sentiment(session, sentiment_id, gt_sentiment, comments, transcript=None):
    """Update gt_sentiment, comments, and transcript fields for a sentiment record"""
    try:
        sentiment = session.query(EntitySentiment).filter(EntitySentiment.id == sentiment_id).first()
        if sentiment:
            # Update sentiment fields
            sentiment.gt_sentiment = gt_sentiment
            sentiment.comments = comments
            
            # If transcript provided, update the episode's transcript
            if transcript is not None:
                episode = session.query(Episode).filter(Episode.id == sentiment.episode_id).first()
                if episode:
                    episode.transcript = transcript
            
            session.commit()
            return True
        return False
    except Exception as e:
        print(f"Error updating sentiment: {e}")
        logger.error(f"Error updating sentiment: {e}")
        st.error(f"Error updating sentiment: {e}")

        session.rollback()
        return False

# Highlight text in a string (case-insensitive)
def highlight_text(text, search_term):
    """Highlight search term in text (case-insensitive)"""
    if not search_term or not text:
        return text
    
    import re
    pattern = re.compile(f'({re.escape(search_term)})', re.IGNORECASE)
    # Count occurrences
    matches = re.findall(pattern, text)
    count = len(matches)
    return pattern.sub(r'<mark style="background-color: yellow; color: black;">\1</mark>', text), count

# Function to load sentiment edits from file
def load_sentiment_edits():
    edits_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_edits.json")
    if os.path.exists(edits_file):
        try:
            with open(edits_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading sentiment edits: {e}")
            return {}
    return {}

# Function to save sentiment edits to file
def save_sentiment_edit(sentiment_id, edit_data):
    edits_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_edits.json")
    
    # Convert to dict with string keys for JSON serialization
    sentiment_id = str(sentiment_id)
    
    # Load existing edits
    all_edits = load_sentiment_edits()
    
    # Update with new edit
    if sentiment_id not in all_edits:
        all_edits[sentiment_id] = []
    
    # Add this edit to the history for this sentiment ID
    all_edits[sentiment_id].append(edit_data)
    
    # Save back to file
    try:
        with open(edits_file, 'w') as f:
            json.dump(all_edits, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving sentiment edits: {e}")
        return False

# Main app
def main():
    st.set_page_config(
        page_title="Podcast Sentiment Validator",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("Podcast Sentiment Validator")
    
    # Initialize database
    db_manager, engine, Session = init_db()
    if not db_manager:
        st.stop()
    
    session = Session()
    
    # Initialize session state for filters and UI state
    if 'filters' not in st.session_state:
        st.session_state.filters = {}
    if 'expanded_rows' not in st.session_state:
        st.session_state.expanded_rows = set()
    if 'table_data' not in st.session_state:
        st.session_state.table_data = None
    if 'last_filters' not in st.session_state:
        st.session_state.last_filters = {}
    # Add backup for sentiment edits
    if 'sentiment_edits' not in st.session_state:
        st.session_state.sentiment_edits = {}
    # Load persistent sentiment edits
    if 'persistent_edits' not in st.session_state:
        st.session_state.persistent_edits = load_sentiment_edits()
        
    # Safety check - if expanded_rows isn't a set, initialize it
    if not isinstance(st.session_state.get('expanded_rows'), set):
        st.session_state.expanded_rows = set()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Get filter options
    options = get_filter_options(session)
    
    # Date range filter
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", value=None)
    with col2:
        end_date = st.date_input("To", value=None)
    
    # Convert to datetime if not None
    if start_date:
        start_date = datetime.datetime.combine(start_date, datetime.time.min)
    if end_date:
        end_date = datetime.datetime.combine(end_date, datetime.time.max)
    
    date_range = (start_date, end_date) if start_date or end_date else None
    
    # Company filter
    company = st.sidebar.selectbox("Company", [""] + options['companies'])
    
    # Podcast filter
    podcast = st.sidebar.selectbox("Podcast", [""] + options['podcasts'])
    
    # Sentiment filter
    sentiment = st.sidebar.selectbox("Sentiment", [""] + options['sentiments'])
    
    # Text search with no rerender behavior
    search = st.sidebar.text_input("Search (title, company, or transcript)")
    
    # Apply filters button
    apply_filters = st.sidebar.button("Apply Filters")
    
    # Reset filters button
    reset_filters = st.sidebar.button("Reset Filters")
    
    # Handle filter changes
    if reset_filters:
        st.session_state.filters = {}
        st.session_state.expanded_rows = set()
        st.session_state.last_filters = {}
        st.rerun()
    
    # Build filters dictionary
    current_filters = {
        'date_range': date_range,
        'company': company if company else None,
        'podcast': podcast if podcast else None,
        'sentiment': sentiment if sentiment else None,
        'search': search if search else None
    }
    
    # Remove None values
    current_filters = {k: v for k, v in current_filters.items() if v is not None}
    
    # Only update filters when Apply is clicked or if filters were reset
    if apply_filters or st.session_state.filters != st.session_state.last_filters:
        st.session_state.filters = current_filters
        st.session_state.last_filters = current_filters.copy()
    
    # Convert dict to hashable tuple for caching
    filters_tuple = tuple(sorted(st.session_state.filters.items()))
    
    # Query data using cached function
    results = get_filtered_data(session, filters_tuple)
    
    # Convert to DataFrame
    if results:
        # Convert SQLAlchemy ResultProxy to list of dicts
        data = []
        for row in results:
            row_dict = {key: getattr(row, key) for key in row._fields}
            data.append(row_dict)
        
        df = pd.DataFrame(data)
        
        # Store filtered data in session state
        st.session_state.table_data = df
        
        # Main table with key columns
        col1, col2 = st.columns([6, 1])
        with col1:
            st.subheader(f"Results: {len(df)} records found")
        with col2:
            # Add button to close all expanded rows
            if st.button("Close All Expanded", key="close_all"):
                st.session_state.expanded_rows = set()
                st.experimental_rerun()
        
        # Display the table with key columns only
        display_df = df[['date', 'company_name', 'podcast', 'sentiment', 'gt_sentiment', 'comments', 'episode']].copy()
        
        # Format date column
        if 'date' in display_df.columns:
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        # Add an explicit expander column
        display_df['expand'] = False
        
        # For each previously expanded row, set expand to True
        for i, row in display_df.iterrows():
            if i in st.session_state.expanded_rows:
                display_df.at[i, 'expand'] = True
        
        # Display the main table with editable checkboxes
        edited_df = st.data_editor(
            display_df,
            column_config={
                "expand": st.column_config.CheckboxColumn(
                    "Expand",
                    help="Select to expand this row",
                    width="small",
                ),
                "date": st.column_config.TextColumn("Date", width="small"),
                "company_name": st.column_config.TextColumn("Company", width="medium"),
                "podcast": st.column_config.TextColumn("Podcast", width="medium"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "gt_sentiment": st.column_config.TextColumn("GT Sentiment", width="small"),
                "comments": st.column_config.TextColumn("Comments", width="large"),
                "episode": st.column_config.TextColumn("Episode", width="large"),
            },
            height=400,
            key="data_editor"
        )
        
        # Track expanded rows in session state
        st.session_state.expanded_rows = set()
        for i, expand_value in enumerate(edited_df['expand']):
            if expand_value:
                st.session_state.expanded_rows.add(i)
                
        # Display expanded rows
        for idx in st.session_state.expanded_rows:
            row = df.iloc[idx]
            
            # Create expander with title
            with st.expander(f"{row['company_name']} - {row['episode']}", expanded=True):
                # Create tabs for different views
                tabs = st.tabs(["Overview", "Edit Data"])
                
                # Tab 1: Overview
                with tabs[0]:
                    # Create two columns for basic info - make transcript column wider
                    col1, col2 = st.columns([1, 3])
                    
                    # Left column - Basic info
                    with col1:
                        st.write("**ID:**", row['id'])
                        st.write("**Date:**", row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], datetime.datetime) else row['date'])
                        st.write("**Company:**", row['company_name'])
                        st.write("**Ticker:**", row['company_ticker'])
                        st.write("**Podcast:**", row['podcast'])
                        st.write("**Episode:**", row['episode'])
                        st.write("**Sentiment:**", row['sentiment'])
                        st.write("**Confidence:**", f"{row['confidence']:.2f}" if row['confidence'] else 'N/A')
                        
                        # Add snippet text below the basic info
                        if row['snippet_text']:
                            st.markdown("---")
                            st.subheader("Snippet Text")
                            st.text_area("", value=row['snippet_text'], height=150, disabled=True, key=f"snippet_{row['id']}")
                    
                    # Right column - Transcript with search
                    with col2:
                        st.subheader("Transcript")
                        
                        # Initialize session state for this row if needed
                        row_id = str(row['id'])
                        
                        # Always set the company name as the default search value when expanding
                        company_name = row['company_name'] or ""
                        
                        # Initialize or update the search term in session state
                        if f"search_term_{row_id}" not in st.session_state:
                            st.session_state[f"search_term_{row_id}"] = company_name
                        
                        # Create search input with callback to update session state
                        def update_search():
                            st.session_state[f"search_term_{row_id}"] = st.session_state[f"search_input_{row_id}"]
                        
                        # Set initial value directly from the company name
                        search_term = st.text_input(
                            "Search in transcript:", 
                            value=company_name,
                            key=f"search_input_{row_id}",
                            on_change=update_search
                        )
                        
                        # Use the session state value for searching
                        session_search_term = st.session_state[f"search_term_{row_id}"]
                        
                        # Get transcript text
                        transcript = row['transcript'] if row['transcript'] else "No transcript available"
                        
                        # Check if transcript is available
                        if transcript != "No transcript available":
                            # Add toggle for view mode
                            if f"dialog_view_{row_id}" not in st.session_state:
                                st.session_state[f"dialog_view_{row_id}"] = True
                                
                            dialog_view = st.toggle("Dialog View", value=st.session_state[f"dialog_view_{row_id}"], key=f"view_toggle_{row_id}")
                            st.session_state[f"dialog_view_{row_id}"] = dialog_view
                            
                            if dialog_view:
                                # Create a scrollable container for chat messages
                                with st.container():
                                    # Set maximum height for scrolling
                                    st.markdown("""
                                    <style>
                                        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
                                            max-height: 400px;
                                            overflow-y: auto;
                                        }
                                    </style>
                                    """, unsafe_allow_html=True)
                                    
                                    # Split transcript into segments based on timestamps
                                    # Assuming format: [00:00:00.000 --> 00:00:00.000] Text content
                                    segments = re.split(r'(\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\])', transcript)
                                    
                                    # Filter out empty segments
                                    segments = [seg.strip() for seg in segments if seg.strip()]
                                    
                                    # Group segments by timestamp and content
                                    i = 0
                                    match_count = 0
                                    first_match_index = -1
                                    
                                    # Pre-scan to count matches
                                    if session_search_term:
                                        for j in range(0, len(segments) - 1, 2):
                                            if j+1 < len(segments) and session_search_term.lower() in segments[j+1].lower():
                                                match_count += segments[j+1].lower().count(session_search_term.lower())
                                                if first_match_index == -1:
                                                    first_match_index = j // 2
                                        
                                        # Display match count
                                        if match_count > 0:
                                            st.write(f"Found {match_count} instances of '{session_search_term}'")
                                        else:
                                            st.write(f"No matches found for '{session_search_term}'")
                                    
                                    while i < len(segments) - 1:
                                        timestamp = segments[i]
                                        content = segments[i+1]
                                        
                                        # Extract just the starting time for display
                                        time_display = "unknown time"
                                        time_match = re.search(r'\[(\d{2}:\d{2}:\d{2})', timestamp)
                                        if time_match:
                                            time_display = time_match.group(1)
                                        
                                        # Determine if this segment matches the search term (for highlighting)
                                        highlight = False
                                        if session_search_term and session_search_term.lower() in content.lower():
                                            highlight = True
                                            # Apply highlighting to the content
                                            highlighted_content, _ = highlight_text(content, session_search_term)
                                            content = highlighted_content
                                        
                                        # Create chat message
                                        with st.chat_message("assistant", avatar="🎙️"):
                                            st.write(f"**{time_display}**")
                                            if highlight:
                                                st.markdown(content, unsafe_allow_html=True)
                                            else:
                                                st.write(content)
                                        
                                        i += 2
                                    
                                    # Add JavaScript to scroll to first match
                                    if first_match_index > 0 and session_search_term:
                                        st.markdown(f"""
                                        <script>
                                            // Wait for DOM to be fully loaded
                                            document.addEventListener('DOMContentLoaded', function() {{
                                                // Find all chat message containers
                                                const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
                                                // Scroll to the {first_match_index}th message if it exists
                                                if (chatMessages && chatMessages.length > {first_match_index}) {{
                                                    chatMessages[{first_match_index}].scrollIntoView({{behavior: 'auto', block: 'center'}});
                                                }}
                                            }});
                                        </script>
                                        """, unsafe_allow_html=True)
                            else:
                                # Text view - show as before
                                formatted_transcript = re.sub(r'(\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\])', r'\n\1', transcript)
                                
                                if session_search_term:
                                    # Use the highlight_text function for case-insensitive search
                                    highlighted_transcript, count = highlight_text(formatted_transcript, session_search_term)
                                    
                                    # Display match count
                                    if count > 0:
                                        st.write(f"Found {count} instances of '{session_search_term}'")
                                    else:
                                        st.write(f"No matches found for '{session_search_term}'")
                                    
                                    # Add the HTML with JavaScript to scroll to first match
                                    st.markdown(
                                        f'''<div id="transcript-container" style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: white; color: black; font-family: monospace;">
                                            <div id="transcript-content">{highlighted_transcript}</div>
                                        </div>
                                        <script>
                                            // Wait for DOM to fully load
                                            window.addEventListener('load', function() {{
                                                try {{
                                                    // Get the container and find the first highlighted term
                                                    const container = document.getElementById('transcript-container');
                                                    const firstMark = container.querySelector('mark');
                                                    // Scroll to the first match if found
                                                    if (firstMark) {{
                                                        setTimeout(function() {{
                                                            firstMark.scrollIntoView({{behavior: 'auto', block: 'center'}});
                                                        }}, 100);
                                                    }}
                                                }} catch (e) {{
                                                    console.error('Error scrolling to match:', e);
                                                }}
                                            }});
                                        </script>''', 
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.text_area("", value=formatted_transcript, height=400, disabled=True, key=f"transcript_{row_id}")
                        else:
                            st.info("No transcript available")
                
                # Tab 2: Edit Data
                with tabs[1]:
                    with st.form(f"edit_form_{row['id']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_id = row['id']
                            
                            gt_sentiment = st.selectbox(
                                "Ground Truth Sentiment",
                                sentiment_options := ["", "No sentiment", "Buy", "Hold", "Sell"],
                                index=sentiment_options.index(row['gt_sentiment']) if row['gt_sentiment'] in sentiment_options else 0
                            )
                        
                        with col2:
                            # Add metadata display if available
                            if row['meta_data']:
                                try:
                                    import json
                                    meta_data = json.loads(row['meta_data'])
                                    st.json(meta_data)
                                except:
                                    st.text(row['meta_data'])
                        
                        st.subheader("Comments")
                        comments = st.text_area("", value=row['comments'] if row['comments'] else "", height=150)
                        
                        # Option to edit transcript
                        edit_transcript = st.checkbox("Edit transcript", key=f"edit_transcript_{row['id']}")
                        
                        transcript_value = None
                        if edit_transcript:
                            st.subheader("Edit Transcript")
                            transcript_value = st.text_area(
                                "Transcript", 
                                value=row['transcript'] if row['transcript'] else "", 
                                height=300
                            )
                        
                        if st.form_submit_button("Save Changes"):
                            # Store edits in session state as backup
                            edit_data = {
                                'gt_sentiment': gt_sentiment,
                                'comments': comments,
                                'transcript': transcript_value if edit_transcript else None,
                                'timestamp': datetime.datetime.now().isoformat()
                            }
                            
                            st.session_state.sentiment_edits[sentiment_id] = edit_data
                            
                            # Save to persistent file
                            save_sentiment_edit(sentiment_id, edit_data)
                            
                            # Pass transcript value to update function if checkbox is checked
                            if update_sentiment(
                                session, 
                                sentiment_id, 
                                gt_sentiment, 
                                comments, 
                                transcript=transcript_value if edit_transcript else None
                            ):
                                st.success("Changes saved successfully!")
                                st.experimental_rerun()
                            else:
                                st.error(f"Failed to save changes to database. Your edits have been saved to backup file (ID: {sentiment_id}).")
                                
                        # Show backup status if this row has backup data
                        if sentiment_id in st.session_state.sentiment_edits:
                            backup = st.session_state.sentiment_edits[sentiment_id]
                            st.info(f"Session backup exists from {backup['timestamp']}. Sentiment: {backup['gt_sentiment']}")
                            
                        # Show persistent backup status
                        persistent_edits = st.session_state.persistent_edits
                        if str(sentiment_id) in persistent_edits and persistent_edits[str(sentiment_id)]:
                            last_edit = persistent_edits[str(sentiment_id)][-1]  # Get most recent edit
                            st.success(f"Persistent backup exists from {last_edit['timestamp']}. Sentiment: {last_edit['gt_sentiment']}")
                            if len(persistent_edits[str(sentiment_id)]) > 1:
                                st.info(f"Edit history: {len(persistent_edits[str(sentiment_id)])} versions saved")
    else:
        st.info("No records found with the current filters.")
    
    # Close session
    session.close()

if __name__ == "__main__":
    main()
