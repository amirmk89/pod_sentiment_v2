import os
import sys
import streamlit as st
import pandas as pd
import datetime
from sqlalchemy import create_engine, func, or_, and_
from sqlalchemy.orm import sessionmaker

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

# Query data with filters
def get_filtered_data(session, filters):
    """Get sentiment data with filters"""
    query = session.query(
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
            Episode.title.like(search_term),
            EntitySentiment.company_name.like(search_term),
            EntitySentiment.snippet_text.like(search_term)
        ))
    
    # Order by date descending
    query = query.order_by(Episode.date.desc())
    
    return query.all()

# Get filter options
def get_filter_options(session):
    """Get distinct values for filters"""
    podcasts = [row[0] for row in session.query(Podcast.name).distinct().order_by(Podcast.name).all()]
    companies = [row[0] for row in session.query(EntitySentiment.company_name).distinct().order_by(EntitySentiment.company_name).all()]
    sentiments = [row[0] for row in session.query(EntitySentiment.sentiment).distinct().filter(EntitySentiment.sentiment != None).filter(EntitySentiment.sentiment != '').order_by(EntitySentiment.sentiment).all()]
    
    return {
        'podcasts': podcasts,
        'companies': companies,
        'sentiments': sentiments
    }

# Update sentiment record
def update_sentiment(session, sentiment_id, gt_sentiment, comments, transcript=None):
    """Update gt_sentiment, comments, and transcript fields for a sentiment record"""
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

# Highlight text in a string (case-insensitive)
def highlight_text(text, search_term):
    """Highlight search term in text (case-insensitive)"""
    if not search_term or not text:
        return text
    
    import re
    pattern = re.compile(f'({re.escape(search_term)})', re.IGNORECASE)
    return pattern.sub(r'<span style="background-color: yellow;">\1</span>', text)

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
    
    # Text search
    search = st.sidebar.text_input("Search (title, company, or transcript)")
    
    # Apply filters button
    if st.sidebar.button("Apply Filters"):
        st.experimental_rerun()
    
    # Reset filters button
    if st.sidebar.button("Reset Filters"):
        # This will be caught on the next rerun and reset all filters
        st.experimental_set_query_params()
        st.experimental_rerun()
    
    # Build filters dictionary
    filters = {
        'date_range': date_range,
        'company': company if company else None,
        'podcast': podcast if podcast else None,
        'sentiment': sentiment if sentiment else None,
        'search': search if search else None
    }
    
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    # Query data
    results = get_filtered_data(session, filters)
    
    # Convert to DataFrame
    if results:
        # Convert SQLAlchemy ResultProxy to list of dicts
        data = []
        for row in results:
            row_dict = {key: getattr(row, key) for key in row._fields}
            data.append(row_dict)
        
        df = pd.DataFrame(data)
        
        # Main table with key columns
        st.subheader(f"Results: {len(df)} records found")
        
        # Display the table with key columns only
        display_df = df[['date', 'company_name', 'podcast', 'sentiment', 'episode']].copy()
        
        # Format date column
        if 'date' in display_df.columns:
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        # Add an explicit expander column
        display_df['expand'] = False
        
        # Display the main table with editable checkboxes
        edited_df = st.data_editor(
            display_df,
            column_config={
                "expand": st.column_config.CheckboxColumn(
                    "Expand",
                    help="Select to expand this row",
                    width="small",
                ),
                "date": st.column_config.TextColumn("Date"),
                "company_name": st.column_config.TextColumn("Company"),
                "podcast": st.column_config.TextColumn("Podcast"),
                "sentiment": st.column_config.TextColumn("Sentiment"),
                "episode": st.column_config.TextColumn("Episode"),
            },
            height=400,
            key="data_editor"
        )
        
        # Get indices of rows to expand
        expanded_indices = []
        for i, expand_value in enumerate(edited_df['expand']):
            if expand_value:
                expanded_indices.append(i)
                
        # Display expanded rows
        for idx in expanded_indices:
            row = df.iloc[idx]
            
            # Create expander with title
            with st.expander(f"{row['company_name']} - {row['episode']}", expanded=True):
                # Create tabs for different views
                tabs = st.tabs(["Overview", "Edit Data"])
                
                # Tab 1: Overview
                with tabs[0]:
                    # Create two columns for basic info
                    col1, col2 = st.columns([1, 2])
                    
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
                        
                        # Display transcript with search functionality
                        # Create search input that triggers updates automatically
                        search_term = st.text_input(
                            "Search in transcript:", 
                            key=f"search_input_{row['id']}"
                        )
                        
                        # Display transcript with highlighting
                        transcript = row['transcript'] if row['transcript'] else "No transcript available"
                        
                        if search_term and transcript != "No transcript available":
                            # Use the highlight_text function for case-insensitive search
                            highlighted_transcript = highlight_text(transcript, search_term)
                            st.markdown(
                                f'<div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: white;">{highlighted_transcript}</div>', 
                                unsafe_allow_html=True
                            )
                        else:
                            st.text_area("", value=transcript, height=400, disabled=True, key=f"transcript_{row['id']}")
                
                # Tab 2: Edit Data
                with tabs[1]:
                    with st.form(f"edit_form_{row['id']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            gt_sentiment = st.selectbox(
                                "Ground Truth Sentiment",
                                ["", "Buy", "Hold", "Sell"],
                                index=["", "Buy", "Hold", "Sell"].index(row['gt_sentiment']) if row['gt_sentiment'] in ["Buy", "Hold", "Sell"] else 0
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
                            # Pass transcript value to update function if checkbox is checked
                            if update_sentiment(
                                session, 
                                row['id'], 
                                gt_sentiment, 
                                comments, 
                                transcript=transcript_value if edit_transcript else None
                            ):
                                st.success("Changes saved successfully!")
                                st.experimental_rerun()
                            else:
                                st.error("Failed to save changes")
    else:
        st.info("No records found with the current filters.")
    
    # Close session
    session.close()

if __name__ == "__main__":
    main()
