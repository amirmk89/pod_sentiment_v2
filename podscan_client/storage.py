"""
Storage module for managing database operations.
"""

import os
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    func,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from podscan_client.config import DB_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Create SQLAlchemy base
Base = declarative_base()


class Podcast(Base):
    """SQLAlchemy model for podcasts."""

    __tablename__ = "podcasts"

    id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    episodes = relationship("Episode", back_populates="podcast", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Podcast(id='{self.id}', name='{self.name}')>"


class Episode(Base):
    """SQLAlchemy model for podcast episodes."""

    __tablename__ = "episodes"

    id = Column(String(100), primary_key=True)
    podcast_id = Column(String(100), ForeignKey("podcasts.id"), nullable=False, index=True)
    title = Column(String(255))
    description = Column(Text)
    transcript = Column(Text)  # Full episode transcript
    date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    podcast = relationship("Podcast", back_populates="episodes")
    sentiments = relationship("EntitySentiment", back_populates="episode", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Episode(id='{self.id}', title='{self.title}')>"


class EntitySentiment(Base):
    """SQLAlchemy model for entity sentiment analysis."""

    __tablename__ = "entity_sentiments"

    id = Column(Integer, primary_key=True)
    company_name = Column(String(100), nullable=False, index=True)
    company_ticker = Column(String(10), nullable=False, index=True)
    episode_id = Column(String(100), ForeignKey("episodes.id"), nullable=False, index=True)
    snippet_text = Column(Text, nullable=True)  # Optional for episode-level sentiment
    sentiment = Column(String(10), nullable=False, index=True)  # Buy, Hold, Sell
    gt_sentiment = Column(String(10), nullable=True, index=True)  # Ground truth sentiment
    confidence = Column(Float)
    is_episode_level = Column(Boolean, default=False)  # Flag for episode vs snippet level
    is_mock = Column(Boolean, default=False)  # Flag for mock/test data
    processed_at = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(Text)  # JSON string for additional metadata
    comments = Column(Text, nullable=True)  # Additional comments
    
    # Relationships
    episode = relationship("Episode", back_populates="sentiments")

    def __repr__(self):
        return f"<EntitySentiment(company='{self.company_name}', sentiment='{self.sentiment}', is_episode_level={self.is_episode_level})>"


class DatabaseManager:
    """Manager for database operations."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or DB_CONFIG["db_path"]
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        logger.info(f"Database initialized at: {self.db_path}")

    def get_or_create_podcast(self, podcast_id: str, podcast_name: str) -> Podcast:
        """Get or create a podcast record.

        Args:
            podcast_id: Unique identifier for the podcast
            podcast_name: Name of the podcast

        Returns:
            Podcast object
        """
        session = self.Session()
        
        try:
            podcast = session.query(Podcast).filter(Podcast.id == podcast_id).first()
            
            if not podcast:
                podcast = Podcast(id=podcast_id, name=podcast_name)
                session.add(podcast)
                session.commit()
                logger.info(f"Created new podcast: {podcast_name}")
            
            return podcast
        
        finally:
            session.close()

    def get_or_create_episode(
        self, 
        episode_id: str, 
        podcast_id: str, 
        title: str = None, 
        description: str = None,
        transcript: str = None,
        date: datetime = None
    ) -> Episode:
        """Get or create an episode record.

        Args:
            episode_id: Unique identifier for the episode
            podcast_id: ID of the podcast this episode belongs to
            title: Episode title
            description: Episode description
            transcript: Episode transcript
            date: Episode publication date

        Returns:
            Episode object
        """
        session = self.Session()
        
        try:
            episode = session.query(Episode).filter(Episode.id == episode_id).first()
            
            if not episode:
                episode = Episode(
                    id=episode_id,
                    podcast_id=podcast_id,
                    title=title,
                    description=description,
                    transcript=transcript,
                    date=date
                )
                session.add(episode)
                session.commit()
                logger.info(f"Created new episode: {title}")
            elif transcript and not episode.transcript:
                # Update transcript if it's provided and not already set
                episode.transcript = transcript
                session.commit()
                logger.info(f"Updated transcript for episode: {title}")
            
            return episode
        
        finally:
            session.close()

    def save_sentiment(self, sentiment_data: Dict[str, Any], is_episode_level: bool = False) -> int:
        """Save a sentiment result to the database.

        Args:
            sentiment_data: Dictionary containing sentiment data
            is_episode_level: Whether this is an episode-level sentiment

        Returns:
            ID of the inserted record
        """
        session = self.Session()

        try:
            # Extract podcast and episode data
            podcast_id = sentiment_data.pop("podcast_id", None)
            podcast_name = sentiment_data.pop("podcast_name", None)
            episode_transcript = sentiment_data.pop("episode_transcript", None)
            if "podcast_transcript" in sentiment_data:
                # For backward compatibility
                if not episode_transcript:
                    episode_transcript = sentiment_data.pop("podcast_transcript", None)
                else:
                    sentiment_data.pop("podcast_transcript", None)
            episode_id = sentiment_data.pop("episode_id", None)
            episode_title = sentiment_data.pop("episode_title", None)
            episode_description = sentiment_data.pop("episode_description", None)
            episode_date = sentiment_data.pop("episode_date", None)
            
            # Ensure podcast exists
            self.get_or_create_podcast(podcast_id, podcast_name)
            
            # Ensure episode exists
            self.get_or_create_episode(
                episode_id, 
                podcast_id, 
                episode_title, 
                episode_description,
                episode_transcript,  # Pass transcript to episode
                episode_date
            )
            
            # Prepare sentiment data
            sentiment_data["episode_id"] = episode_id
            sentiment_data["is_episode_level"] = is_episode_level
            
            # Convert metadata to JSON string if it exists
            if "metadata" in sentiment_data and isinstance(sentiment_data["metadata"], dict):
                sentiment_data["meta_data"] = json.dumps(sentiment_data["metadata"])
                del sentiment_data["metadata"]

            # Create a new sentiment object
            sentiment = EntitySentiment(**sentiment_data)

            # Add and commit
            session.add(sentiment)
            session.commit()

            logger.info(
                f"Saved {'episode-level' if is_episode_level else 'snippet-level'} "
                f"sentiment for {sentiment_data.get('company_name')}: {sentiment_data.get('sentiment')}"
            )

            return sentiment.id

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving sentiment to database: {e}")
            raise

        finally:
            session.close()

    def save_batch_sentiments(
        self, sentiments: List[Dict[str, Any]], is_episode_level: bool = False
    ) -> List[int]:
        """Save multiple sentiment results in batch.

        Args:
            sentiments: List of sentiment dictionaries
            is_episode_level: Whether these are episode-level sentiments

        Returns:
            List of inserted record IDs
        """
        session = self.Session()
        sentiment_ids = []
        
        # Track unique podcasts and episodes to create first
        podcasts = {}
        episodes = {}

        try:
            # First pass: extract podcast and episode data
            for data in sentiments:
                podcast_id = data.get("podcast_id")
                podcast_name = data.get("podcast_name")
                # Handle both field names for backward compatibility
                episode_transcript = data.get("episode_transcript")
                if not episode_transcript and "podcast_transcript" in data:
                    episode_transcript = data.get("podcast_transcript")
                episode_id = data.get("episode_id")
                
                if podcast_id and podcast_name:
                    podcasts[podcast_id] = {
                        "name": podcast_name
                    }
                
                if episode_id and podcast_id:
                    episodes[episode_id] = {
                        "podcast_id": podcast_id,
                        "title": data.get("episode_title"),
                        "description": data.get("episode_description"),
                        "transcript": episode_transcript,  # Add transcript to episode data
                        "date": data.get("episode_date")
                    }
            
            # Create podcasts
            for podcast_id, podcast_data in podcasts.items():
                podcast = session.query(Podcast).filter(Podcast.id == podcast_id).first()
                if not podcast:
                    podcast = Podcast(
                        id=podcast_id, 
                        name=podcast_data["name"]
                    )
                    session.add(podcast)
            
            # Flush to ensure podcasts are created
            session.flush()
            
            # Create episodes
            for episode_id, episode_data in episodes.items():
                episode = session.query(Episode).filter(Episode.id == episode_id).first()
                if not episode:
                    episode = Episode(
                        id=episode_id,
                        podcast_id=episode_data["podcast_id"],
                        title=episode_data["title"],
                        description=episode_data["description"],
                        transcript=episode_data["transcript"],  # Add transcript
                        date=episode_data["date"]
                    )
                    session.add(episode)
                elif episode_data["transcript"] and not episode.transcript:
                    # Update transcript if provided and not already set
                    episode.transcript = episode_data["transcript"]
            
            # Flush to ensure episodes are created
            session.flush()
            
            # Second pass: create sentiment records
            for data in sentiments:
                # Remove podcast and episode fields
                sentiment_data = {k: v for k, v in data.items() if k not in [
                    "podcast_name", "podcast_id", "podcast_title", "podcast_transcript", "episode_title", 
                    "episode_description", "episode_date", "episode_transcript"
                ]}
                
                # Set episode-level flag
                sentiment_data["is_episode_level"] = is_episode_level
                
                # Convert metadata to JSON string if it exists
                if "metadata" in sentiment_data and isinstance(sentiment_data["metadata"], dict):
                    sentiment_data["meta_data"] = json.dumps(sentiment_data["metadata"])
                    del sentiment_data["metadata"]
                
                # Create sentiment
                sentiment = EntitySentiment(**sentiment_data)
                session.add(sentiment)
                
                # Flush to get ID
                session.flush()
                sentiment_ids.append(sentiment.id)
            
            # Commit all changes
            session.commit()
            
            logger.info(f"Saved batch of {len(sentiments)} sentiment results to database")
            
            return sentiment_ids
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving batch sentiments to database: {e}")
            raise
            
        finally:
            session.close()

    def get_sentiments_by_company(
        self, 
        company_name: str = None, 
        company_ticker: str = None,
        is_episode_level: Optional[bool] = None
    ) -> List[Dict]:
        """Get sentiment results for a specific company.

        Args:
            company_name: Company name to filter by
            company_ticker: Company ticker to filter by
            is_episode_level: Filter by episode-level flag (None for all)

        Returns:
            List of sentiment dictionaries with podcast and episode data
        """
        session = self.Session()

        try:
            query = session.query(
                EntitySentiment, 
                Episode.title.label("episode_title"),
                Episode.description.label("episode_description"),
                Episode.date.label("episode_date"),
                Podcast.name.label("podcast_name")
            ).join(
                Episode, EntitySentiment.episode_id == Episode.id
            ).join(
                Podcast, Episode.podcast_id == Podcast.id
            )

            if company_name:
                query = query.filter(EntitySentiment.company_name == company_name)

            if company_ticker:
                query = query.filter(EntitySentiment.company_ticker == company_ticker)
                
            if is_episode_level is not None:
                query = query.filter(EntitySentiment.is_episode_level == is_episode_level)

            results = query.all()

            # Convert to dictionaries
            result_dicts = []
            for row in results:
                sentiment = row[0]
                
                # Create base dictionary from sentiment object
                result_dict = {
                    c.name: getattr(sentiment, c.name) for c in sentiment.__table__.columns
                }
                
                # Add episode and podcast data
                result_dict["episode_title"] = row.episode_title
                result_dict["episode_description"] = row.episode_description
                result_dict["episode_date"] = row.episode_date
                result_dict["podcast_name"] = row.podcast_name

                # Parse metadata JSON if it exists
                if result_dict.get("meta_data"):
                    try:
                        result_dict["metadata"] = json.loads(result_dict["meta_data"])
                        del result_dict["meta_data"]
                    except json.JSONDecodeError:
                        pass

                result_dicts.append(result_dict)

            return result_dicts

        finally:
            session.close()

    def check_duplicate_sentiment(
        self, company_name: str, episode_id: str, snippet_text: str = None, is_episode_level: bool = False
    ) -> bool:
        """Check if a similar sentiment already exists in the database.

        Args:
            company_name: Company name
            episode_id: Episode ID
            snippet_text: Snippet text (None for episode-level sentiments)
            is_episode_level: Whether this is an episode-level sentiment

        Returns:
            True if a duplicate exists, False otherwise
        """
        session = self.Session()

        try:
            query = session.query(EntitySentiment).filter(
                EntitySentiment.company_name == company_name,
                EntitySentiment.episode_id == episode_id,
                EntitySentiment.is_episode_level == is_episode_level
            )
            
            if snippet_text and not is_episode_level:
                query = query.filter(EntitySentiment.snippet_text == snippet_text)
            elif is_episode_level:
                # For episode-level, we don't need to check snippet text
                pass
                
            count = query.count()
            return count > 0

        finally:
            session.close()

    def delete_episode_level_sentiment(self, company_name: str, episode_id: str) -> bool:
        """Delete existing episode-level sentiment for a company and episode.
        
        Args:
            company_name: Company name
            episode_id: Episode ID
            
        Returns:
            True if a record was deleted, False otherwise
        """
        session = self.Session()
        
        try:
            query = session.query(EntitySentiment).filter(
                EntitySentiment.company_name == company_name,
                EntitySentiment.episode_id == episode_id,
                EntitySentiment.is_episode_level == True
            )
            
            count = query.count()
            if count > 0:
                query.delete()
                session.commit()
                logger.info(f"Deleted existing episode-level sentiment for {company_name} in episode {episode_id}")
                return True
            return False
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting episode-level sentiment: {e}")
            return False
            
        finally:
            session.close()

    def export_to_csv(
        self,
        output_path: str,
        company_name: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        is_episode_level: Optional[bool] = None
    ) -> str:
        """Export results to a CSV file.

        Args:
            output_path: Path to save the CSV file
            company_name: Filter by company name
            start_date: Filter by start date
            end_date: Filter by end date
            is_episode_level: Filter by episode-level flag (None for all)

        Returns:
            Path to the saved CSV file
        """
        session = self.Session()

        try:
            query = session.query(
                EntitySentiment,
                Episode.title.label("episode_title"),
                Episode.description.label("episode_description"),
                Episode.date.label("episode_date"),
                Podcast.name.label("podcast_name"),
                Podcast.id.label("podcast_id")
            ).join(
                Episode, EntitySentiment.episode_id == Episode.id
            ).join(
                Podcast, Episode.podcast_id == Podcast.id
            )

            if company_name:
                query = query.filter(EntitySentiment.company_name == company_name)

            if start_date:
                query = query.filter(Episode.date >= start_date)

            if end_date:
                query = query.filter(Episode.date <= end_date)
                
            if is_episode_level is not None:
                query = query.filter(EntitySentiment.is_episode_level == is_episode_level)

            results = query.all()

            # Convert to DataFrame
            result_dicts = []
            for row in results:
                sentiment = row[0]
                
                # Create base dictionary from sentiment object
                result_dict = {
                    c.name: getattr(sentiment, c.name) for c in sentiment.__table__.columns
                }
                
                # Add episode and podcast data
                result_dict["episode_title"] = row.episode_title
                result_dict["episode_description"] = row.episode_description
                result_dict["episode_date"] = row.episode_date
                result_dict["podcast_name"] = row.podcast_name
                result_dict["podcast_id"] = row.podcast_id
                
                # Parse metadata
                if result_dict.get("meta_data"):
                    try:
                        result_dict["metadata"] = json.loads(result_dict["meta_data"])
                        del result_dict["meta_data"]
                    except json.JSONDecodeError:
                        pass
                
                result_dicts.append(result_dict)

            df = pd.DataFrame(result_dicts)

            # Save to CSV
            df.to_csv(output_path, index=False)

            logger.info(f"Exported {len(result_dicts)} results to {output_path}")

            return output_path

        finally:
            session.close()

    def get_sentiment_summary(
        self, 
        company_name: str = None, 
        company_ticker: str = None,
        is_episode_level: Optional[bool] = None
    ) -> Dict:
        """Get a summary of sentiment results.

        Args:
            company_name: Filter by company name
            company_ticker: Filter by company ticker
            is_episode_level: Filter by episode-level flag (None for all)

        Returns:
            Dictionary with sentiment summary
        """
        session = self.Session()

        try:
            query = session.query(
                EntitySentiment.sentiment, 
                func.count(EntitySentiment.id).label("count")
            )

            if company_name:
                query = query.filter(EntitySentiment.company_name == company_name)

            if company_ticker:
                query = query.filter(EntitySentiment.company_ticker == company_ticker)
                
            if is_episode_level is not None:
                query = query.filter(EntitySentiment.is_episode_level == is_episode_level)

            query = query.group_by(EntitySentiment.sentiment)

            results = query.all()

            # Convert to dictionary
            summary = {"Buy": 0, "Hold": 0, "Sell": 0, "total": 0}

            for sentiment, count in results:
                summary[sentiment] = count
                summary["total"] += count

            return summary

        finally:
            session.close()

    def get_all_sentiments_count(self, is_episode_level: Optional[bool] = None) -> int:
        """Get the total count of all sentiments in the database.

        Args:
            is_episode_level: Filter by episode-level flag (None for all)

        Returns:
            Total number of sentiments
        """
        session = self.Session()

        try:
            query = session.query(func.count(EntitySentiment.id))
            
            if is_episode_level is not None:
                query = query.filter(EntitySentiment.is_episode_level == is_episode_level)
                
            count = query.scalar()
            return count or 0
        finally:
            session.close()

    def get_company_sentiments_count(self, company_name: str, is_episode_level: Optional[bool] = None) -> int:
        """Get the count of sentiments for a specific company.

        Args:
            company_name: Company name to filter by
            is_episode_level: Filter by episode-level flag (None for all)

        Returns:
            Number of sentiments for the company
        """
        session = self.Session()

        try:
            query = session.query(func.count(EntitySentiment.id)).filter(
                EntitySentiment.company_name == company_name
            )
            
            if is_episode_level is not None:
                query = query.filter(EntitySentiment.is_episode_level == is_episode_level)
                
            count = query.scalar()
            return count or 0
        finally:
            session.close()

    # Migration helper method
    def migrate_from_single_table(self) -> None:
        """Migrate data from the old single-table structure to the new multi-table structure.
        
        This assumes the old table 'sentiment_results' still exists.
        """
        try:
            # Create a direct connection to access the old table
            conn = sqlite3.connect(self.db_path)
            old_data = pd.read_sql("SELECT * FROM sentiment_results", conn)
            conn.close()
            
            if old_data.empty:
                logger.info("No data to migrate from old table structure")
                return
                
            # Process in batches to avoid memory issues
            batch_size = 100
            total_records = len(old_data)
            
            for i in range(0, total_records, batch_size):
                batch = old_data.iloc[i:i+batch_size]
                
                # Convert to dictionaries
                records = batch.to_dict('records')
                
                # Process each record
                for record in records:
                    # Extract podcast data
                    podcast_data = {
                        "id": record["podcast_id"],
                        "name": record["podcast_name"]
                    }
                    
                    # Extract episode data
                    episode_data = {
                        "id": record["episode_id"],
                        "podcast_id": record["podcast_id"],
                        "title": record["episode_title"],
                        "description": record["episode_description"],
                        "date": record["episode_date"]
                    }
                    
                    # Handle transcript field
                    if "episode_transcript" in record:
                        episode_data["transcript"] = record["episode_transcript"]
                    elif "podcast_transcript" in record:
                        episode_data["transcript"] = record["podcast_transcript"]
                    
                    # Extract sentiment data
                    sentiment_data = {
                        "company_name": record["company_name"],
                        "company_ticker": record["company_ticker"],
                        "episode_id": record["episode_id"],
                        "snippet_text": record["snippet_text"],
                        "sentiment": record["sentiment"],
                        "confidence": record["confidence"],
                        "processed_at": record["processed_at"],
                        "is_episode_level": False  # All old records are snippet-level
                    }
                    
                    # Handle metadata
                    if "snippet_metadata" in record and record["snippet_metadata"]:
                        sentiment_data["meta_data"] = record["snippet_metadata"]
                    
                    # Save to new structure
                    self.get_or_create_podcast(podcast_data["id"], podcast_data["name"])
                    self.get_or_create_episode(
                        episode_data["id"],
                        episode_data["podcast_id"],
                        episode_data["title"],
                        episode_data["description"],
                        episode_data["transcript"],  # Pass transcript
                        episode_data["date"]
                    )
                    self.save_sentiment(sentiment_data)
                
                logger.info(f"Migrated batch {i//batch_size + 1} ({min(i+batch_size, total_records)}/{total_records} records)")
                
            logger.info(f"Migration complete: {total_records} records migrated to new structure")
            
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            raise

    def get_all_processed_episodes(self) -> List[Dict]:
        """Get all episodes that have been processed and have transcripts.

        Returns:
            List of episode dictionaries with transcript data
        """
        session = self.Session()

        try:
            query = session.query(
                Episode.id,
                Episode.podcast_id,
                Episode.title,
                Episode.description,
                Episode.transcript,
                Episode.date,
                Podcast.name.label("podcast_name")
            ).join(
                Podcast, Episode.podcast_id == Podcast.id
            ).filter(
                Episode.transcript.isnot(None)
            )

            results = query.all()

            # Convert to dictionaries
            episode_dicts = []
            for row in results:
                episode_dict = {
                    "id": row.id,
                    "podcast_id": row.podcast_id,
                    "title": row.title,
                    "description": row.description,
                    "transcript": row.transcript,
                    "date": row.date,
                    "podcast_name": row.podcast_name
                }
                episode_dicts.append(episode_dict)

            return episode_dicts

        finally:
            session.close()

    def delete_snippet_level_sentiments(self, company_name: str = None, company_ticker: str = None) -> int:
        """Delete all snippet-level sentiment records from the database.
        
        Args:
            company_name: Optional company name to filter by
            company_ticker: Optional company ticker to filter by
            
        Returns:
            Number of records deleted
        """
        session = self.Session()
        
        try:
            query = session.query(EntitySentiment).filter(
                EntitySentiment.is_episode_level == False
            )
            
            # Apply optional filters
            if company_name:
                query = query.filter(EntitySentiment.company_name == company_name)
            
            if company_ticker:
                query = query.filter(EntitySentiment.company_ticker == company_ticker)
            
            count = query.count()
            if count > 0:
                query.delete()
                session.commit()
                logger.info(f"Deleted {count} snippet-level sentiments")
            return count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting snippet-level sentiments: {e}")
            raise
            
        finally:
            session.close()

    def delete_sentiments(
        self, 
        company_name: Optional[str | List[str]] = None, 
        company_ticker: Optional[str | List[str]] = None,
        is_episode_level: Optional[bool] = None
    ) -> int:
        """Delete sentiment records from the database based on filters.
        
        Args:
            company_name: Company name(s) to filter by (string or list of strings)
            company_ticker: Company ticker(s) to filter by (string or list of strings)
            is_episode_level: Optional flag to filter by episode vs snippet level
                             (None deletes both types)
            
        Returns:
            Number of records deleted
        """
        session = self.Session()
        
        try:
            query = session.query(EntitySentiment)
            
            # Convert string inputs to lists for uniform handling
            if company_name and isinstance(company_name, str):
                company_name = [company_name]
            
            if company_ticker and isinstance(company_ticker, str):
                company_ticker = [company_ticker]
            
            # Apply company name filter
            if company_name:
                query = query.filter(EntitySentiment.company_name.in_(company_name))
            
            # Apply company ticker filter
            if company_ticker:
                query = query.filter(EntitySentiment.company_ticker.in_(company_ticker))
            
            # Apply episode-level filter
            if is_episode_level is not None:
                query = query.filter(EntitySentiment.is_episode_level == is_episode_level)
            
            # Require at least one filter for safety
            if not company_name and not company_ticker and is_episode_level is None:
                logger.warning("Attempted to delete sentiments without any filters")
                return 0
            
            count = query.count()
            if count > 0:
                query.delete()
                session.commit()
                
                # Log which filters were used
                filters_used = []
                if company_name:
                    filters_used.append(f"company_name={len(company_name)} companies")
                if company_ticker:
                    filters_used.append(f"company_ticker={len(company_ticker)} tickers")
                if is_episode_level is not None:
                    filters_used.append(f"is_episode_level={is_episode_level}")
                    
                logger.info(f"Deleted {count} sentiment records with filters: {', '.join(filters_used)}")
            return count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting sentiment records: {e}")
            raise
            
        finally:
            session.close()


if __name__ == "__main__":
    db_manager = DatabaseManager()
    # db_manager.delete_snippet_level_sentiments()
    # db_manager.delete_sentiments(company_ticker=["UA", "UL", "DEO"])