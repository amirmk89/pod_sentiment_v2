import sqlite3
import os
from podscan_client.storage import DatabaseManager

def diagnose_database():
    """Check database connection and table data"""
    # Initialize the database manager to get the path
    db_manager = DatabaseManager()
    db_path = db_manager.db_path
    
    print(f"Database path: {db_path}")
    print(f"Database exists: {os.path.exists(db_path)}")
    
    # Connect to the database
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        print("Successfully connected to database")
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"\nTables in database: {[table['name'] for table in tables]}")
        
        # Check row counts
        for table in tables:
            table_name = table['name']
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            row_count = cursor.fetchone()['count']
            print(f"Table {table_name}: {row_count} rows")
        
        # Check sample entity_sentiments data
        try:
            cursor.execute("SELECT * FROM entity_sentiments LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                print("\nSample entity_sentiment row:")
                print(f"ID: {sample['id']}")
                print(f"Company: {sample['company_name']} ({sample['company_ticker']})")
                print(f"Sentiment: {sample['sentiment']}")
                print(f"Episode ID: {sample['episode_id']}")
            else:
                print("\nNo rows in entity_sentiments table")
        except Exception as e:
            print(f"Error checking entity_sentiments: {e}")
        
        # Try a join query
        try:
            query = """
            SELECT 
                es.id,
                p.name as podcast,
                e.title as episode,
                es.company_name,
                es.company_ticker,
                es.sentiment
            FROM entity_sentiments es
            JOIN episodes e ON es.episode_id = e.id
            JOIN podcasts p ON e.podcast_id = p.id
            LIMIT 1
            """
            cursor.execute(query)
            row = cursor.fetchone()
            if row:
                print("\nJoin query successful, sample result:")
                print(f"ID: {row['id']}")
                print(f"Podcast: {row['podcast']}")
                print(f"Episode: {row['episode']}")
                print(f"Company: {row['company_name']} ({row['company_ticker']})")
                print(f"Sentiment: {row['sentiment']}")
            else:
                print("\nJoin query returned no results (tables may be empty or not properly related)")
        except Exception as e:
            print(f"\nError running join query: {e}")
            print("This might indicate missing foreign keys or relationship issues")
        
        conn.close()
        
    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    diagnose_database() 