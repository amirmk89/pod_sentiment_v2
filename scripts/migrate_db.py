"""
Migration script to convert from single-table to multi-table database structure.
"""

import logging
import argparse
from storage import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Migrate database from single-table to multi-table structure")
    parser.add_argument("--db-path", help="Path to the database file", default=None)
    args = parser.parse_args()
    
    logger.info("Starting database migration...")
    
    # Initialize database manager
    db_manager = DatabaseManager(db_path=args.db_path)
    
    # Run migration
    db_manager.migrate_from_single_table()
    
    logger.info("Migration completed successfully")

if __name__ == "__main__":
    main() 