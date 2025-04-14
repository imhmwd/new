#!/usr/bin/env python3
import os
import sys
import logging
import psycopg2
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import hashlib
import sqlite3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/migrations.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Migrations")

class DatabaseMigrations:
    def __init__(self):
        # Database connection parameters
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        
        # Migration settings
        self.migrations_dir = Path("database/migrations")
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize migrations table
        self.init_migrations_table()
        
    def connect(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return None
            
    def init_migrations_table(self):
        """Initialize migrations tracking table"""
        try:
            conn = self.connect()
            if not conn:
                return False
                
            cur = conn.cursor()
            
            # Create migrations table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    checksum VARCHAR(64) NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'pending',
                    error TEXT
                )
            """)
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize migrations table: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
                
    def create_migration(self, name):
        """Create a new migration file"""
        try:
            # Generate migration filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{name}.sql"
            filepath = self.migrations_dir / filename
            
            # Create migration file with template
            with open(filepath, 'w') as f:
                f.write("-- Migration: " + name + "\n")
                f.write("-- Created: " + datetime.now().isoformat() + "\n\n")
                f.write("-- Up Migration\n")
                f.write("BEGIN;\n\n")
                f.write("-- Add your up migration SQL here\n\n")
                f.write("COMMIT;\n\n")
                f.write("-- Down Migration\n")
                f.write("BEGIN;\n\n")
                f.write("-- Add your down migration SQL here\n\n")
                f.write("COMMIT;\n")
                
            logger.info(f"Created migration file: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create migration: {str(e)}")
            return None
            
    def get_migration_checksum(self, filepath):
        """Calculate checksum of migration file"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {str(e)}")
            return None
            
    def parse_migration(self, filepath):
        """Parse migration file into up and down SQL"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Split into up and down migrations
            parts = content.split('-- Down Migration')
            if len(parts) != 2:
                raise ValueError("Invalid migration file format")
                
            up_sql = parts[0].split('-- Up Migration')[1].strip()
            down_sql = parts[1].strip()
            
            return {
                'up': up_sql,
                'down': down_sql
            }
            
        except Exception as e:
            logger.error(f"Failed to parse migration: {str(e)}")
            return None
            
    def apply_migration(self, filepath):
        """Apply a migration"""
        try:
            conn = self.connect()
            if not conn:
                return False
                
            cur = conn.cursor()
            
            # Get migration details
            version = filepath.stem.split('_')[0]
            name = '_'.join(filepath.stem.split('_')[1:])
            checksum = self.get_migration_checksum(filepath)
            
            if not checksum:
                return False
                
            # Check if migration already applied
            cur.execute("""
                SELECT id, status FROM schema_migrations
                WHERE version = %s AND checksum = %s
            """, (version, checksum))
            
            existing = cur.fetchone()
            if existing:
                if existing[1] == 'applied':
                    logger.info(f"Migration already applied: {filepath}")
                    return True
                elif existing[1] == 'failed':
                    logger.info(f"Retrying failed migration: {filepath}")
                    
            # Parse migration SQL
            migration = self.parse_migration(filepath)
            if not migration:
                return False
                
            # Record migration attempt
            cur.execute("""
                INSERT INTO schema_migrations (version, name, checksum, status)
                VALUES (%s, %s, %s, 'pending')
                ON CONFLICT (version, checksum) DO UPDATE
                SET status = 'pending', error = NULL
            """, (version, name, checksum))
            
            try:
                # Apply up migration
                cur.execute(migration['up'])
                conn.commit()
                
                # Update migration status
                cur.execute("""
                    UPDATE schema_migrations
                    SET status = 'applied'
                    WHERE version = %s AND checksum = %s
                """, (version, checksum))
                
                conn.commit()
                logger.info(f"Applied migration: {filepath}")
                return True
                
            except Exception as e:
                # Record failure
                cur.execute("""
                    UPDATE schema_migrations
                    SET status = 'failed', error = %s
                    WHERE version = %s AND checksum = %s
                """, (str(e), version, checksum))
                
                conn.commit()
                logger.error(f"Migration failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply migration: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
                
    def rollback_migration(self, filepath):
        """Rollback a migration"""
        try:
            conn = self.connect()
            if not conn:
                return False
                
            cur = conn.cursor()
            
            # Get migration details
            version = filepath.stem.split('_')[0]
            checksum = self.get_migration_checksum(filepath)
            
            if not checksum:
                return False
                
            # Parse migration SQL
            migration = self.parse_migration(filepath)
            if not migration:
                return False
                
            try:
                # Apply down migration
                cur.execute(migration['down'])
                conn.commit()
                
                # Update migration status
                cur.execute("""
                    UPDATE schema_migrations
                    SET status = 'rolled_back'
                    WHERE version = %s AND checksum = %s
                """, (version, checksum))
                
                conn.commit()
                logger.info(f"Rolled back migration: {filepath}")
                return True
                
            except Exception as e:
                logger.error(f"Rollback failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to rollback migration: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
                
    def get_migration_status(self):
        """Get status of all migrations"""
        try:
            conn = self.connect()
            if not conn:
                return None
                
            cur = conn.cursor()
            
            # Get all migrations
            cur.execute("""
                SELECT version, name, checksum, applied_at, status, error
                FROM schema_migrations
                ORDER BY version DESC
            """)
            
            migrations = []
            for row in cur.fetchall():
                migrations.append({
                    'version': row[0],
                    'name': row[1],
                    'checksum': row[2],
                    'applied_at': row[3].isoformat() if row[3] else None,
                    'status': row[4],
                    'error': row[5]
                })
                
            return migrations
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
                
    def migrate(self, direction='up'):
        """Run all pending migrations"""
        try:
            # Get all migration files
            migration_files = sorted(
                self.migrations_dir.glob('*.sql'),
                key=lambda x: x.stem.split('_')[0]
            )
            
            if not migration_files:
                logger.info("No migrations found")
                return True
                
            # Get applied migrations
            status = self.get_migration_status()
            applied = {m['version']: m for m in status} if status else {}
            
            if direction == 'up':
                # Apply pending migrations
                for filepath in migration_files:
                    version = filepath.stem.split('_')[0]
                    if version not in applied or applied[version]['status'] != 'applied':
                        if not self.apply_migration(filepath):
                            return False
            else:
                # Rollback migrations in reverse order
                for filepath in reversed(migration_files):
                    version = filepath.stem.split('_')[0]
                    if version in applied and applied[version]['status'] == 'applied':
                        if not self.rollback_migration(filepath):
                            return False
                            
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return False

def main():
    """Main function to manage migrations"""
    migrations = DatabaseMigrations()
    
    if len(sys.argv) < 2:
        logger.error("Please specify action: create, migrate, rollback, or status")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    
    if action == 'create':
        if len(sys.argv) < 3:
            logger.error("Please specify migration name")
            sys.exit(1)
            
        name = sys.argv[2]
        filepath = migrations.create_migration(name)
        if filepath:
            sys.exit(0)
        else:
            sys.exit(1)
            
    elif action == 'migrate':
        direction = sys.argv[2] if len(sys.argv) > 2 else 'up'
        if migrations.migrate(direction):
            sys.exit(0)
        else:
            sys.exit(1)
            
    elif action == 'rollback':
        if len(sys.argv) < 3:
            logger.error("Please specify migration file to rollback")
            sys.exit(1)
            
        filepath = Path(sys.argv[2])
        if migrations.rollback_migration(filepath):
            sys.exit(0)
        else:
            sys.exit(1)
            
    elif action == 'status':
        status = migrations.get_migration_status()
        if status:
            print(json.dumps(status, indent=2))
            sys.exit(0)
        else:
            sys.exit(1)
            
    else:
        logger.error("Invalid action. Use: create, migrate, rollback, or status")
        sys.exit(1)

if __name__ == "__main__":
    main() 