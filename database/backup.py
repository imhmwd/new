#!/usr/bin/env python3
import os
import sys
import logging
import psycopg2
import json
import time
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import subprocess
import hashlib

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/backup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Backup")

class DatabaseBackup:
    def __init__(self):
        # Database connection parameters
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        
        # Backup settings
        self.backup_dir = Path("database/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = int(os.getenv('BACKUP_RETENTION_DAYS', '7'))
        self.compression = os.getenv('BACKUP_COMPRESSION', 'gzip').lower() == 'true'
        
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
            
    def create_backup(self):
        """Create a database backup"""
        try:
            # Generate backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"backup_{timestamp}.sql"
            
            # Create backup using pg_dump
            cmd = [
                'pg_dump',
                '-h', self.db_host,
                '-p', self.db_port,
                '-U', self.db_user,
                '-d', self.db_name,
                '-F', 'c',  # Custom format
                '-f', str(backup_file)
            ]
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_password
            
            # Execute backup command
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Backup failed: {result.stderr}")
                return None
                
            # Compress backup if enabled
            if self.compression:
                compressed_file = backup_file.with_suffix('.sql.gz')
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                backup_file.unlink()  # Remove original file
                backup_file = compressed_file
                
            # Calculate checksum
            checksum = self.calculate_checksum(backup_file)
            
            # Save backup metadata
            metadata = {
                'filename': backup_file.name,
                'timestamp': timestamp,
                'size': backup_file.stat().st_size,
                'checksum': checksum,
                'compressed': self.compression
            }
            
            metadata_file = backup_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Backup created: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return None
            
    def restore_backup(self, backup_file):
        """Restore a database backup"""
        try:
            backup_file = Path(backup_file)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False
                
            # Verify checksum
            metadata_file = backup_file.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if metadata['checksum'] != self.calculate_checksum(backup_file):
                    logger.error("Backup checksum verification failed")
                    return False
                    
            # Decompress if needed
            if backup_file.suffix == '.gz':
                temp_file = backup_file.with_suffix('')
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(temp_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                backup_file = temp_file
                
            # Restore using pg_restore
            cmd = [
                'pg_restore',
                '-h', self.db_host,
                '-p', self.db_port,
                '-U', self.db_user,
                '-d', self.db_name,
                '-c',  # Clean (drop) database objects before recreating
                '-1',  # Process all commands in a single transaction
                str(backup_file)
            ]
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_password
            
            # Execute restore command
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Restore failed: {result.stderr}")
                return False
                
            # Clean up temporary file
            if backup_file.suffix != '.gz':
                backup_file.unlink()
                
            logger.info(f"Backup restored: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return False
            
    def list_backups(self):
        """List all available backups"""
        try:
            backups = []
            
            # Find all backup files
            for file in self.backup_dir.glob('*.sql*'):
                if file.suffix == '.gz':
                    backup_file = file
                else:
                    backup_file = file
                    
                metadata_file = backup_file.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    backups.append(metadata)
                    
            # Sort by timestamp
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {str(e)}")
            return None
            
    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            removed = 0
            
            # Get all backups
            backups = self.list_backups()
            if not backups:
                return 0
                
            # Remove old backups
            for backup in backups:
                backup_date = datetime.strptime(backup['timestamp'], '%Y%m%d_%H%M%S')
                if backup_date < cutoff_date:
                    backup_file = self.backup_dir / backup['filename']
                    metadata_file = backup_file.with_suffix('.json')
                    
                    if backup_file.exists():
                        backup_file.unlink()
                    if metadata_file.exists():
                        metadata_file.unlink()
                        
                    removed += 1
                    
            logger.info(f"Removed {removed} old backups")
            return removed
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return 0
            
    def calculate_checksum(self, filepath):
        """Calculate SHA-256 checksum of backup file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {str(e)}")
            return None

def main():
    """Main function to manage backups"""
    backup = DatabaseBackup()
    
    if len(sys.argv) < 2:
        logger.error("Please specify action: create, restore, list, or cleanup")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    
    if action == 'create':
        backup_file = backup.create_backup()
        if backup_file:
            sys.exit(0)
        else:
            sys.exit(1)
            
    elif action == 'restore':
        if len(sys.argv) < 3:
            logger.error("Please specify backup file to restore")
            sys.exit(1)
            
        backup_file = sys.argv[2]
        if backup.restore_backup(backup_file):
            sys.exit(0)
        else:
            sys.exit(1)
            
    elif action == 'list':
        backups = backup.list_backups()
        if backups:
            print(json.dumps(backups, indent=2))
            sys.exit(0)
        else:
            sys.exit(1)
            
    elif action == 'cleanup':
        removed = backup.cleanup_old_backups()
        if removed >= 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    else:
        logger.error("Invalid action. Use: create, restore, list, or cleanup")
        sys.exit(1)

if __name__ == "__main__":
    main() 