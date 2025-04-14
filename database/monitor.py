#!/usr/bin/env python3
import os
import sys
import logging
import psycopg2
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tabulate import tabulate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/monitor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Monitor")

class DatabaseMonitor:
    def __init__(self):
        # Database connection parameters
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        
        # Monitoring settings
        self.check_interval = int(os.getenv('MONITOR_INTERVAL', '60'))
        self.alert_threshold = float(os.getenv('ALERT_THRESHOLD', '0.8'))
        
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
            
    def get_connection_stats(self):
        """Get database connection statistics"""
        try:
            conn = self.connect()
            if not conn:
                return None
                
            cur = conn.cursor()
            
            # Get active connections
            cur.execute("""
                SELECT count(*) as active_connections,
                       count(*) filter (where state = 'idle') as idle_connections,
                       count(*) filter (where state = 'active') as active_queries
                FROM pg_stat_activity
                WHERE datname = current_database()
            """)
            
            stats = cur.fetchone()
            return {
                'active_connections': stats[0],
                'idle_connections': stats[1],
                'active_queries': stats[2]
            }
            
        except Exception as e:
            logger.error(f"Failed to get connection stats: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
                
    def get_table_stats(self):
        """Get table statistics"""
        try:
            conn = self.connect()
            if not conn:
                return None
                
            cur = conn.cursor()
            
            # Get table statistics
            cur.execute("""
                SELECT schemaname, tablename,
                       n_live_tup as row_count,
                       n_dead_tup as dead_rows,
                       last_vacuum,
                       last_autovacuum,
                       last_analyze,
                       last_autoanalyze
                FROM pg_stat_user_tables
                ORDER BY n_live_tup DESC
            """)
            
            tables = []
            for row in cur.fetchall():
                tables.append({
                    'schema': row[0],
                    'table': row[1],
                    'row_count': row[2],
                    'dead_rows': row[3],
                    'last_vacuum': row[4],
                    'last_autovacuum': row[5],
                    'last_analyze': row[6],
                    'last_autoanalyze': row[7]
                })
                
            return tables
            
        except Exception as e:
            logger.error(f"Failed to get table stats: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
                
    def get_index_stats(self):
        """Get index statistics"""
        try:
            conn = self.connect()
            if not conn:
                return None
                
            cur = conn.cursor()
            
            # Get index statistics
            cur.execute("""
                SELECT schemaname, tablename, indexname,
                       idx_scan as index_scans,
                       idx_tup_read as tuples_read,
                       idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
            """)
            
            indexes = []
            for row in cur.fetchall():
                indexes.append({
                    'schema': row[0],
                    'table': row[1],
                    'index': row[2],
                    'scans': row[3],
                    'tuples_read': row[4],
                    'tuples_fetched': row[5]
                })
                
            return indexes
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
                
    def get_query_stats(self):
        """Get query statistics"""
        try:
            conn = self.connect()
            if not conn:
                return None
                
            cur = conn.cursor()
            
            # Get query statistics
            cur.execute("""
                SELECT query,
                       calls,
                       total_time,
                       mean_time,
                       rows
                FROM pg_stat_statements
                ORDER BY total_time DESC
                LIMIT 10
            """)
            
            queries = []
            for row in cur.fetchall():
                queries.append({
                    'query': row[0],
                    'calls': row[1],
                    'total_time': row[2],
                    'mean_time': row[3],
                    'rows': row[4]
                })
                
            return queries
            
        except Exception as e:
            logger.error(f"Failed to get query stats: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
                
    def get_vacuum_stats(self):
        """Get vacuum statistics"""
        try:
            conn = self.connect()
            if not conn:
                return None
                
            cur = conn.cursor()
            
            # Get vacuum statistics
            cur.execute("""
                SELECT schemaname, tablename,
                       n_dead_tup as dead_tuples,
                       n_live_tup as live_tuples,
                       last_vacuum,
                       last_autovacuum
                FROM pg_stat_user_tables
                WHERE n_dead_tup > 0
                ORDER BY n_dead_tup DESC
            """)
            
            vacuum_stats = []
            for row in cur.fetchall():
                vacuum_stats.append({
                    'schema': row[0],
                    'table': row[1],
                    'dead_tuples': row[2],
                    'live_tuples': row[3],
                    'last_vacuum': row[4],
                    'last_autovacuum': row[5]
                })
                
            return vacuum_stats
            
        except Exception as e:
            logger.error(f"Failed to get vacuum stats: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
                
    def check_health(self):
        """Perform health check"""
        try:
            health = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'metrics': {}
            }
            
            # Check connections
            conn_stats = self.get_connection_stats()
            if conn_stats:
                health['metrics']['connections'] = conn_stats
                if conn_stats['active_connections'] > 100:  # Example threshold
                    health['status'] = 'warning'
                    
            # Check table stats
            table_stats = self.get_table_stats()
            if table_stats:
                health['metrics']['tables'] = table_stats
                for table in table_stats:
                    if table['dead_rows'] > table['row_count'] * 0.2:  # 20% dead rows
                        health['status'] = 'warning'
                        
            # Check vacuum stats
            vacuum_stats = self.get_vacuum_stats()
            if vacuum_stats:
                health['metrics']['vacuum'] = vacuum_stats
                for stat in vacuum_stats:
                    if stat['dead_tuples'] > stat['live_tuples'] * 0.3:  # 30% dead tuples
                        health['status'] = 'critical'
                        
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
            
    def monitor(self):
        """Continuous monitoring"""
        try:
            logger.info("Starting database monitoring")
            
            while True:
                # Perform health check
                health = self.check_health()
                
                # Log health status
                if health['status'] == 'healthy':
                    logger.info("Database health check passed")
                elif health['status'] == 'warning':
                    logger.warning("Database health check warning")
                elif health['status'] == 'critical':
                    logger.error("Database health check critical")
                else:
                    logger.error(f"Database health check failed: {health.get('error')}")
                    
                # Print metrics
                if 'metrics' in health:
                    print("\nDatabase Metrics:")
                    print(json.dumps(health['metrics'], indent=2))
                    
                # Wait for next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {str(e)}")
            
    def generate_report(self):
        """Generate monitoring report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'health': self.check_health(),
                'connections': self.get_connection_stats(),
                'tables': self.get_table_stats(),
                'indexes': self.get_index_stats(),
                'queries': self.get_query_stats(),
                'vacuum': self.get_vacuum_stats()
            }
            
            # Save report
            report_file = f"reports/db_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs('reports', exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            return None

def main():
    """Main function to run monitoring tasks"""
    monitor = DatabaseMonitor()
    
    if len(sys.argv) < 2:
        logger.error("Please specify action: monitor, report, or stats")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    
    if action == 'monitor':
        monitor.monitor()
    elif action == 'report':
        report_file = monitor.generate_report()
        if report_file:
            sys.exit(0)
        else:
            sys.exit(1)
    elif action == 'stats':
        # Get all statistics
        conn_stats = monitor.get_connection_stats()
        table_stats = monitor.get_table_stats()
        index_stats = monitor.get_index_stats()
        query_stats = monitor.get_query_stats()
        vacuum_stats = monitor.get_vacuum_stats()
        
        # Print statistics
        if conn_stats:
            print("\nConnection Statistics:")
            print(tabulate([conn_stats], headers='keys', tablefmt='grid'))
            
        if table_stats:
            print("\nTable Statistics:")
            print(tabulate(table_stats, headers='keys', tablefmt='grid'))
            
        if index_stats:
            print("\nIndex Statistics:")
            print(tabulate(index_stats, headers='keys', tablefmt='grid'))
            
        if query_stats:
            print("\nQuery Statistics:")
            print(tabulate(query_stats, headers='keys', tablefmt='grid'))
            
        if vacuum_stats:
            print("\nVacuum Statistics:")
            print(tabulate(vacuum_stats, headers='keys', tablefmt='grid'))
            
        sys.exit(0)
    else:
        logger.error("Invalid action. Use: monitor, report, or stats")
        sys.exit(1)

if __name__ == "__main__":
    main() 