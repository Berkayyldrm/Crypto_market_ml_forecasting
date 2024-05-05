import psycopg2
from psycopg2 import sql

class Postgres:
    def __init__(self, database, user, password, host, port):
        self.conn = psycopg2.connect(
            database=database,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def fetch_all(self, schema_name, table_name):
        query = f'SELECT * FROM "{schema_name}"."{table_name}" ORDER BY "date" ASC'
        self.cur.execute(query)
        rows = self.cur.fetchall()
        return rows
    
    def fetch_general(self, schema_name, table_name):
        query = f'SELECT * FROM "{schema_name}"."{table_name}" ORDER BY "date" ASC'
        self.cur.execute(query)
        rows = self.cur.fetchall()
        col_names = [desc[0] for desc in self.cur.description]
        return rows, col_names

    def fetch_all_with_column_names(self, schema_name, table_name):
        query = f'SELECT * FROM "{schema_name}"."{table_name}" ORDER BY "date" ASC'
        self.cur.execute(query)
        rows = self.cur.fetchall()

        query = f""" SELECT column_name FROM information_schema.columns WHERE table_schema = '{schema_name}' AND table_name = '{table_name}';"""
        self.cur.execute(query)
        column_names = [row[0] for row in self.cur.fetchall()]

        return rows, column_names
    
    def delete_all_tables(self):
        self.cur.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            AND table_type = 'BASE TABLE';
        """)

        for table_schema, table_name in self.cur.fetchall():
            self.cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE;").format(
                sql.Identifier(table_schema),
                sql.Identifier(table_name)
            ))
            print(f"Table deleted: {table_schema}.{table_name}")
    
    def create_schemas(self, schema_list):
        for schema in schema_list:
            self.cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s", (schema,))
            result = self.cur.fetchone()
            if result is None:
                self.cur.execute(f"CREATE SCHEMA {schema}")
                print(f"Schema Created: {schema}")

    def close(self):
        self.cur.close
        self.conn.close()