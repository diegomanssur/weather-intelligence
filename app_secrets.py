import base64
from databricks.sdk import WorkspaceClient

# Initialize the Databricks client
w = WorkspaceClient()

# Connection details for weather-intelligence Lakebase project
USERNAME = "weather-reporter"
PASSWORD = input("Enter the password for weather-reporter role: ")  # Secure password input
HOST = "ep-broad-mud-d8hqfmp6.database.us-east-2.cloud.databricks.com"
PORT = "5432"
DATABASE = "databricks_postgres"

# Build the connection URL
connection_url = f"postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?sslmode=require"

# Encode to base64 (matching the format of your existing secret)
encoded_url = base64.b64encode(connection_url.encode("utf-8")).decode("utf-8")

# Create or update the secret
w.secrets.put_secret(
    scope="database",
    key="lakebase-weather-url",
    string_value=encoded_url
)

print("✓ Secret 'lakebase-weather-url' created successfully!")
print(f"  Host: {HOST}")
print(f"  Database: {DATABASE}")
print(f"  User: {USERNAME}")