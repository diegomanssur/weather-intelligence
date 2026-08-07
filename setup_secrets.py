"""
One-time setup script: creates the Databricks secret scope and stores the
Massive API key. Run this locally (with the Databricks CLI configured) or
from a notebook - never commit the resulting secret value anywhere.

Note: weather_client.py's NWS/Census integration needs NO secret here - the
NWS API and the Census geocoder are both free and keyless. It only needs the
NWS_USER_AGENT env var (set in app.yaml / the job yml's base_parameters, not
a Databricks secret), since api.weather.gov requires a descriptive
User-Agent but not an API key.

Usage:
    python setup_secrets.py
"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import getpass

w = WorkspaceClient()

# w.secrets.create_scope(scope="massive")
# w.secrets.put_secret(
#     scope="massive",
#     key="api-key",
#     string_value=getpass.getpass("Paste your Massive API key: ")
# )

# w.secrets.create_scope(scope="database")
w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase URL: ")
)


w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)

w.secrets.put_acl(
    scope="massive",
    principal="users",
    permission=workspace.AclPermission.READ,
)
