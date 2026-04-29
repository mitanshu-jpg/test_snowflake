"""
IDSA – Step 1: ETL Knowledge Scraper
=====================================
Scrapes ETL/ELT documentation from public sources and loads the
structured content into Snowflake table: RAW_ETL_KNOWLEDGE.

Usage:
    pip install requests beautifulsoup4 pandas snowflake-snowpark-python python-dotenv
    python scraper.py

Environment variables (create a .env file):
    SNOWFLAKE_ACCOUNT   = your_account_identifier
    SNOWFLAKE_USER      = your_username
    SNOWFLAKE_PASSWORD  = your_password
    SNOWFLAKE_DATABASE  = IDSA_DB
    SNOWFLAKE_SCHEMA    = IDSA_SCHEMA
    SNOWFLAKE_WAREHOUSE = IDSA_WH
    SNOWFLAKE_ROLE      = SYSADMIN   (or your role)
"""

import os
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 15      # seconds
DELAY_BETWEEN_PAGES = 2   # polite crawl delay

# ── Target URLs ───────────────────────────────────────────────────────────────
SCRAPE_TARGETS: List[Dict] = [
    # ── Snowflake ──────────────────────────────────────────────────────────
    {
        "site": "Snowflake",
        "url": "https://docs.snowflake.com/en/user-guide/data-load-overview",
        "category": "architecture",
    },
    {
        "site": "Snowflake",
        "url": "https://docs.snowflake.com/en/user-guide/data-pipelines-intro",
        "category": "architecture",
    },
    {
        "site": "Snowflake",
        "url": "https://docs.snowflake.com/en/guides-overview-loading",
        "category": "connectors",
    },
    {
        "site": "Snowflake",
        "url": "https://www.snowflake.com/en/data-cloud/pricing-options/",
        "category": "pricing",
    },
    # ── Fivetran ───────────────────────────────────────────────────────────
    {
        "site": "Fivetran",
        "url": "https://www.fivetran.com/blog/what-is-etl",
        "category": "architecture",
    },
    {
        "site": "Fivetran",
        "url": "https://www.fivetran.com/connectors",
        "category": "connectors",
    },
    # ── dbt ────────────────────────────────────────────────────────────────
    {
        "site": "dbt",
        "url": "https://docs.getdbt.com/docs/introduction",
        "category": "architecture",
    },
    # ── Airbyte ────────────────────────────────────────────────────────────
    {
        "site": "Airbyte",
        "url": "https://docs.airbyte.com/understanding-airbyte/high-level-view",
        "category": "architecture",
    },
    {
        "site": "Airbyte",
        "url": "https://airbyte.com/pricing",
        "category": "pricing",
    },
    # ── AWS Glue ───────────────────────────────────────────────────────────
    {
        "site": "AWS Glue",
        "url": "https://aws.amazon.com/glue/pricing/",
        "category": "pricing",
    },
    {
        "site": "AWS Glue",
        "url": "https://aws.amazon.com/glue/features/",
        "category": "architecture",
    },
    # ── Azure Data Factory ─────────────────────────────────────────────────
    {
        "site": "Azure Data Factory",
        "url": "https://azure.microsoft.com/en-us/products/data-factory",
        "category": "architecture",
    },
    # ── Informatica ────────────────────────────────────────────────────────
    {
        "site": "Informatica",
        "url": "https://www.informatica.com/products/data-integration.html",
        "category": "architecture",
    },
    # ── Stitch ─────────────────────────────────────────────────────────────
    {
        "site": "Stitch",
        "url": "https://www.stitchdata.com/pricing/",
        "category": "pricing",
    },
]


# ── Scraping helpers ──────────────────────────────────────────────────────────

def fetch_page(url: str) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        log.warning("Failed to fetch %s  →  %s", url, e)
        return None


def extract_text_blocks(soup: BeautifulSoup, url: str, site: str, category: str) -> List[Dict]:
    """
    Extract heading + paragraph pairs from a BeautifulSoup page.
    Returns a list of row dicts ready for Snowflake insertion.
    """
    rows: List[Dict] = []

    # Walk through heading/paragraph elements in document order
    current_heading = "Introduction"
    buffer: List[str] = []

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = tag.get_text(separator=" ", strip=True)
        if not text or len(text) < 20:
            continue

        if tag.name in ("h1", "h2", "h3", "h4"):
            # Flush buffered paragraphs under the previous heading
            if buffer:
                rows.append(_make_row(url, site, category, current_heading, " ".join(buffer)))
                buffer = []
            current_heading = text[:500]
        else:
            buffer.append(text)

        # Flush every ~600 chars to keep chunks manageable
        if sum(len(b) for b in buffer) > 600:
            rows.append(_make_row(url, site, category, current_heading, " ".join(buffer)))
            buffer = []

    # Final flush
    if buffer:
        rows.append(_make_row(url, site, category, current_heading, " ".join(buffer)))

    return rows


def _make_row(url, site, category, heading, content) -> Dict:
    return {
        "SOURCE_URL": url[:2000],
        "SOURCE_SITE": site[:200],
        "HEADING": heading[:1000],
        "CONTENT": content,
        "CATEGORY": category[:200],
        "SCRAPED_AT": datetime.utcnow(),
    }


# ── Snowflake helpers ─────────────────────────────────────────────────────────

def get_snowflake_connection():
    """Create and return a Snowflake connection from env variables."""
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        database=os.environ.get("SNOWFLAKE_DATABASE", "IDSA_DB"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "IDSA_SCHEMA"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "IDSA_WH"),
        role=os.environ.get("SNOWFLAKE_ROLE", "SYSADMIN"),
    )


def bulk_insert(conn, rows: List[Dict]) -> int:
    """Insert a list of row dicts into RAW_ETL_KNOWLEDGE. Returns rows inserted."""
    if not rows:
        return 0

    sql = """
        INSERT INTO RAW_ETL_KNOWLEDGE
            (SOURCE_URL, SOURCE_SITE, HEADING, CONTENT, CATEGORY, SCRAPED_AT)
        VALUES
            (%(SOURCE_URL)s, %(SOURCE_SITE)s, %(HEADING)s,
             %(CONTENT)s,   %(CATEGORY)s,    %(SCRAPED_AT)s)
    """
    cursor = conn.cursor()
    try:
        cursor.executemany(sql, rows)
        conn.commit()
        return len(rows)
    finally:
        cursor.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== IDSA Scraper starting ===")

    # Validate env
    required_env = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    conn = get_snowflake_connection()
    log.info("Connected to Snowflake ✓")

    total_inserted = 0

    for target in SCRAPE_TARGETS:
        url = target["url"]
        site = target["site"]
        category = target["category"]

        log.info("Scraping  [%s]  %s", site, url)
        soup = fetch_page(url)
        if soup is None:
            continue

        rows = extract_text_blocks(soup, url, site, category)
        if rows:
            inserted = bulk_insert(conn, rows)
            log.info("  → Inserted %d rows", inserted)
            total_inserted += inserted
        else:
            log.info("  → No usable content found")

        time.sleep(DELAY_BETWEEN_PAGES)

    conn.close()
    log.info("=== Scraping complete. Total rows inserted: %d ===", total_inserted)
    log.info("Next step: Create Cortex Search Service via setup_snowflake.sql")


if __name__ == "__main__":
    main()
