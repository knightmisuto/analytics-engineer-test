import pandas as pd
from sqlalchemy import create_engine
import logging

# --- Configuration ---
DB_FILE = 'yelp_weather.db'
DB_URI = f'sqlite:///{DB_FILE}'

# Optional: Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting analysis...")
    try:
        engine = create_engine(DB_URI)
        logging.info(f"Connected to database: {DB_URI}")
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        exit()

    # --- Define Analysis Queries ---

    # Query 1: Average Rating on Days with vs. Without Precipitation
    query_precip = """
    SELECT
        d.had_precipitation,
        COUNT(f.review_fact_key) AS num_reviews,
        AVG(f.rating) AS average_rating,
        MIN(f.rating) AS min_rating,
        MAX(f.rating) AS max_rating
    FROM fact_reviews f
    JOIN dim_date d ON f.date_key = d.date_key
    WHERE d.precipitation_mm IS NOT NULL -- Ensure we compare days with valid weather data
    GROUP BY d.had_precipitation;
    """

    # Query 2: Average Rating by Temperature Range (using Max Temp C)
    query_temp = """
    SELECT
        CASE
            WHEN d.max_temp_c < 5 THEN 'Very Cold (<5 C)'
            WHEN d.max_temp_c >= 5 AND d.max_temp_c < 15 THEN 'Cold (5-15 C)'
            WHEN d.max_temp_c >= 15 AND d.max_temp_c < 25 THEN 'Mild (15-25 C)'
            WHEN d.max_temp_c >= 25 AND d.max_temp_c < 35 THEN 'Warm (25-35 C)'
            WHEN d.max_temp_c >= 35 THEN 'Hot (>=35 C)'
            ELSE 'Unknown Temp'
        END AS temp_range,
        COUNT(f.review_fact_key) AS num_reviews,
        AVG(f.rating) AS average_rating
    FROM fact_reviews f
    JOIN dim_date d ON f.date_key = d.date_key
    WHERE d.max_temp_c IS NOT NULL -- Ensure valid temperature data
    GROUP BY temp_range
    ORDER BY MIN(d.max_temp_c); -- Order ranges logically
    """

    # Query 3: Does Rain Impact Ratings Differently on Weekends vs. Weekdays?
    query_weekend_precip = """
    SELECT
        d.is_weekend,
        d.had_precipitation,
        COUNT(f.review_fact_key) AS num_reviews,
        AVG(f.rating) AS average_rating
    FROM fact_reviews f
    JOIN dim_date d ON f.date_key = d.date_key
    WHERE d.precipitation_mm IS NOT NULL
    GROUP BY d.is_weekend, d.had_precipitation
    ORDER BY d.is_weekend, d.had_precipitation;
    """

    # --- Execute Queries and Print Results ---
    try:
        logging.info("\n--- Analysis: Impact of Precipitation ---")
        df_precip = pd.read_sql_query(query_precip, engine)
        print(df_precip.to_string(index=False)) # Use to_string for better console printing

        logging.info("\n--- Analysis: Impact of Temperature ---")
        df_temp = pd.read_sql_query(query_temp, engine)
        print(df_temp.to_string(index=False))

        logging.info("\n--- Analysis: Weekend vs Weekday Precipitation Impact ---")
        df_weekend = pd.read_sql_query(query_weekend_precip, engine)
        print(df_weekend.to_string(index=False))

        logging.info("\nAnalysis completed.")

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}")