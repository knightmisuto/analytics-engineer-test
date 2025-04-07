# --- IMPORTANT NOTE ---
# The provided yelp_academic_dataset_business.json dataset was found to lack entries for Las Vegas.
# Therefore, this script has been adapted to analyze businesses in 'Reno', NV instead.
# However, it still utilizes the provided LAS VEGAS weather data due to data constraints
# for this exercise. This geographical mismatch is a major limitation acknowledged
# in the project documentation (README.md). The goal is to demonstrate the technical process.
# --- END NOTE ---

import pandas as pd
from sqlalchemy import create_engine
import sqlite3 # Use sqlite3 to execute the initial SQL script easily
import os
import logging # Optional: for better logging

# --- Configuration ---
# Adjust these paths to where your data is located
DATA_DIR = 'data'
YELP_DATA = 'Yelp JSON'
CLIMATE_DATA = 'Climate Data'
BUSINESS_FILE = os.path.join(DATA_DIR, YELP_DATA, 'yelp_academic_dataset_business.json')
REVIEW_FILE = os.path.join(DATA_DIR, YELP_DATA, 'yelp_academic_dataset_review.json')
USER_FILE = os.path.join(DATA_DIR, YELP_DATA, 'yelp_academic_dataset_user.json') # Optional: can get users from reviews
TEMP_FILE = os.path.join(DATA_DIR, CLIMATE_DATA, 'temperature-degreef.csv')
PRECIP_FILE = os.path.join(DATA_DIR, CLIMATE_DATA, 'las-vegas-mccarran-intl-ap-precipitation-inch.csv')

DB_FILE = 'yelp_weather.db'
SQL_SCHEMA_FILE = 'create_tables.sql'
DB_URI = f'sqlite:///{DB_FILE}' # Database connection string

# If data is too large, set nrows for testing, e.g., nrows=100000
# Set to None to load all data
DATA_LIMIT_ROWS = None # Or e.g., 100000

# Optional: Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def execute_sql_script(db_file, script_file):
    """Executes an SQL script against an SQLite database."""
    logging.info(f"Connecting to database {db_file} to execute schema.")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    with open(script_file, 'r') as sql_file:
        sql_script = sql_file.read()
    try:
        cursor.executescript(sql_script)
        conn.commit()
        logging.info(f"Successfully executed SQL script: {script_file}")
    except sqlite3.Error as e:
        logging.error(f"Error executing SQL script: {e}")
        conn.rollback()
    finally:
        conn.close()

def load_and_clean_weather(temp_path, precip_path):
    """Loads, cleans, joins, and transforms weather data."""
    logging.info("Loading weather data...")
    try:
        temp_df = pd.read_csv(temp_path)
        precip_df = pd.read_csv(precip_path)
    except FileNotFoundError as e:
        logging.error(f"Error loading weather file: {e}")
        return None

    # Clean column names if needed (e.g., remove leading/trailing spaces)
    temp_df.columns = temp_df.columns.str.strip()
    precip_df.columns = precip_df.columns.str.strip()

    # Convert date columns
    weather_date_format = '%Y%m%d' # Format code for YYYYMMDD
    temp_df['date'] = pd.to_datetime(temp_df['date'], format=weather_date_format, errors='coerce')
    precip_df['date'] = pd.to_datetime(precip_df['date'], format=weather_date_format, errors='coerce')

    # Drop rows where date conversion failed
    temp_df.dropna(subset=['date'], inplace=True)
    precip_df.dropna(subset=['date'], inplace=True)

    # Select and rename columns
    temp_df = temp_df[['date', 'min', 'max']].rename(columns={'min': 'min_temp_f', 'max': 'max_temp_f'})
    precip_df = precip_df[['date', 'precipitation']].rename(columns={'precipitation': 'precipitation_in'})

    # Merge weather data
    weather_df = pd.merge(temp_df, precip_df, on='date', how='outer')

    # Ensure source columns are numeric before attempting calculations.
    # 'coerce' will turn non-numeric entries (like "Trace") into NaN
    weather_df['precipitation_in'] = pd.to_numeric(weather_df['precipitation_in'], errors='coerce')
    weather_df['min_temp_f'] = pd.to_numeric(weather_df['min_temp_f'], errors='coerce')
    weather_df['max_temp_f'] = pd.to_numeric(weather_df['max_temp_f'], errors='coerce')

    # Convert units
    weather_df['min_temp_c'] = (weather_df['min_temp_f'] - 32) * 5.0/9.0
    weather_df['max_temp_c'] = (weather_df['max_temp_f'] - 32) * 5.0/9.0
    weather_df['precipitation_mm'] = weather_df['precipitation_in'] * 25.4

    # Handle missing values AFTER conversion
    weather_df['precipitation_mm'] = weather_df['precipitation_mm'].fillna(0)
    # Decide how to handle missing temps (e.g., leave as NaN, ffill, bfill) - leaving NaN here
    weather_df['min_temp_c'] = weather_df['min_temp_c'].round(1)
    weather_df['max_temp_c'] = weather_df['max_temp_c'].round(1)
    weather_df['precipitation_mm'] = weather_df['precipitation_mm'].round(1)

    logging.info("Weather data processed.")
    return weather_df[['date', 'min_temp_c', 'max_temp_c', 'precipitation_mm']]

def load_and_filter_businesses(business_path, nrows=None):
    """
    Loads business data, filters for a specified TARGET_CITY within Nevada (NV),
    and then filters for restaurant-related categories.
    Handles potential case/whitespace variations in city names.
    """
    logging.info("Loading business data...")
    try:
        business_df = pd.read_json(business_path, lines=True, nrows=nrows)
        logging.info(f"Loaded {len(business_df)} total businesses.")
        if business_df.empty:
            logging.warning("Business DataFrame is empty after loading.")
            return None
    except FileNotFoundError:
        logging.error(f"Business file not found: {business_path}")
        return None
    except ValueError as e:
         logging.error(f"Error reading business JSON: {e}. Check file format.")
         return None

    # --- State Filtering ---
    # Ensure 'state' column exists
    if 'state' not in business_df.columns:
        logging.error("Column 'state' not found in business data. Cannot filter by state.")
        return None

    nevada_businesses = business_df[business_df['state'] == 'NV'].copy()
    logging.info(f"Found {len(nevada_businesses)} businesses with State='NV'.")

    if nevada_businesses.empty:
        logging.warning("No businesses found with State='NV'. Cannot proceed.")
        return nevada_businesses # Return empty

    # --- City Filtering ---
    # Ensure 'city' column exists
    if 'city' not in nevada_businesses.columns:
        logging.error("Column 'city' not found in Nevada business data. Cannot filter by city.")
        return None

    # !!! --- IMPORTANT: SET YOUR TARGET CITY HERE --- !!!
    # Replace 'Reno' with the city you want to analyze from the NV list
    TARGET_CITY = 'Reno'
    # --- END IMPORTANT ---

    target_city_lower = TARGET_CITY.lower() # Use lowercase for case-insensitive comparison

    logging.info(f"Attempting to filter for City = '{TARGET_CITY}' (case-insensitive) in State = 'NV'")

    # Filter by city using case-insensitive and stripped comparison
    target_city_businesses = nevada_businesses[
        nevada_businesses['city'].str.strip().str.lower() == target_city_lower
    ].copy()

    logging.info(f"Found {len(target_city_businesses)} businesses matching State='NV' and City='{TARGET_CITY}' (case-insensitive).")

    if target_city_businesses.empty:
        logging.warning(f"No businesses found matching City='{TARGET_CITY}' (case-insensitive) within NV.")
        return target_city_businesses # Return empty, error handled later

    # --- Category Filtering ---
    # Ensure 'categories' column exists
    if 'categories' not in target_city_businesses.columns:
         logging.warning("Column 'categories' not found in filtered business data. Cannot filter by category.")
         # Decide if you want to return here or proceed without category filter
         return target_city_businesses # Returning matches based on location only

    logging.info(f"Checking categories for matched '{TARGET_CITY}' businesses before filtering:")
    # Optional: Print unique categories for debugging if needed
    # try:
    #     all_categories = set()
    #     valid_categories = target_city_businesses['categories'].dropna().astype(str)
    #     for cat_string in valid_categories:
    #         all_categories.update(c.strip() for c in cat_string.split(','))
    #     print(f"Unique Categories Found in '{TARGET_CITY}' Businesses (sample):", sorted(list(all_categories))[:50])
    # except Exception as e:
    #     print(f"Could not process categories for printing: {e}")

    # Filter for restaurants/food related businesses
    category_filter_regex = 'Restaurant|Food' # Adjust this regex if needed based on actual categories
    final_filtered_businesses = target_city_businesses[
        target_city_businesses['categories'].astype(str).str.contains(category_filter_regex, case=False, na=False)
    ].copy()

    logging.info(f"Found {len(final_filtered_businesses)} businesses matching location (State='NV', City='{TARGET_CITY}') AND category filter '{category_filter_regex}'.")

    if final_filtered_businesses.empty:
         logging.warning(f"No businesses found matching category filter '{category_filter_regex}' within '{TARGET_CITY}', NV.")
         # Return empty DataFrame, subsequent steps (like loading reviews) might fail if they expect businesses.

    return final_filtered_businesses


def load_reviews(review_path, business_ids_set, chunk_size=50000, nrows=None): # Added chunk_size, changed business_ids to a set for faster lookup
    """
    Loads review data in chunks, filtering each chunk for specific business IDs.
    Returns a list of filtered DataFrames (chunks).
    """
    logging.info("Loading review data in chunks...")
    if not isinstance(business_ids_set, set):
         logging.warning("Converting business_ids to set for efficient filtering.")
         business_ids_set = set(business_ids_set) # Convert to set for faster 'isin' check inside loop

    if not business_ids_set:
         logging.warning("No business IDs provided to filter reviews. Returning empty list.")
         return []

    filtered_chunks = []
    total_lines_read = 0
    total_reviews_kept = 0

    try:
        # Use chunksize to iterate
        reader = pd.read_json(review_path, lines=True, chunksize=chunk_size)

        for i, chunk in enumerate(reader):
            logging.info(f"Processing review chunk {i+1}...")

            # Filter chunk for the selected business IDs
            filtered_chunk = chunk[chunk['business_id'].isin(business_ids_set)]

            if not filtered_chunk.empty:
                 # Convert review date (keep only date part)
                 filtered_chunk['date'] = pd.to_datetime(filtered_chunk['date'], errors='coerce').dt.normalize()
                 filtered_chunk.dropna(subset=['date'], inplace=True) # Drop reviews with invalid dates

                 # Select necessary columns
                 filtered_chunks.append(filtered_chunk[['review_id', 'user_id', 'business_id', 'stars', 'date']])
                 total_reviews_kept += len(filtered_chunk)
                 logging.info(f"   Kept {len(filtered_chunk)} reviews from this chunk. Total kept so far: {total_reviews_kept}")

            total_lines_read += len(chunk) # Or use chunk_size, accounting for last smaller chunk

            # Apply nrows limit if specified
            if nrows is not None and total_lines_read >= nrows:
                logging.info(f"Reached specified limit of {nrows} lines read.")
                break

    except FileNotFoundError:
        logging.error(f"Review file not found: {review_path}")
        return None # Signal error
    except ValueError as e:
         logging.error(f"Error reading review JSON: {e}. Check file format/chunksize.")
         return None # Signal error
    except Exception as e:
         logging.error(f"An unexpected error occurred reading reviews: {e}")
         return None # Signal error


    if not filtered_chunks:
         logging.warning("No reviews found matching the provided business IDs after processing all chunks.")
         return [] # Return empty list, not None, to indicate successful run but no data

    logging.info(f"Finished processing review chunks. Total reviews kept: {total_reviews_kept}")
    return filtered_chunks # Return list of DataFrames

def create_dim_date(min_date, max_date, weather_df):
    """Creates the date dimension table including weather data."""
    logging.info("Creating Date Dimension...")
    # Create a full date range
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    dim_date_df = pd.DataFrame({'full_date': all_dates})

    # Generate date attributes
    dim_date_df['year'] = dim_date_df['full_date'].dt.year
    dim_date_df['month'] = dim_date_df['full_date'].dt.month
    dim_date_df['day'] = dim_date_df['full_date'].dt.day
    dim_date_df['day_of_week'] = dim_date_df['full_date'].dt.dayofweek # Monday=0, Sunday=6
    dim_date_df['day_name'] = dim_date_df['full_date'].dt.strftime('%A')
    dim_date_df['month_name'] = dim_date_df['full_date'].dt.strftime('%B')
    dim_date_df['quarter'] = dim_date_df['full_date'].dt.quarter
    dim_date_df['is_weekend'] = dim_date_df['day_of_week'].isin([5, 6])
    dim_date_df['date_key'] = dim_date_df['full_date'].dt.strftime('%Y%m%d').astype(int)

    # Merge with weather data
    dim_date_df = pd.merge(dim_date_df, weather_df, left_on='full_date', right_on='date', how='left')
    dim_date_df.drop(columns=['date'], inplace=True) # Drop redundant date column

    # Create boolean weather flag
    # Ensure precipitation_mm is numeric before comparison
    dim_date_df['precipitation_mm'] = pd.to_numeric(dim_date_df['precipitation_mm'], errors='coerce').fillna(0)
    dim_date_df['had_precipitation'] = dim_date_df['precipitation_mm'] > 0

    logging.info("Date Dimension created.")
    # Select final columns matching SQL schema (order doesn't matter for loading)
    final_cols = ['date_key', 'full_date', 'year', 'month', 'day', 'day_of_week',
                  'day_name', 'month_name', 'quarter', 'is_weekend', 'max_temp_c',
                  'min_temp_c', 'precipitation_mm', 'had_precipitation']
    return dim_date_df[final_cols]


def create_dim_restaurant(business_df):
    """Creates the restaurant dimension table."""
    logging.info("Creating Restaurant Dimension...")
    dim_rest_df = business_df[['business_id', 'name', 'categories', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count']].copy()
    # Rename columns to match SQL schema
    dim_rest_df = dim_rest_df.rename(columns={'stars': 'business_stars', 'review_count': 'business_review_count'})

    # Convert categories array/list to string if needed (e.g., comma-separated)
    # Check the type first
    if not pd.api.types.is_string_dtype(dim_rest_df['categories']) and dim_rest_df['categories'].notna().any():
         # Handle potential lists/arrays - join them into a string
         mask = dim_rest_df['categories'].apply(lambda x: isinstance(x, (list, tuple)))
         dim_rest_df.loc[mask, 'categories'] = dim_rest_df.loc[mask, 'categories'].apply(lambda x: ', '.join(x) if x else None)
         # If it's not lists but something else unexpected, convert to string directly
         dim_rest_df['categories'] = dim_rest_df['categories'].astype(str)


    # Add surrogate key
    dim_rest_df.reset_index(drop=True, inplace=True)
    dim_rest_df['restaurant_key'] = dim_rest_df.index + 1 # Simple key generation

    logging.info("Restaurant Dimension created.")
    # Select final columns
    final_cols = ['restaurant_key', 'business_id', 'name', 'categories', 'city', 'state',
                  'postal_code', 'latitude', 'longitude', 'business_stars', 'business_review_count']
    return dim_rest_df[final_cols]

def create_dim_user(reviews_df):
     """Creates the user dimension from unique user IDs in reviews."""
     logging.info("Creating User Dimension...")
     unique_users = reviews_df[['user_id']].drop_duplicates().copy()
     # Here you could merge with user.json if needed/available for more attributes like name

     # Placeholder for name if user.json isn't used
     unique_users['name'] = 'N/A'

     # Add surrogate key
     unique_users.reset_index(drop=True, inplace=True)
     unique_users['user_key'] = unique_users.index + 1

     logging.info("User Dimension created.")
     final_cols = ['user_key', 'user_id', 'name']
     return unique_users[final_cols]


def create_fact_reviews(reviews_df, dim_date_map, dim_rest_map, dim_user_map):
    """Creates the fact table by merging reviews with dimension keys."""
    logging.info("Creating Fact Table...")
    fact_df = reviews_df[['review_id', 'date', 'business_id', 'user_id', 'stars']].copy()
    fact_df = fact_df.rename(columns={'stars': 'rating'}) # Rename to match schema

    # Map date to date_key
    fact_df = pd.merge(fact_df, dim_date_map[['full_date', 'date_key']], left_on='date', right_on='full_date', how='inner')
    # Inner join ensures we only keep reviews on dates present in our date dimension (which includes weather range)

    # Map business_id to restaurant_key
    fact_df = pd.merge(fact_df, dim_rest_map[['business_id', 'restaurant_key']], on='business_id', how='inner')
    # Inner join ensures we only keep reviews for businesses in our restaurant dimension

    # Map user_id to user_key
    fact_df = pd.merge(fact_df, dim_user_map[['user_id', 'user_key']], on='user_id', how='inner')
    # Inner join ensures we only keep reviews from users in our user dimension

    # Select and order final columns
    final_cols = ['review_id', 'date_key', 'restaurant_key', 'user_key', 'rating']
    fact_df = fact_df[final_cols]

    logging.info(f"Fact Table created with {len(fact_df)} rows.")
    return fact_df

def load_to_db(engine, table_name, df):
    """Loads a DataFrame into a database table."""
    logging.info(f"Loading data into {table_name}...")
    try:
        # Using 'replace' for simplicity in testing; use 'append' if running incrementally
        # Due to issues automatically creating tables with 'replace' in SQLite, we use 'append'
        # and create the table manually with the correct schema.
        df.to_sql(table_name, engine, if_exists='append', index=False)
        logging.info(f"Successfully loaded data into {table_name}.")
    except Exception as e:
        logging.error(f"Error loading data into {table_name}: {e}")

# Test --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting ETL process...")

    # --- STEP 1: SCHEMA CREATION IS NOW DONE MANUALLY ---
    # The following lines are commented out because we assume
    # the schema has been created using 'sqlite3 yelp_weather.db < create_tables.sql'
    # or '.read create_tables.sql' inside the sqlite3 prompt BEFORE running this script.
    # -----------------------------------------------------
    # # if os.path.exists(DB_FILE):
    # #    logging.warning(f"Database file {DB_FILE} already exists.")
    # #    # Keep the manually created DB file, do not remove.
    # # execute_sql_script(DB_FILE, SQL_SCHEMA_FILE) # <<< THIS IS SKIPPED
    # -----------------------------------------------------

    # Create SQLAlchemy engine (Still needed to connect to the existing DB)
    logging.info(f"Connecting to existing database: {DB_URI}")
    engine = create_engine(DB_URI)

    # 2. Process Weather Data
    weather_df = load_and_clean_weather(TEMP_FILE, PRECIP_FILE)
    if weather_df is None:
        logging.error("Failed to process weather data. Exiting.")
        exit()

    # 3. Process Business Data
    business_df = load_and_filter_businesses(BUSINESS_FILE, nrows=DATA_LIMIT_ROWS)
    if business_df is None or business_df.empty:
        logging.error("Failed to load or filter business data. Exiting.")
        exit()
    # Convert business IDs to a set for faster review filtering
    lv_restaurant_ids_set = set(business_df['business_id'].unique())
    if not lv_restaurant_ids_set:
        logging.error("No valid business IDs found after filtering businesses. Cannot filter reviews.")
        exit()

    # 4. Process Review Data (using chunks)
    # Pass the SET of business IDs
    review_chunks = load_reviews(REVIEW_FILE, lv_restaurant_ids_set, chunk_size=50000, nrows=DATA_LIMIT_ROWS)

    if review_chunks is None: # Check if load_reviews signaled an error
        logging.error("Failed to load or filter review data due to errors. Exiting.")
        exit()
    elif not review_chunks: # Check if the list is empty (no reviews found)
         logging.warning("No matching reviews found for the selected businesses. Exiting.")
         exit()
    else:
         # Combine the filtered chunks into one DataFrame
         logging.info("Combining filtered review chunks...")
         reviews_df = pd.concat(review_chunks, ignore_index=True)
         logging.info(f"Combined DataFrame has {len(reviews_df)} reviews.")

    # Check if reviews_df is empty after concatenation
    if reviews_df.empty:
         logging.error("Review DataFrame is empty after concatenation, cannot proceed.")
         exit()

    # Determine date range for dim_date
    min_review_date = reviews_df['date'].min()
    max_review_date = reviews_df['date'].max()
    # Add checks for weather_df date validity here for safety
    if weather_df is None or weather_df.empty or weather_df['date'].isna().all():
        logging.error("Valid weather data not available for date range calculation. Exiting.")
        exit()
    min_weather_date = weather_df['date'].min()
    max_weather_date = weather_df['date'].max()
    if pd.isna(min_weather_date) or pd.isna(max_weather_date):
        logging.error("Weather data has invalid min/max dates. Exiting.")
        exit()

    logging.info(f"Review date range: Min={min_review_date}, Max={max_review_date}")
    logging.info(f"Weather date range: Min={min_weather_date}, Max={max_weather_date}")

    # Use the intersection of dates available in both datasets
    start_date = max(min_review_date, min_weather_date)
    end_date = min(max_review_date, max_weather_date)

    # Check the calculated range
    if start_date > end_date:
         logging.error(f"Calculated start_date ({start_date}) is after end_date ({end_date}). Check input date ranges.")
         exit()
    if not isinstance(start_date, pd.Timestamp) or not isinstance(end_date, pd.Timestamp):
        logging.error("Calculated start_date or end_date is not a valid Timestamp.")
        exit()

    logging.info(f"Effective date range for analysis: {start_date.date()} to {end_date.date()}")

    # 5. Create Dimensions (These lines define the variables)
    logging.info("Creating dimensions...")
    dim_date = create_dim_date(start_date, end_date, weather_df)
    dim_restaurant = create_dim_restaurant(business_df)
    dim_user = create_dim_user(reviews_df) # Using combined reviews_df

    # Add checks after creation
    if dim_date is None or dim_date.empty:
         logging.error("Date dimension creation failed or resulted in an empty table. Exiting.")
         exit()
    if dim_restaurant is None or dim_restaurant.empty:
         logging.error("Restaurant dimension creation failed or resulted in an empty table. Exiting.")
         exit()
    if dim_user is None or dim_user.empty:
         logging.error("User dimension creation failed or resulted in an empty table. Exiting.")
         exit()
    logging.info("Dimensions created successfully.")

    # Filter the combined reviews_df to match the final date range in dim_date
    logging.info(f"Filtering combined reviews to date range: {start_date.date()} to {end_date.date()}...")
    reviews_df = reviews_df[(reviews_df['date'] >= start_date) & (reviews_df['date'] <= end_date)]
    logging.info(f"Reviews remaining after date filtering: {len(reviews_df)}")

    if reviews_df.empty:
        logging.error("No reviews remaining after filtering by final date range. Cannot create fact table.")
        exit()

    # 6. Create Fact Table (This defines the variable)
    logging.info("Creating fact table...")
    fact_reviews = create_fact_reviews(reviews_df, dim_date[['full_date', 'date_key']], dim_restaurant[['business_id', 'restaurant_key']], dim_user[['user_id', 'user_key']])

    if fact_reviews is None or fact_reviews.empty:
         logging.error("Fact table creation failed or resulted in an empty table.")
         exit()
    logging.info("Fact table created successfully.")

    # 7. Load DataFrames to Database
    # Make sure the load_to_db function DEFINITION uses if_exists='append'
    logging.info("Loading dimensions and fact table to database...")
    load_to_db(engine, 'dim_date', dim_date)           # Now dim_date exists
    load_to_db(engine, 'dim_restaurant', dim_restaurant) # Now dim_restaurant exists
    load_to_db(engine, 'dim_user', dim_user)           # Now dim_user exists
    load_to_db(engine, 'fact_reviews', fact_reviews)   # Now fact_reviews exists

    logging.info("ETL process completed successfully.")