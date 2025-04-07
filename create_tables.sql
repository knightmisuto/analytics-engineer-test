-- Drop tables if they exist (for easy re-running during testing)
DROP TABLE IF EXISTS fact_reviews;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_restaurant;
DROP TABLE IF EXISTS dim_user;

-- Dimension Table for Dates and Weather
CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY, -- Format: YYYYMMDD
    full_date DATE NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    day INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL, -- 0=Monday, 6=Sunday
    day_name VARCHAR(10),
    month_name VARCHAR(10),
    quarter INTEGER NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    -- Weather Attributes (Joined from GHCN-D)
    max_temp_c REAL, -- Using REAL for floating point in SQLite (DECIMAL equivalent)
    min_temp_c REAL,
    precipitation_mm REAL,
    had_precipitation BOOLEAN -- Flag if PRCP > 0
);

-- Dimension Table for Restaurants
CREATE TABLE dim_restaurant (
    restaurant_key INTEGER PRIMARY KEY AUTOINCREMENT, -- SQLite auto-increment
    business_id VARCHAR(50) NOT NULL UNIQUE, -- Natural key from Yelp data
    name VARCHAR(255),
    categories TEXT, -- Store comma-separated or as JSON string
    city VARCHAR(100),
    state VARCHAR(10),
    postal_code VARCHAR(20),
    latitude REAL,
    longitude REAL,
    business_stars REAL, -- Overall stars for the business
    business_review_count INTEGER
);

-- Dimension Table for Users
CREATE TABLE dim_user (
    user_key INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(50) NOT NULL UNIQUE, -- Natural key from Yelp data
    name VARCHAR(255)
    -- Add other user attributes if needed (yelping_since, etc.)
);

-- Fact Table for Reviews
CREATE TABLE fact_reviews (
    review_fact_key INTEGER PRIMARY KEY AUTOINCREMENT, -- Surrogate key for the fact record
    review_id VARCHAR(50) NOT NULL UNIQUE, -- Natural key from Yelp data
    -- Foreign Keys linking to Dimensions
    date_key INTEGER NOT NULL,
    restaurant_key INTEGER NOT NULL,
    user_key INTEGER NOT NULL,
    -- Metric(s)
    rating REAL NOT NULL, -- The star rating given in the review
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (restaurant_key) REFERENCES dim_restaurant(restaurant_key),
    FOREIGN KEY (user_key) REFERENCES dim_user(user_key)
);

-- Add Indexes for Performance (Good Practice)
CREATE INDEX idx_fact_reviews_date ON fact_reviews(date_key);
CREATE INDEX idx_fact_reviews_restaurant ON fact_reviews(restaurant_key);
CREATE INDEX idx_fact_reviews_user ON fact_reviews(user_key);
CREATE INDEX idx_dim_date_full_date ON dim_date(full_date);
CREATE INDEX idx_dim_restaurant_business_id ON dim_restaurant(business_id);
CREATE INDEX idx_dim_user_user_id ON dim_user(user_id);