-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Optional: Create a sample table to test the extension
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(3) -- Example for 3-dimensional vectors
);

INSERT INTO items (embedding) VALUES ('[0.1, 0.2, 0.3]'), ('[0.4, 0.5, 0.6]');
