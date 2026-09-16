-- Runs once on first Postgres startup (docker-entrypoint-initdb.d).
-- POSTGRES_DB already created the app's "fraud_detection" database;
-- this adds a second database for the MLflow backend store.
CREATE DATABASE mlflow;
