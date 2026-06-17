CREATE TABLE IF NOT EXISTS candles (
    symbol VARCHAR,
    trade_date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    PRIMARY KEY(symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS scan_results (
    scan_timestamp TIMESTAMP,
    strategy_id INTEGER,
    symbol VARCHAR,
    signal_date DATE,
    score DOUBLE,
    trigger_price DOUBLE,
    metrics JSON
);

CREATE TABLE IF NOT EXISTS scan_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol ON candles(symbol);
CREATE INDEX IF NOT EXISTS idx_candles_date ON candles(trade_date);
CREATE INDEX IF NOT EXISTS idx_scan_results_ts ON scan_results(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_scan_results_strategy ON scan_results(strategy_id);
