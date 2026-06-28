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

CREATE TABLE IF NOT EXISTS scan_runs (
    run_id VARCHAR PRIMARY KEY,
    scan_timestamp TIMESTAMP,
    segment VARCHAR,
    signal_count INTEGER,
    market_uptrend BOOLEAN,
    strategy_counts JSON
);

-- F&O intraday candles (5minute default)
CREATE TABLE IF NOT EXISTS candles_intraday (
    symbol VARCHAR,
    interval VARCHAR,
    trade_datetime TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    PRIMARY KEY(symbol, interval, trade_datetime)
);

CREATE TABLE IF NOT EXISTS fno_scan_results (
    scan_timestamp TIMESTAMP,
    strategy_id INTEGER,
    symbol VARCHAR,
    signal_datetime TIMESTAMP,
    score DOUBLE,
    trigger_price DOUBLE,
    metrics JSON
);

-- scan_run_id links each result row to a specific scan execution
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS scan_run_id VARCHAR;
ALTER TABLE fno_scan_results ADD COLUMN IF NOT EXISTS scan_run_id VARCHAR;

CREATE INDEX IF NOT EXISTS idx_scan_results_run ON scan_results(scan_run_id);
CREATE INDEX IF NOT EXISTS idx_fno_scan_results_run ON fno_scan_results(scan_run_id);

CREATE INDEX IF NOT EXISTS idx_candles_symbol ON candles(symbol);
CREATE INDEX IF NOT EXISTS idx_candles_date ON candles(trade_date);
CREATE INDEX IF NOT EXISTS idx_scan_results_ts ON scan_results(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_scan_results_strategy ON scan_results(strategy_id);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id VARCHAR PRIMARY KEY,
    segment VARCHAR,
    start_date DATE,
    end_date DATE,
    config JSON,
    summary JSON,
    created_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_intraday_symbol ON candles_intraday(symbol);
CREATE INDEX IF NOT EXISTS idx_intraday_dt ON candles_intraday(trade_datetime);

-- Paper trading: snapshot scan picks at entry prices, track real P/L later
CREATE TABLE IF NOT EXISTS paper_trade_batches (
    batch_id VARCHAR PRIMARY KEY,
    scan_run_id VARCHAR,
    created_at TIMESTAMP,
    entry_date DATE,
    position_count INTEGER,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS paper_trade_positions (
    position_id VARCHAR PRIMARY KEY,
    batch_id VARCHAR,
    symbol VARCHAR,
    strategy_id INTEGER,
    entry_date DATE,
    entry_price DOUBLE,
    score DOUBLE,
    UNIQUE(batch_id, symbol, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_positions_batch ON paper_trade_positions(batch_id);
CREATE INDEX IF NOT EXISTS idx_paper_batches_scan ON paper_trade_batches(scan_run_id);

-- Paper portfolio: cumulative holdings (1 share per scan pick)
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    holding_id VARCHAR PRIMARY KEY,
    scan_run_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    source_type VARCHAR NOT NULL,
    source_label VARCHAR NOT NULL,
    strategy_ids JSON,
    purchase_date DATE NOT NULL,
    purchase_price DOUBLE NOT NULL,
    quantity INTEGER DEFAULT 1,
    score DOUBLE,
    created_at TIMESTAMP NOT NULL,
    UNIQUE(scan_run_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio_holdings(symbol);
CREATE INDEX IF NOT EXISTS idx_portfolio_purchase ON portfolio_holdings(purchase_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_scan ON portfolio_holdings(scan_run_id);
