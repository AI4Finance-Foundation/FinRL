import pandera as pa

DownloadDataSchema = pa.DataFrameSchema(
    {


        "timestamp": pa.Column("datetime64[ns, America/New_York]", nullable=False),
        "close": pa.Column("float64"),
        "high": pa.Column("float64"),
        "low": pa.Column("float64"),
        # "trade_count": pa.Column("int64"),
        "open": pa.Column("float64"),
        "volume": pa.Column("int64"),
        # "vwap": pa.Column("float64"),
        "tic": pa.Column("object"),
    },
)
