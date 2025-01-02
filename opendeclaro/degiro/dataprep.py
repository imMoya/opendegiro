import polars as pl
from polars import DataFrame


# fmt:off
class DataPrep:
    def __init__(self, data: DataFrame):
        self.data = data
        
    def prepare_id_orders(self):
        df = (
            self.data
            .filter(
                (pl.col("id_order").str.len_chars() > 0) & 
                (pl.col("action").str.len_chars() > 0)
            )
            .select(pl.exclude("curr_rate"))
        )
        print(df)
        df_costs = (
            self.data
            .filter(
                (pl.col("id_order").str.len_chars() > 0) &
                (pl.col("action").str.len_chars().is_null()) & 
                (pl.col("desc").str.contains("Divisa")).not_()
            )
            .group_by("id_order")
            .agg(pl.col("var").sum().alias("commision"))
        )
        print(df_costs)
        df_curr_rate = (
            self.data
            .filter(
                (pl.col("curr_rate").is_not_null()) &
                (pl.col("id_order").str.len_chars() > 0)
            )
            .select(
                "id_order", "curr_rate"
            )
        )
        df_final = df.join(df_costs, on="id_order").join(df_curr_rate, left_on="id_order", right_on="id_order", how="left").sort("date", descending=True)
        return df_final.with_columns(pl.lit(False).alias("unintended"))
    
    def prepare_involuntary_orders(self):
        df = (
            self.data
            .filter(
                (pl.col("action").str.len_chars() > 0) & 
                (pl.col("id_order").str.len_chars() == 0)
            )
            .with_columns(pl.lit(None).alias("commision").cast(pl.Float32, strict=False))
            .with_columns(pl.lit(True).alias("unintended"))
        )
        return df.sort("date", descending=True)
    
    @property
    def stocks_orders(self) -> DataFrame:
        df_stocks = self.map_eur_curr_rate(
            pl.concat(
                [self.prepare_id_orders(),
                self.prepare_involuntary_orders().select(self.prepare_id_orders().columns)],
                how="align"
            ).unique().sort("date", descending=True)
        )
        df_stocks = self.add_isin_change_col(df_stocks)
        return df_stocks

    
    @staticmethod
    def map_eur_curr_rate(df: DataFrame) -> DataFrame:
        return df.with_columns(
            pl.when(pl.col("cashcur").str.contains("EUR"))
            .then(1.0)
            .otherwise(pl.col("curr_rate"))
            .alias("curr_rate")
        )
    
    @staticmethod
    def add_isin_change_col(df: DataFrame) -> DataFrame:
        df_isin = df.filter(
            (pl.col("unintended") == True) & 
            (pl.col("desc").str.contains("CAMBIO DE ISIN"))
        )
        df_updated_with_isin = df.with_columns(pl.lit(None).alias("isin_change"))
        for _, data in df_isin.group_by("value_date", maintain_order=True):
            for row in data.iter_rows(named=True):
                isin_opp = data.filter(pl.col("isin")!=row["isin"]).select("isin")
                df_updated_with_isin = df_updated_with_isin.with_columns(
                    pl.when(
                        (pl.col("isin") == row["isin"]) &
                        (pl.col("action") != row["action"]) &
                        (pl.col("value_date") > row["value_date"])
                    )
                    .then(isin_opp)
                    .otherwise(pl.col("isin_change"))
                    .alias("isin_change")
                )
        return df_updated_with_isin


# fmt:on
