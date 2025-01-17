from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import polars as pl
from polars import DataFrame

from opendeclaro.degiro.utils import (
    filter_df_inside_dates,
    filter_rowdate_inside_dates,
    opposite_transaction,
)


@dataclass
class ReturnsGlobal:
    isin_summary: pl.DataFrame
    global_return: float


class FIFO:
    def __init__(self, row: dict, df: DataFrame):
        """Initialization of class

        Parameters
        ----------
        row : dict
            dictionary containing the row of a transaction of the stocks dataframe
        df : DataFrame
            dataframe containing stock transactions
        """
        self.row = row
        self.df = df

    def opp_df(self, opp_df: DataFrame = pl.DataFrame([])) -> DataFrame:
        """Computes the opposite transaction to the traded row (e.g. if the traded row is "sell", it returns a dataframe
        containing "buy" transactions).
        It creates a "pending" column with the pending stocks of the transaction which are available for next trades,
        and a "shares_effective" column that defines which is the number of shares from the transaction effective to a
        given trade.


        Parameters
        ----------
        opp_df : DataFrame, optional
            dataframe with the opposite transaction to the traded row, by default pl.DataFrame([])

        Returns
        -------
        DataFrame
            dataframe with the opposite transaction to the traded row, after the FIFO method is applied
        """
        if opp_df.is_empty():
            opp_df_affected = (
                self.df.sort("value_date")
                .filter(
                    (pl.col("action") == opposite_transaction(self.row["action"]))
                    & (pl.col("value_date") < self.row["value_date"])
                    & (pl.col("unintended") == False)
                )
                .with_columns(
                    pl.cum_sum("number").sub(self.row["shares_effective"]).sub(pl.col("number")).alias("pending"),
                )
                .filter(pl.col("pending") < 0)
                .with_columns(
                    pl.when(abs(pl.col("pending")) > pl.col("number"))
                    .then(pl.col("number"))
                    .otherwise(abs(pl.col("pending")))
                    .alias("shares_effective")
                )
                .with_columns((pl.col("number") - pl.col("shares_effective")).alias("number"))
            )
            opp_df_untouched = (
                self.df.sort("value_date")
                .filter((pl.col("action") == opposite_transaction(self.row["action"])))
                .with_columns(
                    pl.cum_sum("number").sub(self.row["shares_effective"]).sub(pl.col("number")).alias("pending"),
                )
                .filter(pl.col("pending") >= 0)
                .with_columns(pl.lit(0.0).alias("shares_effective"))
            )

        else:
            opp_df_affected = (
                opp_df.sort("value_date")
                .filter(pl.col("number") > 0)
                .with_columns(
                    pl.cum_sum("number").sub(self.row["shares_effective"]).sub(pl.col("number")).alias("pending")
                )
                .filter(pl.col("pending") < 0)
                .with_columns(
                    pl.when(abs(pl.col("pending")) > pl.col("number"))
                    .then(pl.col("number"))
                    .otherwise(abs(pl.col("pending")))
                    .alias("shares_effective")
                )
                .with_columns((pl.col("number") - pl.col("shares_effective")).alias("number"))
            )
            opp_df_untouched = (
                opp_df.sort("value_date")
                .filter(pl.col("action") == opposite_transaction(self.row["action"]))
                .with_columns(
                    pl.cum_sum("number").sub(self.row["shares_effective"]).sub(pl.col("number")).alias("pending"),
                )
                .filter(pl.col("pending") >= 0)
                .with_columns(pl.lit(0.0).alias("shares_effective"))
            )

        return pl.concat([opp_df_affected, opp_df_untouched])


# fmt: off
class Returns:
    def __init__(self, df: DataFrame, end_date: Optional[str] = None, start_date: Optional[str] = None):
        """Initialization of Class

        Parameters
        ----------
        df : DataFrame
            dataframe containing stock transactions
        end_date : Optional[str], optional
            ending date from which valid transactions are filtered, by default None
        start_date : Optional[str], optional
            starting date from which valid transactions are filtered, by default None
        """
        self.df = df
        self.end_date = end_date
        self.start_date  = start_date
    
    @property
    def unique_isin(self) -> List[str]:
        isin_list = (
            filter_df_inside_dates(self.df, col_name="value_date", start_date=self.start_date, end_date=self.end_date)
            .filter(pl.col("isin").str.len_bytes() > 1)
            .select(pl.col("isin").unique())
            .to_series()
            .to_list()
        )
        return isin_list
    
    def return_on_all_stocks(self) -> ReturnsGlobal:
        return_all = 0
        isin_dict = defaultdict(list)
        for isin in self.unique_isin:
            return_isin = self.return_on_stock(isin)
            isin_dict["isin"].append(isin)
            isin_dict["return"].append(return_isin)
            return_all += return_isin
        return ReturnsGlobal(pl.DataFrame(isin_dict), return_all)
    
    def return_on_stock(self, isin: str) -> float:
        """Compute the return of a given stock ISIN

        Parameters
        ----------
        isin : str
            ISIN of the stock traded

        Returns
        -------
        float
            return on the stock
        """
        df = self.df.filter(pl.col("isin") == isin).with_columns(pl.col("number").alias("number_orig"))
        opp_df = pl.DataFrame([])
        return_stock = 0
        for row in df.sort(pl.col("value_date")).iter_rows(named=True):
            if row["isin_change"] != None:
                df = (
                    self.df
                    .filter((pl.col("isin") == isin) | (pl.col("isin") == row["isin_change"]))
                    .with_columns(pl.col("number").alias("number_orig"))
                )
            stocks_before = self.get_stocks_purchased_before(row, df)
            if self.choose_compute_transaction(row, stocks_before) == True:
                row["date_2m_limit"] = row["value_date"] + timedelta(days=60)
                row["shares_effective"] = min(abs(stocks_before), row["number"])
                opp_df = FIFO(row, df).opp_df(opp_df)
                row_res = row["var"]/row["curr_rate"] + row["commision"]
                opp_df_res = (
                    (opp_df["var"]/opp_df["curr_rate"] + opp_df["commision"]) * opp_df["shares_effective"] / opp_df["number_orig"]
                ).sum()
                if (row_res + opp_df_res < 0) & (
                    opp_df.filter(pl.col("value_date") < row["date_2m_limit"]).shape != 
                    opp_df.filter(pl.col("value_date") < row["value_date"]).shape
                ):
                    return_stock += 0
                elif filter_rowdate_inside_dates(row["value_date"], self.start_date, self.end_date) == False:
                    return_stock += 0
                else:
                    return_stock += row_res + opp_df_res

        return return_stock
                

    @staticmethod
    def get_stocks_purchased_before(row: dict, df: DataFrame) -> float:
        """Returns the total number of stocks purchased for a df
        Positive: net long position
        Negative: net short position

        Parameters
        ----------
        row: dict
            dictionary containing the row of a transaction of the stocks dataframe
        df : DataFrame
            dataframe of transactions (before a particular time)

        Returns
        -------
        float
            number of stocks purchased (long if positive, short if negative, zero if no position)
        """
        stocks_purchased_before = (
            df
            .filter(
                (pl.col("date") < row["date"]) &
                (pl.col("action") == "buy")
            )
            .select(pl.col("number")).sum().item()
        )
        stocks_sold_before = (
            df
            .filter(
                (pl.col("date") < row["date"]) &
                (pl.col("action") == "sell")
            )
            .select(pl.col("number")).sum().item()
        )
        return stocks_purchased_before - stocks_sold_before
    
    @staticmethod
    def choose_compute_transaction(row: dict, stocks_before: float) -> bool:
        """Decide whether to compute the transaction or not

        Parameters
        ----------
        row : dict
            dictionary containing the row of a transaction of the stocks dataframe
        stocks_before : float
            number of stocks purchased (long if positive, short if negative, zero if no position)

        Returns
        -------
        bool
            True if return is to be computed, False otherwise
        """
        if (row["action"] == "sell") & (stocks_before <= 0):
            return False
        if (row["action"] == "sell") & (stocks_before > 0):
            return True
        if (row["action"] == "buy") & (stocks_before >= 0):
            return False
        if (row["action"] == "buy") & (stocks_before < 0):
            return True

# fmt: on
