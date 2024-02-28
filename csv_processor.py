import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from s2_bands import S2Bands
import numpy as np
import os


class CSVProcessor:
    @staticmethod
    def get_neighbour_columns():
        base = S2Bands.get_all_bands()
        cols = []
        for i in range(8):
            for b in base:
                cols.append(f"{b}_{i}")

        return cols

    @staticmethod
    def aggregate(complete, ag):
        df = pd.read_csv(complete)
        df.drop(columns=CSVProcessor.get_geo_columns(), axis=1, inplace=True)
        spatial_columns = CSVProcessor.get_spatial_columns(df)
        columns_to_agg = df.columns.drop(spatial_columns)

        agg_dict = {}
        agg_dict["counter"] = ("som", 'count')
        agg_dict["som_std"] = ("som", 'std')
        for col in columns_to_agg:
            agg_dict[col] = (col, "mean")

        df_group_object = df.groupby(spatial_columns)
        df_mean = df_group_object.agg(**agg_dict).reset_index()
        df_mean.insert(0, "cell", df_mean.index)
        df_mean = df_mean[df_mean["counter"] >= 1]
        df_mean = df_mean.round(4)
        df_mean.to_csv(ag, index=False)

    @staticmethod
    def make_ml_ready(ag, ml):
        df = pd.read_csv(ag)
        df = CSVProcessor.make_ml_ready_df(df)
        df = df.round(4)
        df.to_csv(ml, index=False)

    @staticmethod
    def make_ml_ready_df(df):
        for col in ["when"]:
            if col in df.columns:
                df.drop(inplace=True, columns=[col], axis=1)
        for col in df.columns:
            if col not in ["scene","row","column","counter","som_std","cell","row_offset","column_offset"]:
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
        return df

    @staticmethod
    def get_spatial_columns(df):
        spatial_columns = ["row", "column"]
        if "scene" in df.columns:
            spatial_columns = ["scene"] + spatial_columns
        return spatial_columns

    @staticmethod
    def get_geo_columns():
        return ["lon", "lat", "when"]

    @staticmethod
    def get_ml_columns():
        non_band_columns = ["scene","cell"]
        band_columns = S2Bands.get_all_bands()
        return non_band_columns, band_columns

    @staticmethod
    def get_computable_columns():
        non_band_columns, band_columns = CSVProcessor.get_ml_columns()
        return band_columns
