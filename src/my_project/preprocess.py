import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LinearRegression


class Preprocessing:
    def __init__(self, cat_encoding, num_scaling, remove_outliers, data_valid):
        self.cat_encoding = cat_encoding
        self.num_scaling = num_scaling
        self.remove_outliers = remove_outliers
        self.data_valid = data_valid
        self.train_dataset = None

    def fit(self, dataset):
        self.train_dataset = dataset
        return self

    def transform(self, dataset):
        cat_dataset, num_dataset = self.cat_num_split(dataset)

        if self.data_valid:
            num_dataset = self.num_data_valid(num_dataset)

        if self.remove_outliers:
            num_dataset = self.remove_outliers_f(num_dataset)

        if self.cat_encoding == "frequency":
            modified_cat_dataset = self.freq_encoding(cat_dataset)
        elif self.cat_encoding == "ordinal":
            modified_cat_dataset = self.ordinal_encoding(cat_dataset)
        elif self.cat_encoding == "no":
            modified_cat_dataset = cat_dataset

        if self.num_scaling == "standard":
            modified_num_dataset = self.standard_scaling(num_dataset)
        elif self.num_scaling == "min_max":
            modified_num_dataset = self.min_max_scaling(num_dataset)
        elif self.num_scaling == "no":
            modified_num_dataset = num_dataset

        transformed_dataset = modified_num_dataset.join(modified_cat_dataset)

        return transformed_dataset

    # Divide dataset info categorical and numerical ones
    def cat_num_split(self, dataset):
        all_columns = dataset.columns.to_list()

        categorical_columns = []
        numerical_columns = []

        for item in all_columns:
            if item.startswith(("Wilderness_Area", "Soil_Type")):
                categorical_columns.append(item)
            else:
                numerical_columns.append(item)

        num_dataset = dataset[numerical_columns]
        cat_dataset = dataset[categorical_columns].astype("category")

        return cat_dataset, num_dataset

    # Create function for outliers removal
    def remove_outliers_f(self, num_dataset):
        for item in num_dataset.columns.to_list():
            data_mean = np.mean(num_dataset[item])
            data_std = np.std(num_dataset[item])

            cut_off = data_std * 3

            lower, upper = data_mean - cut_off, data_mean + cut_off

            m = len(num_dataset)

            num_dataset[num_dataset[item] < lower][item] = data_mean
            num_dataset[num_dataset[item] > upper][item] = data_mean
        return num_dataset

    # Create function for categorical features frequency encoding
    def freq_encoding(self, cat_dataset):

        wilderness = [
            x for x in cat_dataset.columns.to_list() if x.startswith("Wilderness")
        ]
        soil = [x for x in cat_dataset.columns.to_list() if x.startswith("Soil_Type")]

        m = len(cat_dataset)
        s_t = np.zeros((m), dtype="int32")
        w_a = np.zeros((m), dtype="int32")

        for i in range(m):
            for s in soil:
                if cat_dataset[s].iloc[i] == 1:
                    s_t[i] = int(re.findall(r"\d+", s)[-1])
            for w in wilderness:
                if cat_dataset[w].iloc[i] == 1:
                    w_a[i] = int(re.findall(r"\d+", w)[-1])

        modified_cat_dataset = pd.DataFrame(
            {"Soil_Type": s_t, "Wilderness_Area": w_a}, index=cat_dataset.index
        )

        s_m = modified_cat_dataset["Soil_Type"].value_counts()
        w_m = modified_cat_dataset["Wilderness_Area"].value_counts()

        w_m_c = (modified_cat_dataset["Wilderness_Area"].map(w_m)) / m
        s_m_c = (modified_cat_dataset["Soil_Type"].map(s_m)) / m

        modified_cat_dataset = pd.DataFrame(
            {"Soil_Type": s_m_c, "Wilderness_Area": w_m_c}, index=cat_dataset.index
        )
        return modified_cat_dataset

    # Create function for categorical features ordinal encoding
    def ordinal_encoding(self, cat_dataset):

        wilderness = [
            x for x in cat_dataset.columns.to_list() if x.startswith("Wilderness")
        ]
        soil = [x for x in cat_dataset.columns.to_list() if x.startswith("Soil_Type")]

        m = len(cat_dataset)
        s_t = np.zeros((m), dtype="int32")
        w_a = np.zeros((m), dtype="int32")

        for i in range(m):
            for s in soil:
                if cat_dataset[s].iloc[i] == 1:
                    s_t[i] = int(re.findall(r"\d+", s)[-1])
            for w in wilderness:
                if cat_dataset[w].iloc[i] == 1:
                    w_a[i] = int(re.findall(r"\d+", w)[-1])

        modified_cat_dataset = pd.DataFrame(
            {"Soil_Type": s_t, "Wilderness_Area": w_a}, index=cat_dataset.index
        )

        x_max = np.max(modified_cat_dataset, axis=0)
        x_min = np.min(modified_cat_dataset, axis=0)

        x_st = (modified_cat_dataset - x_min) / (x_max - x_min)

        modified_cat_dataset = x_st * (1 - (0)) + (0)

        return modified_cat_dataset

    # Create MinMax scaler function
    def min_max_scaling(self, num_dataset):
        num_columns = num_dataset.columns.to_list()
        x_max = np.max(self.train_dataset[num_columns], axis=0)
        x_min = np.min(self.train_dataset[num_columns], axis=0)

        x_st = (num_dataset - x_min) / (x_max - x_min)

        scaled_num_dataset = x_st * (1 - (0)) + (0)

        return scaled_num_dataset

    # Create Standard scaler function
    def standard_scaling(self, num_dataset):
        num_columns = num_dataset.columns.to_list()
        x_mean = np.mean(self.train_dataset[num_columns], axis=0)
        x_st_dev = np.std(self.train_dataset[num_columns], axis=0)

        scaled_num_dataset = (num_dataset - x_mean) / x_st_dev

        return scaled_num_dataset

    # Replace any questionable numerical data
    def num_data_valid(self, num_dataset):

        # For features defining distances replace negative values with the same positive
        modified_num_dataset = num_dataset.abs()

        # Fill 0-values using Linear Regression prediction\
        test_data = modified_num_dataset[["Hillshade_9am", "Hillshade_3pm"]][
            modified_num_dataset["Hillshade_3pm"] == 0
        ]
        train_data = modified_num_dataset[["Hillshade_9am", "Hillshade_3pm"]][
            modified_num_dataset["Hillshade_3pm"] != 0
        ]

        x_train = train_data["Hillshade_9am"]
        x_test = test_data["Hillshade_9am"]
        train_indeces = train_data.index
        test_indeces = test_data.index
        y_train = train_data["Hillshade_3pm"]
        model = LinearRegression()
        model.fit(x_train.values.reshape((-1, 1)), y_train.values.reshape((-1, 1)))

        y_pred = model.predict(x_test.values.reshape((-1, 1)))
        y_pred = y_pred.reshape((-1,))

        train_s = pd.Series(data=y_train, index=train_indeces, dtype="int32")
        test_s = pd.Series(data=y_pred, index=test_indeces, dtype="int32").round()

        whole = pd.concat([train_s, test_s]).sort_index()

        modified_num_dataset["Hillshade_3pm"] = whole.values

        return modified_num_dataset
