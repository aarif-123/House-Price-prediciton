from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for House Price feature engineering."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        yr_sold = data["YrSold"] if "YrSold" in data.columns else 2010
        data["HouseAge"] = yr_sold - data["YearBuilt"]
        data["TotalSF"] = data["GrLivArea"] + data["TotalBsmtSF"].fillna(0)
        data["Qual_Area_Interact"] = data["OverallQual"] * data["GrLivArea"]

        if "FullBath" in data.columns and "HalfBath" in data.columns:
            data["TotalBath"] = data["FullBath"] + (0.5 * data["HalfBath"])
        else:
            data["TotalBath"] = data["FullBath"] if "FullBath" in data.columns else 1

        return data
