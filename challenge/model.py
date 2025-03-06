# model.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import xgboost as xgb
from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        self._model = None
        self._top_10_features = [
            "OPERA_Latin American Wings", "MES_7", "MES_10", "OPERA_Grupo LATAM",
            "MES_12", "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
        ]

    # Helper functions for feature engineering
    def _get_period_day(self, date: str) -> str:
        """Classify a datetime string into 'mañana', 'tarde', or 'noche' based on time."""
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("04:59", '%H:%M').time()
        
        if morning_min < date_time < morning_max:
            return 'mañana'
        elif afternoon_min < date_time < afternoon_max:
            return 'tarde'
        elif (evening_min < date_time < evening_max) or (night_min < date_time < night_max):
            return 'noche'

    def _is_high_season(self, fecha: str) -> int:
        """Determine if a date falls within high season periods, returning 1 if true, 0 if false."""
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        return 0

    def _get_min_diff(self, data: pd.Series) -> float:
        """Calculate the minute difference between operated (Fecha-O) and scheduled (Fecha-I) flight times."""
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return ((fecha_o - fecha_i).total_seconds()) / 60

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Add engineered features
        data['period_day'] = data['Fecha-I'].apply(self._get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self._is_high_season)
        data['min_diff'] = data.apply(self._get_min_diff, axis=1)

        # Create feature set with one-hot encoding for relevant categorical variables
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['MES'], prefix='MES'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO')
        ], axis=1)

        # Ensure all top_10_features exist in the DataFrame, fill missing with 0
        for feature in self._top_10_features:
            if feature not in features.columns:
                features[feature] = 0

        # Select only the top 10 features
        X = features[self._top_10_features]

        if target_column is not None:
            # Generate target if target_column is provided (for training)
            y = pd.DataFrame(np.where(data['min_diff'] > 15, 1, 0), columns=[target_column])
            return X, y
        return X

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Split data into training and testing sets for internal validation
        X_train, _, y_train, _ = train_test_split(features, target, test_size=0.33, random_state=42)

        # Calculate class balancing scale
        n_y0 = len(y_train[y_train.iloc[:, 0] == 0])
        n_y1 = len(y_train[y_train.iloc[:, 0] == 1])
        scale = n_y0 / n_y1 if n_y1 > 0 else 1  # Avoid division by zero

        # Train XGBoost model with class balancing
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(X_train, y_train.values.ravel())  # Convert target to 1D array

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Ensure features match the top 10 expected by the model
        for feature in self._top_10_features:
            if feature not in features.columns:
                features[feature] = 0
        features = features[self._top_10_features]

        # Make predictions and return as a list
        predictions = self._model.predict(features)
        return predictions.tolist()

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('../data/data.csv')
    model = DelayModel()

    # Preprocess data for training
    X, y = model.preprocess(data, target_column="delay")

    # Fit the model
    model.fit(X, y)

    # Predict on the same data (for demonstration)
    predictions = model.predict(X)
    print(f"Sample predictions: {predictions[:10]}")