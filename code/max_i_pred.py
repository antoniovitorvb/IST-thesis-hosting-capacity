import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

data_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')

def max_i_pred(lineCodes_df):
    """
    Predicts the maximum current (in kA) for each line code based on its parameters.
    
    Parameters:
    - lineCodes_df: DataFrame containing line codes with columns 'Name', 'R1', 'X1', 'C1'.
    
    Returns:
    - lineCodes_df: DataFrame with an additional column 'max_i_ka' containing the predicted values.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')

    # Load standard line codes
    StdLineCodes_df = pd.read_csv(os.path.join(data_dir, "StandardLineCodes.csv"), sep=';')
    # return(StdLineCodes_df.columns)
    cs_StdLineCodes = StdLineCodes_df.loc[StdLineCodes_df['type'] == 'cs']

    X = cs_StdLineCodes[['r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km']]
    Y = cs_StdLineCodes['max_i_ka']

    x_pred = lineCodes_df[['Name', 'R1', 'X1', 'C1']].set_index('Name')
    x_pred.columns = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_pred_scaled = scaler.fit_transform(x_pred)

    # Optional: Scale Y
    # scaler_Y = StandardScaler()
    # Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1)).flatten()

    # Use LOOCV to estimate error
    loo = LeaveOneOut()

    rig_model = Ridge(alpha=1.0)
    rig_scores = cross_val_score(rig_model, X_scaled, Y, cv=loo, scoring='neg_mean_squared_error')
    # rmse = np.sqrt(-rig_scores.mean())
    # print(f"Estimated RMSE via Ridge LOOCV: {rmse:.4f}")

    # Train on all data and predict
    rig_model.fit(X_scaled, Y)
    Y_pred_rig = rig_model.predict(x_pred_scaled)

    return Y_pred_rig