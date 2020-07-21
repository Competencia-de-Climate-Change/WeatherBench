import numpy as np
import xarray as xr
from sklearn.metrics import mean_squared_error # 'neg_root_mean_squared_error'

def load_test_data(var, ds=None, path=None, years=slice('2017', '2018')):
    """
    Load the test dataset. If z return z500, if t return t850.
    
    Parameters:
    ----------
        var (string) : variable name
        ds (xr.Dataset) : dataset
        path (string): Path to nc files 
        years (slice): Time window
    Returns:
    --------
        dataset: Concatenated dataset for 2017 and 2018
    """
    if (path is None) and (ds is None):
        raise ValueError('Give ds or path')

    if ds is None:
        ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
        if var in ['z', 't']:
            try:
                ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
            except ValueError:
                ds = ds.drop('level')
    elif path is None:
        ds = ds[var]
        if var in ['z', 't']:
            print('Selecting from ds...')
            try:
                ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
            except ValueError:
                ds = ds.drop('level')    
    return ds.sel(time=years)


def create_training_data(da, lead_time_h, return_valid_time=False, return_ds=False):
    """Function to split input and output by lead time."""
    X = da.isel(time=slice(0, -lead_time_h)) # Desde 0 hasta la ultima - 6 hrs
    y = da.isel(time=slice(lead_time_h, None)) # Desde 6 hasta la ultima hora
    valid_time = y.time
    
    _, nlat, nlon = da.shape
    
    # Must be a "flat" array ==> nlat*nlon
    X_np = X.values.reshape(-1, nlat*nlon)
    y_np = y.values.reshape(-1, nlat*nlon)
    
    if return_valid_time:
        return X_np, y_np, valid_time
    
    elif return_ds:
        return X, y
    
    return X_np, y_np 



def train_regression(model, data, lead_time_h, input_vars, output_vars, data_subsample=1, extra_args=None, verbose=False):
    """
    Create X and y, then train a linear regression and return the predictions.
    
    Params:
    ------
    model                : sklearn's object like model
    data      (iterable) : Contains data_train, data_test, nlat, nlon, data_std, data_mean
    lead_time_h    (int) : time between *now* and prediction time
    input_var (iterable) : Contains strings of the input variable names
    input_var (iterable) : Contains strings of the output variable names
    data_subsample (int) : Subsample value
    extra_args    (dict) : Extra arguments to be passed to the model
    verbose       (bool) : Verbose level
    
    Returns:
    -------
    pred_ds (xr.Dataset) : predictions dataset
    model_res   (object) : fitted model object
    """
    
    data_train, data_test, nlat, nlon, data_std, data_mean  = data
    
    X_train, y_train, X_test, y_test = [], [], [], []
    for v in input_vars:
        # Create Training Data for a lead_time_h prediction
        X, y = create_training_data(data_train[v], lead_time_h)
        X_train.append(X)

        if v in output_vars: 
            y_train.append(y)

        # Create Test Data for a lead_time_h prediction
        X, y, valid_time = create_training_data(data_test[v], lead_time_h, return_valid_time=True)
        X_test.append(X)
        
        if v in output_vars: 
            y_test.append(y)

    X_train, y_train, X_test, y_test = [np.concatenate(d, 1) for d in [X_train, y_train, X_test, y_test]]
    if data_subsample > 1:
        X_train = X_train[::data_subsample]
        y_train = y_train[::data_subsample]
    try:
        model_res = model(n_jobs=-1, **extra_args)
    except Exception:
        model_res = model(**extra_args)
    model_res.fit(X_train, y_train)
    
    mse_train = mean_squared_error(y_train, model_res.predict(X_train))
    mse_test  = mean_squared_error(y_test,  model_res.predict(X_test))

    if verbose:
        print(f'Train MSE = {mse_train}')
        print(f'Test  MSE = {mse_test}')

    preds = model_res.predict(X_test).reshape((-1, len(output_vars), nlat, nlon))
    
    # Save predictions + unnormalize
    preds_ds = []
    for idx, value in enumerate(output_vars):
        pred_array = xr.DataArray(
            preds[:, idx] * data_std[value].values + data_mean[value].values, 
            dims=['time', 'lat', 'lon'],
            coords={
                'time': valid_time,
                'lat': data_train.lat,
                'lon': data_train.lon
            },
            name=value
        )
        preds_ds.append(pred_array)

    return xr.merge(preds_ds), model_res, mse_train, mse_test


def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    
    Parameters:
    ----------
        y_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        y_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
    -------
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse
