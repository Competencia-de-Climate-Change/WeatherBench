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


def input_vars_train_test(input_vars, data, lead_time_h):
    """
    input_vars  (iterable) : Contains strings of the input variable names
    data        (iterable) : Contains data_train ,data_test
    lead_time_h    (int) : time between *now* and prediction time
    """
    data_train, data_test = data
    
    X_train, X_test = [], [], 
    
    for idx, v in enumerate(input_vars):
        # Create Training Data for a lead_time_h prediction
        X, _ = create_training_data(data_train[v], lead_time_h)
        X_train.append(X)

        # Create Test Data for a lead_time_h prediction
        X, _, valid_time = create_training_data(data_test[v], lead_time_h, return_valid_time=True)
        X_test.append(X)
        
    return X_train, X_test, valid_time

def output_vars_train_test(output_vars, data, lead_time_h):
    """
    output_vars (iterable) : Contains strings of the output variable names
    data        (iterable) : Contains data_train ,data_test
    lead_time_h    (int) : time between *now* and prediction time
    """
    
    y_train, y_test = [], []
    data_train, data_test = data
    for v in output_vars:
        if v == '':
            continue
        
        # Create Training Data for a lead_time_h prediction
        _, y = create_training_data(data_train[v], lead_time_h)
        y_train.append(y)
            

        # Create Test Data for a lead_time_h prediction
        _, y, valid_time = create_training_data(data_test[v], lead_time_h, return_valid_time=True)
        y_test.append(y)
        
    return y_train, y_test

def create_X_y_time(input_vars, output_vars, data, lead_time, data_subsample=1):
    """
    Creates X, y and a valid datetime list from the given input_vars, output_vars.
    
    Raw data is in data = [data_train, data_test].
    
    Subsamples the X, y arrays if needed.
    
    Params:
    ------
    output_vars (iterable) : Contains strings of the output variable names
    data        (iterable) : Contains data_train, data_test
    lead_time_h    (int) : time between *now* and prediction time
    data_subsample (int) : Subsample value
    
    Returns:
    -------
    X_train
    y_train 
    X_test 
    y_test 
    valid_time
    """
    old_out = output_vars
    X_train, X_test, valid_time = input_vars_train_test(input_vars, data, lead_time)

    y_train, y_test = output_vars_train_test(output_vars, data, lead_time)

    # Add intercept
    X_train, y_train, X_test, y_test = [np.concatenate(d, 1) for d in [X_train, y_train, X_test, y_test]]
    
    if data_subsample > 1:
        X_train = X_train[::data_subsample]
        y_train = y_train[::data_subsample]
        
    ouput_vars = old_out
    return X_train, y_train, X_test, y_test, valid_time


def train_regression(model, data, num_outputs, extra_args=None, verbose=False):
    """
    Train a linear regression and return the predictions.
    
    Params:
    ------
    model                : sklearn's object like model
    data      (iterable) : Contains X_train, y_train, X_test, y_test, nlat, nlon
    num_outputs (int)    : Number of output variables
    extra_args    (dict) : Extra arguments to be passed to the model
    verbose       (bool) : Verbose level
    
    Returns:
    -------
    pred_ds (xr.Dataset) : predictions dataset
    model_res   (object) : fitted model object
    mse_train            : mse_train 
    mse_test             : mse_test
    """
    
    X_train, y_train, X_test, y_test, nlat, nlon = data
    
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
    
    preds = model_res.predict(X_test).reshape((-1, num_outputs, nlat, nlon))
    
    return preds, model_res, mse_train, mse_test


def unnormalize_preds(preds, output_vars, valid_time, lat_lon, data_std, data_mean):
    """
    output_vars (iterable) : Contains strings of the output variable names
    valid_time (iterable): Contains a list with the predicted datetimes
    lat_lon   (iterable) : Contains latitude and longitude arrays (coordinates)
    data_std  (xarray object) : Contains output_vars std's
    data_mean (xarray object) : Contains output_vars mean's
    """
    # Save predictions + unnormalize
    preds_ds = []
    for idx, value in enumerate(output_vars):
        pred_array = xr.DataArray(
            preds[:, idx] * data_std[value].values + data_mean[value].values, 
            dims=['time', 'lat', 'lon'],
            coords={
                'time': valid_time,
                'lat': lat_lon[0],
                'lon': lat_lon[1]
            },
            name=value
        )
        preds_ds.append(pred_array)
    return xr.merge(preds_ds)
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
