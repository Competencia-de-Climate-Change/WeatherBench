import numpy as np
import xarray as xr

def create_training_data(da, lead_time_h, return_valid_time=False):
    """Function to split input and output by lead time."""
    X = da.isel(time=slice(0, -lead_time_h))
    y = da.isel(time=slice(lead_time_h, None))
    valid_time = y.time

    # Must be a "flat" array ==> nlat*nlon
    data = X.values.reshape(-1, nlat*nlon)
    obj = y.values.reshape(-1, nlat*nlon)
    
    if return_valid_time:
        return data, obj, valid_time
    
    return data, obj

def train_regression(model, lead_time_h, input_vars, output_vars, data_subsample=1):
    """Create data, train a linear regression and return the predictions."""
    X_train, y_train, X_test, y_test = [], [], [], []
    for v in input_vars:
        # Create Training Data for a 3 day prediction
        X, y = create_training_data(data_train[v], lead_time_h)
        X_train.append(X)

        if v in output_vars: 
            y_train.append(y)

        # Create Test Data for a 3 day prediction
        X, y, valid_time = create_training_data(data_test[v], lead_time_h, return_valid_time=True)
        X_test.append(X)
        
        if v in output_vars: 
            y_test.append(y)

    X_train, y_train, X_test, y_test = [np.concatenate(d, 1) for d in [X_train, y_train, X_test, y_test]]
    if data_subsample > 1:
        X_train = X_train[::data_subsample]
        y_train = y_train[::data_subsample]
    
    lr = model(n_jobs=16)
    lr.fit(X_train, y_train)
    
    mse_train = mean_squared_error(y_train, lr.predict(X_train))
    mse_test = mean_squared_error(y_test, lr.predict(X_test))

    print(f'Train MSE = {mse_train}')
    print(f'Test MSE = {mse_test}')

    preds = lr.predict(X_test).reshape((-1, len(output_vars), nlat, nlon))
    
    # Save predictions + unnormalize
    preds_ds = []
    for i, v in enumerate(output_vars):
        pred_array = xr.DataArray(
            preds[:, i] * data_std[v].values + data_mean[v].values, 
            dims=['time', 'lat', 'lon'],
            coords={
                'time': valid_time,
                'lat': data_train.lat,
                'lon': data_train.lon
            },
            name=v
        )
        preds_ds.append(pred_array)

    return xr.merge(preds_ds), lr

def create_iterative_lr(state, model, lead_time_h=6, max_lead_time_h=5*24):
    
    max_fc_steps = max_lead_time_h // lead_time_h

    preds_z500, preds_t850 = [], []

    for fc_step in tqdm(range(max_fc_steps)):
        # predict next state and update current with next
        state = model.predict(state)
        
        # Unnormalize
        fc_z500 = state[:, :nlat*nlon].copy() * data_std.z.values + data_mean.z.values
        fc_t850 = state[:, nlat*nlon:].copy() * data_std.t.values + data_mean.t.values
        
        # Reshape
        fc_z500 = fc_z500.reshape((-1, nlat, nlon))
        fc_t850 = fc_t850.reshape((-1, nlat, nlon))
        
        preds_z500.append(fc_z500)
        preds_t850.append(fc_t850)

    return [xr.DataArray(
        np.array(fcs), 
        dims=['lead_time', 'time', 'lat', 'lon'],
        coords={
            'lead_time': np.arange(lead_time_h, max_lead_time_h + lead_time_h, lead_time_h),
            'time': z_test.time,
            'lat': z_test.lat,
            'lon': z_test.lon
        }
    ) for fcs in [preds_z500, preds_t850]]