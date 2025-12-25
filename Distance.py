import math
import statistics

# Hard-coded dataset of 125 (token, loss) pairs. Can be substituted for any dataset.
data = [
(12, 5.523),
(13, 5.799),
(12, 5.54),
(12, 5.879),
(13, 5.381),
(14, 5.481),
(14, 5.738),
(13, 5.793),
(12, 6.024),
(14, 5.53),
(14, 5.905),
(14, 5.546),
(13, 5.682),
(13, 6.021),
(13, 5.9),
(13, 5.804),
(14, 5.834),
(14, 5.769),
(14, 6.074),
(14, 5.892),
(13, 5.706),
(14, 5.781),
(14, 5.543),
(14, 5.859),
(14, 5.706),
(11, 5.819),
(13, 5.91),
(11, 5.528),
(12, 5.588),
(13, 5.848),
(11, 6.093),
(11, 5.918),
(14, 5.552),
(12, 5.866),
(14, 5.622),
(14, 5.499),
(13, 5.733),
(12, 5.975),
(14, 5.563),
(12, 6.264),
(14, 5.475),
(13, 6.121),
(12, 6.036),
(14, 5.632),
(12, 5.827),
(13, 5.831),
(14, 5.52),
(12, 5.909),
(14, 5.743),
(14, 5.781),
(13, 5.525),
(13, 5.836),
(12, 5.415),
(12, 5.912),
(13, 5.857),
(12, 6.125),
(12, 5.778),
(11, 6.093),
(14, 5.616),
(13, 5.98),
(12, 5.802),
(13, 5.785),
(13, 5.458),
(12, 5.94),
(13, 5.912),
(15, 5.399),
(13, 6.037),
(14, 5.86),
(13, 5.666),
(12, 5.883),
(13, 5.963),
(11, 6.19),
(13, 5.889),
(13, 5.387),
(13, 6.0),
(13, 5.79),
(13, 5.77),
(13, 5.965),
(12, 5.759),
(14, 5.741),
(14, 5.727),
(15, 5.783),
(13, 5.755),
(15, 5.44),
(12, 5.961),
(14, 6.121),
(13, 5.973),
(13, 5.96),
(13, 5.631),
(13, 5.897),
(13, 5.777),
(14, 5.5),
(12, 6.166),
(14, 5.955),
(12, 5.968),
(14, 5.755),
(13, 5.856),
(14, 5.977),
(13, 5.804),
(14, 5.968),
(14, 5.963),
(13, 6.185),
(14, 5.861),
(14, 6.082),
(15, 5.919),
(13, 5.881),
(16, 5.728),
(13, 5.953),
(16, 5.847),
(13, 6.14),
(14, 5.868),
(14, 6.305),
(16, 5.478),
(14, 6.02),
(15, 6.013),
(13, 6.429),
(15, 5.868),
(12, 6.188),
(15, 5.892),
(15, 6.389),
(14, 6.048),
(14, 5.941),
(13, 5.944),
(14, 5.953),
(17, 5.576),
]

# Given sample means and sample standard deviations, pre-calculated in a spreadsheet (used for z-score calculations)
# Can be substituted any values from your dataset
token_mean = 13.28
token_sd = 1.111523221
loss_mean = 5.833128
loss_sd = 0.217117516

# Computer calculation to recompute/verify sample means and sample standard deviations
computed_token_mean = statistics.mean([t for t, _ in data])
computed_token_sd = statistics.stdev([t for t, _ in data])
computed_loss_mean = statistics.mean([l for _, l in data])
computed_loss_sd = statistics.stdev([l for _, l in data])

# Print the hard-coded statistics
print('Given statistics:')
print(f'token_mean (given): {token_mean}')
print(f'token_sd (given): {token_sd}')
print(f'loss_mean (given): {loss_mean}')
print(f'loss_sd (given): {loss_sd}')

# Print the recomputed (verification) statistics from the hard-coded data
print('\nComputed statistics from data:')
print(f'token_mean (computed): {computed_token_mean}')
print(f'token_sd (computed): {computed_token_sd}')
print(f'loss_mean (computed): {computed_loss_mean}')
print(f'loss_sd (computed): {computed_loss_sd}')

# Storage for z-scores and distances
results = []

# Compute z-scores and L2 distance for each row
for idx, (token, loss) in enumerate(data, start=1):
    # Standardized values calculation
    z_token = (token - token_mean) / token_sd
    z_loss = (loss - loss_mean) / loss_sd

    # Euclidean distance calculation in standardized space
    distance = math.sqrt(z_token ** 2 + (z_loss ** 2))

    # Store result entry
    results.append({
    'index': idx,
    'token': token,
    'loss': loss,
    'z_token': z_token,
    'z_loss': z_loss,
    'distance': distance,
    })

# Print table format
print('\nTable of z-scores and distances:')
print(f"{'idx':>3} {'token':>5} {'loss':>10} {'z_token':>12} {'z_loss':>12} {'distance':>12}")

# Print formatted results
for row in results:
    print(f"{row['index']:>3} {row['token']:>5} {row['loss']:>10.6f} {row['z_token']:>12.6f} {row['z_loss']:>12.6f} {row['distance']:>12.6f}")