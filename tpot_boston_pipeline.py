import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = GradientBoostingRegressor(alpha=0.95, learning_rate=1.0, loss="lad", max_depth=5, max_features=0.45, min_samples_leaf=8, subsample=0.35000000000000003)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
