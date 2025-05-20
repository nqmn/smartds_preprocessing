# Smart Dataset Preprocessing (smartds_preprocessing)

`smartds_preprocessing` is a Python utility function for preprocessing feature datasets before feeding them into machine learning algorithms. It automatically removes problematic columns—such as non-numeric, constant, and infinite-value columns—and can optionally remove features that are constant within each class (if labels are provided).

`Please note that this function is not published yet in PyPi Phyton Package Index. Will do soon!`

## Features
- Handles both pandas DataFrames and NumPy arrays.
- Removes non-numeric columns for cleaner numerical analysis.
- Eliminates columns containing infinite (inf) values.
- Removes constant columns (with variance below a small threshold).
- Optionally removes features that are constant within each class (requires label vector y).
- Provides index mapping to original columns for easy interpretation.

## Installation

Copy the function into your Python project.
Requires numpy and pandas as dependencies.

```python
import numpy as np
import pandas as pd

# Paste the clean_dataset function here

```

## Usage

```python

# Example with a pandas DataFrame
X_clean, retained_indices = clean_dataset(X, y)

# Example with a NumPy array
X_clean, retained_indices = clean_dataset(X_np, y)
```

X: Feature matrix (pandas DataFrame or NumPy array)
y: (Optional) Array-like class labels, used to remove features constant within each class.
epsilon: (Optional) Small threshold for detecting constant columns (default: 1e-10).

### Returns:

X_clean: Cleaned feature matrix as a NumPy array.
retained_indices: Indices of the retained features relative to the original input.

## Example

```python

import numpy as np
import pandas as pd

# Sample data
X = pd.DataFrame({
    "a": [1, 1, 1, 1],         # Constant column
    "b": [2, 3, 2, 3],         # Numeric
    "c": ["cat", "dog", "cat", "dog"],  # Non-numeric
    "d": [np.inf, 1, 2, 3],    # Contains inf
})
y = [0, 0, 1, 1]

X_clean, retained_indices = clean_dataset(X, y)
print("Cleaned Data:\n", X_clean)
print("Retained Indices:", retained_indices)

```

## Output

The function prints a summary of what was removed:

```pgsql
Removed 1 non-numeric columns
Removed 1 columns containing infinity values
Removed 1 constant numeric features: ['a']
Removed 0 features constant within each class: []
Total features removed: 3
Retained 1 clean numeric features
```

## Function Signature
```python

def clean_dataset(X, y=None, epsilon=1e-10):
    ...
```

## Notes
- If X cannot be converted to a DataFrame, limited cleaning will be performed (removes only constant columns in the NumPy array).
- Works best with datasets where numeric feature columns are expected for machine learning models.
- Verbose: the function prints each cleaning step to help you audit the preprocessing.

## License
MIT License
## Author
nqmn
