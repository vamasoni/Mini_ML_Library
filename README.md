# Mini_ML_Library : FOML Assignment 3 

## Prereqs
- Python 3.8+
- numpy
- graphviz (optional, for visualization)

Put datasets in `data/`:
- `spambase.data` (UCI Spambase) — CSV, last col label
- `fashion-mnist_train.csv` — CSV with header (label,pixel0,...)

## Quick commands

1. Run logistic unit test:
```bash
python -c "from my_ml_lib.linear_models.classification._logistic import LogisticRegression; import numpy as np; X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([0,1,1,1]); clf=LogisticRegression(alpha=0.1,max_iter=200); clf.fit(X,y); print(clf.predict(X));"
