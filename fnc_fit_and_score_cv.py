def fnc_fit_and_score_cv(data_slice, n_cvs, n_classes, labs, grid):
    """
    Name: Holly Kular\
    Date: 08-08-2024\
    Email: hkular@ucsd.edu\
    fnc_fit_and_score_cv.py\
    Description: Script for decoding fitting linear SVM and scoring accuracy across CV folds
    Fits the model on each CV fold for a given time step

    Args:
      data_slice: The data slice for the current time step.
      n_cvs: Number of CV folds.
      labs: Labels for all trials.
      grid: scikit-learn GridSearchCV object.

    Returns:
      A list of accuracies for each CV fold for the given time step.
    """
    from numpy import setdiff1d, zeros, mean
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_predict
      
    # predict and score!
    y_pred = cross_val_predict(grid, data_slice, labs, cv = n_cvs)
       
    cm = confusion_matrix(labs, y_pred, normalize = "true").diagonal()
    
    return cm