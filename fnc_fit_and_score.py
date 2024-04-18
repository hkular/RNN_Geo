def fnc_fit_and_score(t_step, data_slice, tri_ind, hold_out, n_cvs, labs, label, thresh, grid):
    """
    Name: Holly Kular\
    Date: 03-19-2024\
    Email: hkular@ucsd.edu\
    fnc_fit_and_score.py\
    Description: Script for decoding fitting linear SVM and scoring accuracy across CV folds
    Fits the model on each CV fold for a given time step

    Args:
      t_step: The time step index.
      data_slice: The data slice for the current time step.
      tri_ind: Indices of all trials.
      hold_out: Number of trials to hold out for testing in each fold.
      n_cvs: Number of CV folds.
      labs: Labels for all trials.
      label: Decoding based on stim or choice
      thresh: Thresholds for binary classification.
      grid: scikit-learn GridSearchCV object.

    Returns:
      A list of accuracies for each CV fold for the given time step.
    """
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.svm import SVC  
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import make_classification
    acc = []
    for i in range(n_cvs):
        # loop over CV folds within each t_step
        # trials to hold out as test set on this cv fold
        tst_ind = tri_ind[ i*hold_out : (i+1)*hold_out ]

        # index into the training data on this cv fold
        trn_ind = np.setdiff1d( tri_ind, tst_ind )

        # get the training data (X) and the training labels (y)
        X = data_slice[trn_ind,:]
        if label == 'stim':
            y = labs[trn_ind]
        else:
            y = np.select([labs[trn_ind] >= thresh[1], labs[trn_ind] <= thresh[0]], [0,1], default=0)

        # fit the model
        grid.fit( X,y )

        # get the test data (X) and the test labels (y)
        X_test = data_slice[tst_ind, :]
        if label == 'stim':
            y_test = labs[tst_ind]
        else:
            y_test = np.select([labs[tst_ind] >= thresh[1], labs[tst_ind] <= thresh[0]], [0,1], default=0)

        # predict!
        score = grid.score( X_test,y_test )
        acc.append(score)  # Append accuracy for this CV fold
    return acc
