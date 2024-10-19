def fnc_fit_and_score_garbage(data_slice, tri_ind, hold_out, n_cvs, n_classes, labs, label, thresh, grid, seeds):
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
    from numpy.random import choice, seed, shuffle
    from numpy import setdiff1d, zeros, mean, select
    from sklearn.metrics import confusion_matrix
    seed(seeds)  # Initialize the random number generator with the given seed
    acc = zeros(n_cvs)
    cm = zeros((n_cvs, n_classes))
    for i in range(n_cvs):
        # loop over CV folds within each t_step
        # trials to hold out as test set on this cv fold
        tst_ind = tri_ind[ i*hold_out : (i+1)*hold_out ]
        
        # index into the training data on this cv fold
        trn_ind = choice(setdiff1d( tri_ind, tst_ind ), size = len(setdiff1d( tri_ind, tst_ind )), replace = True)

        # get the training data (X) and the training labels (y)
        X = data_slice[trn_ind,:]
        # shuffle labels for garbage in garbage out
        shuffle(labs)
        if label == 'stim':
            y = labs[trn_ind]
        else:
            y = select([labs[trn_ind] >= thresh[1], labs[trn_ind] <= thresh[0]], [0,1], default=0)

        # fit the model
        grid.fit( X,y )

        # get the test data (X) and the test labels (y)
        X_test = data_slice[tst_ind, :]
        if label == 'stim':
            y_test = labs[tst_ind]
        else:
            y_test = select([labs[tst_ind] >= thresh[1], labs[tst_ind] <= thresh[0]], [0,1], default=0)

        # predict and score!
        y_pred = grid.predict(X_test)
        #acc[i] += grid.score( X_test,y_test )
      

        
        cm[i] = confusion_matrix(y_test, y_pred, normalize = "true").diagonal()
    acc = mean(cm, axis = 0)
    
    return acc
