def fnc_decode_L1(RNN_params, D_params, data_file, task_info):
    '''
Name: Holly Kular
Date: 03-19-2024
Email: hkular@ucsd.edu
fnc_decode_L1.py
Description: Script for decoding analysis on layer 1 of probabilistic RNN

    INPUT
    RNN_params (dict)
       - prob_split: what was the split of the dominant stim vs all (default '70_30')
       - afc: number of alternatives (2 or 6) (default 2)
       - coh: what was the coherence of RDK (default 'hi')
       - feedback: interlayer feedback (true or false) (default false)
       - thresh: the threshold to label RNN output (default [.3, .7])

    D_params (dict)
       - time_avg: whether we want decoding averaged over time or not (true or false) (default false)
       - t_win: what time window to average over if time_avg = true (default [200,-1])
       - n_cvs: cross validation folds (default 5)
       - num_cgs: penalties to eval (default 30)

    data_file: file path for Trials that we want to load
    '''
    # set up defaults
    prob_split = RNN_params.get('prb_split', '70_30')
    afc = RNN_params.get('afc', 2)
    coh = RNN_params.get('coh', 'hi')
    feedback = RNN_params.get('feedback', False)
    thresh = RNN_params.get('thresh', [.3, .7])
    time_avg = D_params.get('time_avg', False)
    t_win = D_params.get('t_win', [200, -1])
    label = D_params.get('label', 'stim')
    n_cvs = D_params.get('n_cvs', 5)
    num_cgs = D_params.get('num_cgs', 30)
    # penalties to eval
    Cs = np.logspace( -5,1, num_cgs )
    
    # store the accuracy
    acc = np.full( ( n_cvs ), np.nan )

    # set up the grid
    param_grid = { 'C': Cs, 'kernel': ['linear'] }

    # define object - use a SVC that balances class weights (because they are biased, e.g. 70/30)
    # note that can also specify cv folds here, but I'm doing it by hand below in a loop
    grid = GridSearchCV( SVC(class_weight = 'balanced'),param_grid,refit=True,verbose=0 )
    
    # load data
    data = np.load(data_file)
    
    # set-up vars for decoding   
    data_d = data['fr1']# layer 1 firing rate [trial x time step x unit] matrix
    labs = data['labs'].squeeze()

    # get some info about structure of the data
    tris = data_d.shape[0]             # number of trials
    tri_ind = np.arange(0,tris)      # list from 0...tris
    hold_out = int( tris / n_cvs )   # how many trials to hold out
    
    if time_avg: # if we are doing average over time there's no need to run in parallel
        
        data_d = np.mean( data_d[ :,t_win[0]:t_win[1],: ], axis = 1 ) # average over time window
        # Within each cross-validation fold
        for i in range(n_cvs):

            # trials to hold out as test set on this cv fold
            tst_ind = tri_ind[ i*hold_out : (i+1)*hold_out ]

            # index into the training data on this cv fold
            trn_ind = np.setdiff1d( tri_ind, tst_ind )

            # get the training data (X) and the training labels (y)
            # note that y is unbalanced unless prob is 50/50
            X = data_d[ trn_ind,: ]
            y = labs[trn_ind]

            # Fit the model on the binary labels
            grid.fit( X, y )

            # get the test data (X) and the test labels (y)
            X_test = data_d[tst_ind, :]
            y_test = labs[tst_ind]

            # predict!
            acc[ i ] = grid.score( X_test,y_test )

        # Evaluate accuracy
    return decoding_acc = np.mean( acc ) 
        
    else: # if we are decoding each time step, run that in parallel
        
        if __name__ == "__main__":

            with Pool(processes=round(os.cpu_count()*.7)):  # use 70% of cpus
                results = pool.starmap(fnc_fit_and_score, [
                    (t_step, data_d[:, t_step, :], tri_ind, hold_out, n_cvs, labs, label, thresh, grid)
                    for t_step in range(task_info['trial_dur'])
                ], chunksize = 10)

            # Process the results from each worker process (list of lists of accuracies)
        return decoding_acc = np.mean(np.array(results), axis=1)  # Calculate mean accuracy for each time step
            
