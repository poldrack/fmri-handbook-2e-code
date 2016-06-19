# -*- coding: utf-8 -*-
"""
load BIDS events file and generate FSL 3-column onsets
"""

import pandas,numpy
import os



def bids_to_3col(eventsfile,outdir,condition_columns,
    parametric_columns=[],parametric_columns_limiter=[],
   parametric_demean=True,parametric_std=False):

    events=pandas.read_csv(eventsfile,sep='\t')
    onsets={}
    # make sure named conditions and onset/durations are there
    assert len(list(set(condition_columns) - set(events.columns)))==0
    assert len(list(set(parametric_columns) - set(events.columns)))==0
    assert 'onset' in events.keys()
    assert 'duration' in events.keys()
    try:
        assert os.path.exists(outdir)
    except AssertionError:
        try:
            print('outdir does not exist, creating it...')
            os.makedirs(outdir)
        except:
            raise OSError('could not make %s'%outdir)
    # find unique values for each condition
    condvals={}
    for cond in condition_columns:
        condvals[cond]=events[cond].unique()

    if len(condition_columns)==1:
        # single condition
        cond=condition_columns[0]
        for cv in condvals[cond]:
            condevents=events[events[cond]==cv]
            onsets[cv]=numpy.vstack((condevents.onset,condevents.duration,
                numpy.ones(len(condevents.onset)))).T
    else:
        # factorial combination of multiple condition columns
        groups=events.groupby(condition_columns)
        for g in groups:
            condlist=list(g[0])
            if 'n/a' in condlist:
                condlist.remove('n/a')
            condname='_'.join(condlist)
            print(condname)
            onsets[condname]=numpy.vstack((g[1].onset,g[1].duration,
                numpy.ones(len(g[1].onset)))).T
    # parametric regressors
    # right now we don't allow combinations of regressors, just single
    for pr in parametric_columns:
            condevents=events.copy()
            # throw out any event that don't have a legit value
            condevents=condevents[condevents[pr]!='n/a']
            condevents[pr]=[float(i) for i in condevents[pr]]
            if parametric_demean:
                condevents[pr]=condevents[pr] - condevents[pr].mean()
            if parametric_std:
                condevents[pr]=condevents[pr]/condevents[pr].std()
            condname=pr+'_param'
            onsets[condname]=numpy.vstack((condevents.onset,condevents.duration,
                condevents[pr])).T


    for cond in onsets.keys():
        print('saving onsets for %s'%cond)
        fname=os.path.join(outdir,'%s_ons.txt'%cond)
        numpy.savetxt(fname,onsets[cond],delimiter='\t')
    return(onsets)
