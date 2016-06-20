def get_contrasts(taskname):
    contrasts={}
    contrasts['stopsignal']=[['go>baseline','T',['go'],[1]],
                         ['succstop>go','T',['succ_stop','go'],[1,-1]],
                         ['unsucc>succstop','T',['unsucc_stop','succ_stop'],[1,-1]]]
    contrasts['emotionalregulation']=[['average>baseline','T',['attend_neutral','attend_negative','suppress_negative'],[1,1,1]],
                         ['attendneg>attendneutral','T',['attend_neutral','attend_negative'],[-1,1]],
                         ['suppressneg>attendneg','T',['suppress_negative','attend_negative'],[1,-1]],
                          ['rate>baseline','T',['rate'],[1]]]

    return(contrasts[taskname])
