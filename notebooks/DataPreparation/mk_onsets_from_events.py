import pandas
import numpy
import os


def write_3col_file(outfile,onsets,durations,weights=[]):
    try:
        assert len(weights)==len(onsets)
    except:
        weights=numpy.ones(onsets.shape)

    try:
        assert len(durations)==len(onsets)
    except:
        durations=numpy.ones(onsets.shape)*durations
    data=numpy.vstack((onsets,durations,weights)).T
    numpy.savetxt(outfile,data,delimiter='\t')
    return(data)

def mk_onsets_from_events(evfile,onsdir):
    events=pandas.read_csv(evfile,sep='\t')
    junkevents=events.query('respnum==0')
    events=events.query('respnum>0')
    events.response_time = events.response_time - events.response_time.mean()

    # write 3 column files for each
    write_3col_file(os.path.join(onsdir,'junk_ons.txt'),
                    junkevents.onset,durations=3)
    
    write_3col_file(os.path.join(onsdir,'trial_ons.txt'),
                    events.onset,durations=3)

    write_3col_file(os.path.join(onsdir,'paramgain_ons.txt'),
                    events.onset,weights=events['parametric gain'],durations=3)
    
    write_3col_file(os.path.join(onsdir,'paramloss_ons.txt'),
                    events.onset,weights=events['parametric loss'],durations=3)
    
    write_3col_file(os.path.join(onsdir,'paramrt_ons.txt'),
                    events.onset,weights=events.response_time,durations=3)
