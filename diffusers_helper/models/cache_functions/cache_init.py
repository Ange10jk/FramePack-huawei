def cache_init(num_steps, model_kwargs=None):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}


    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    for j in range(20):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}
 

    for j in range(40):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
  

    cache_dic['cache'] = cache
    cache_dic['fresh_threshold'] = 5
    cache_dic['max_order'] = 0
    cache_dic['first_enhance'] = 1

    current = {}
    current['step']=0
    current['num_steps'] = num_steps
    current['activated_steps'] = [0]

    return cache_dic, current


def cache_init_terc(num_steps, max_order=0):
    """Initialize the first-section residual cache; default to order zero."""
    # HGC passes cache[-1] directly to the predictor.
    cache_dic = {'cache': {-1: {}}, 'max_order': max_order}

    current = {
        'step': 0,  # Synced by the model before the HGC counter advances.
        'num_steps': num_steps,
        'activated_steps': [0],
    }

    return cache_dic, current
