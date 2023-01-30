from neurogym.envs.collections import yang19

# DM1 to DM2
cur_1 = { 
    'name' : 'cur_1',
    'config_files' : [
        yang19.dm1(timing = {'fixation' : 300}), 
        yang19.dm2(timing = {'fixation' : 300}),
    ],
    'indicators': [
        0,
        1,
    ]
}

# DM2 to DM1
cur_1R = { 
    'name' : 'cur_1R',
    'config_files' : [
        yang19.dm2(timing = {'fixation' : 300}), 
        yang19.dm1(timing = {'fixation' : 300}),
    ],
    'indicators': [
        0,
        1,
    ]
}

# DM1 to CtxDM1
cur_2 = { 
    'name' : 'cur_2',
    'config_files' : [
        yang19.dm1(timing = {'fixation' : 300}), 
        yang19.ctxdm1(timing = {'fixation' : 300}), 
    ],
    'indicators': [
        0,
        1
    ]
}

# CtxDM1 to DM1
cur_2R = { 
    'name' : 'cur_2R',
    'config_files' : [
        yang19.ctxdm1(timing = {'fixation' : 300}), 
        yang19.dm1(timing = {'fixation' : 300}), 
    ],
    'indicators': [
        0,
        1
    ]
}

# DM1 to CtxDM2
cur_3 = { 
    'name' : 'cur_3',
    'config_files' : [
        yang19.dm1(timing = {'fixation' : 300}), 
        yang19.ctxdm2(timing = {'fixation' : 300}), 
    ],
    'indicators': [
        0,
        1
    ]
}

# CtxDM2 to DM1
cur_3R = { 
    'name' : 'cur_3R',
    'config_files' : [
        yang19.ctxdm2(timing = {'fixation' : 300}),
        yang19.dm1(timing = {'fixation' : 300}),
    ],
    'indicators': [
        0,
        1
    ]
}

# DM2 to CtxDM2
cur_4 = { 
    'name' : 'cur_4',
    'config_files' : [
        yang19.dm2(timing = {'fixation' : 300}), 
        yang19.ctxdm2(timing = {'fixation' : 300}), 
    ],
    'indicators': [
        0,
        1
    ]
}

# CtxDM2 to DM2
cur_4R = { 
    'name' : 'cur_4R',
    'config_files' : [
        yang19.ctxdm2(timing = {'fixation' : 300}),
        yang19.dm2(timing = {'fixation' : 300})
    ],
    'indicators': [
        0,
        1
    ]
}

# DM2 to CtxDM1
cur_5 = { 
    'name' : 'cur_5',
    'config_files' : [
        yang19.dm2(timing = {'fixation' : 300}),
        yang19.ctxdm1(timing = {'fixation' : 300}), 
    ],
    'indicators': [
        0,
        1
    ]
}

# CtxDM1 to DM2
cur_5R = { 
    'name' : 'cur_5R',
    'config_files' : [
        yang19.ctxdm1(timing = {'fixation' : 300}),
        yang19.dm2(timing = {'fixation' : 300}),
    ],
    'indicators': [
        0,
        1
    ]
}

# CtxDM1 to CtxDM2
cur_6 = { 
    'name' : 'cur_6',
    'config_files' : [
        yang19.ctxdm1(timing = {'fixation' : 300}), 
        yang19.ctxdm2(timing = {'fixation' : 300}), 
    ],
    'indicators': [
        0,
        1
    ]
}

# CtxDM2 to CtxDM1
cur_6R = { 
    'name' : 'cur_6R',
    'config_files' : [
        yang19.ctxdm2(timing = {'fixation' : 300}), 
        yang19.ctxdm1(timing = {'fixation' : 300}), 
    ],
    'indicators': [
        0,
        1
    ]
}