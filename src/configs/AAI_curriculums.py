# use threshold > 100 if not training until threshold reward

# 2 Bit XOR
cur_1 = {
    'name' : 'cur_1',
    'folder' : 'src/configs/AAI_tasks/2_bit_XOR/',
    'thresholds' : [
        200
    ],
    'train_files' : [
        '2_bit_XOR-all.yml'
    ],
    'test_files' : [
        '2_bit_XOR-0.yml',
        '2_bit_XOR-1.yml',
        '2_bit_XOR-2.yml',
        '2_bit_XOR-3.yml'
    ],
    'steps_per_config' : 1000000
}

# 2 Bit COPY
cur_2 = { 
    'name' : 'cur_2',
    'folder' : 'src/configs/AAI_tasks/2_bit_COPY/',
    'thresholds' : [
        200
    ],
    'train_files' : [
        '2_bit_COPY-all.yml',
    ],
    'test_files' : [
        '2_bit_COPY-0.yml',
        '2_bit_COPY-1.yml',
        '2_bit_COPY-2.yml',
        '2_bit_COPY-3.yml',
    ],
    'steps_per_config' : 1000000
}

# 3 Bit XOR
cur_3 = { 
    'name' : 'cur_3',
    'folder' : 'src/configs/AAI_tasks/3_bit_XOR/',
    'thresholds' : [
        200
    ],
    'train_files' : [
        '3_bit_XOR-all.yml',
    ],
    'test_files' : [
        '3_bit_XOR-0.yml',
        '3_bit_XOR-1.yml',
        '3_bit_XOR-2.yml',
        '3_bit_XOR-3.yml',
        '3_bit_XOR-4.yml',
        '3_bit_XOR-5.yml',
        '3_bit_XOR-6.yml',
        '3_bit_XOR-7.yml'
    ],
    'steps_per_config' : 6000000
}

# Distance=10 XOR
cur_4 = { 
    'name' : 'cur_4',
    'folder' : 'src/configs/AAI_tasks/2_bit_XOR_L10/',
    'thresholds' : [
        200
    ],
    'train_files' : [
        '2_bit_XOR_L10-all.yml',
    ],
    'test_files' : [
        '2_bit_XOR_L10-0.yml',
        '2_bit_XOR_L10-1.yml',
        '2_bit_XOR_L10-2.yml',
        '2_bit_XOR_L10-3.yml',
    ],
    'steps_per_config' : 6000000
}

# 2 Bit to 3 Bit XOR curriculum
cur_5 = { 
    'name' : 'cur_5',
    'folder' : 'src/configs/AAI_tasks/',
    'thresholds' : [
        3.96,
        200
    ],
    'train_files' : [
        '2_bit_XOR/2_bit_XOR-all.yml',
        '3_bit_XOR/3_bit_XOR-all.yml'
    ],
    'test_files' : [
        '2_bit_XOR/2_bit_XOR-0.yml',
        '2_bit_XOR/2_bit_XOR-1.yml',
        '2_bit_XOR/2_bit_XOR-2.yml',
        '2_bit_XOR/2_bit_XOR-3.yml',
        '3_bit_XOR/3_bit_XOR-0.yml',
        '3_bit_XOR/3_bit_XOR-1.yml',
        '3_bit_XOR/3_bit_XOR-2.yml',
        '3_bit_XOR/3_bit_XOR-3.yml',
        '3_bit_XOR/3_bit_XOR-4.yml',
        '3_bit_XOR/3_bit_XOR-5.yml',
        '3_bit_XOR/3_bit_XOR-6.yml',
        '3_bit_XOR/3_bit_XOR-7.yml'
    ],
    'steps_per_config' : 6000000
}

# Increasing distance XOR curriculum
cur_6 = { 
    'name' : 'cur_6',
    'folder' : 'src/configs/AAI_tasks/',
    'thresholds' : [
        3.96,
        3.93,
        3.9,
        3.88
    ],
    'train_files' : [
        '2_bit_XOR/2_bit_XOR-all.yml',
        '2_bit_XOR_L10/2_bit_XOR_L10-all.yml',
        '2_bit_XOR_L20/2_bit_XOR_L20-all.yml',
        '2_bit_XOR_L30/2_bit_XOR_L30-all.yml'
    ],
    'test_files' : [
        '2_bit_XOR/2_bit_XOR-0.yml',
        '2_bit_XOR/2_bit_XOR-1.yml',
        '2_bit_XOR/2_bit_XOR-2.yml',
        '2_bit_XOR/2_bit_XOR-3.yml',
        '2_bit_XOR_L10/2_bit_XOR_L10-0.yml',
        '2_bit_XOR_L10/2_bit_XOR_L10-1.yml',
        '2_bit_XOR_L10/2_bit_XOR_L10-2.yml',
        '2_bit_XOR_L10/2_bit_XOR_L10-3.yml',
        '2_bit_XOR_L20/2_bit_XOR_L20-0.yml',
        '2_bit_XOR_L20/2_bit_XOR_L20-1.yml',
        '2_bit_XOR_L20/2_bit_XOR_L20-2.yml',
        '2_bit_XOR_L20/2_bit_XOR_L20-3.yml',
        '2_bit_XOR_L30/2_bit_XOR_L30-0.yml',
        '2_bit_XOR_L30/2_bit_XOR_L30-1.yml',
        '2_bit_XOR_L30/2_bit_XOR_L30-2.yml',
        '2_bit_XOR_L30/2_bit_XOR_L30-3.yml'
    ],
    'steps_per_config' : 2000000
}