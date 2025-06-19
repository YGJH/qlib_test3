with open('.qlib/qlib_data/us_data/instruments/all.txt', 'r+') as f:
    metadata = f.read()
    new_metadata = []
    for i in metadata.split('\n'):
        if '2025' in i:
            new_metadata.append(i)
    f.seek(0)
    f.truncate()
    f.write('\n'.join(new_metadata))




