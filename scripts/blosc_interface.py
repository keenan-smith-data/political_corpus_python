import blosc
import pickle

def blosc_pickle(dat, filename):
    pickled = pickle.dumps(dat)
    compressed = blosc.compress(pickled)
    with open(filename, "wb") as f:
        f.write(compressed)

def blosc_read(filename):
    with open(filename, "rb") as f:
        compressed_pickle = f.read()

    pickled = blosc.decompress(compressed_pickle)
    return pickle.loads(pickled)