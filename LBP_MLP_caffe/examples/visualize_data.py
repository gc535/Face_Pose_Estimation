import numpy as np
import lmdb
import caffe


env = lmdb.open('LBP_MLP_test_data_lmdb', readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=float)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        print(key, value)


"""
lmdb_env = lmdb.open('LBP_MLP_test_data_lmdb', readonly=True)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    print('entered')
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    for l, d in zip(label, data):
            print (l, d)

"""