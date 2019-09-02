import numpy as np
import math

class DataLoader:

    """
        dataset: numpy array 
        bs : batch size
    """
    def __init__(self, x, y, bs, drop_last = False):
        self.x = x
        self.y = y if len(y.shape) == 2 else y.reshape((y.shape[0],1))
        self.bs = bs
        self.data_size = x.shape[0]
        self.num_batches = math.ceil( self.data_size / self.bs )

    def __iter__(self):
        self.batch_num = 0
        return self

    def _reshape_data(self, data):
        data = data.transpose()
        data_shape0 = data.shape[0]
        data_shape1 = data.shape[1] if len(data.shape) == 2 else 1
        return data.reshape((data_shape0,) + (1,data_shape1))

    def __next__(self):
        # all batches processed
        if self.batch_num == self.num_batches:
            raise StopIteration

        batch_start_index  = self.batch_num * self.bs
        batch_end_index = batch_start_index + self.bs
        batch_x = self.x[ batch_start_index : batch_end_index , :]
        batch_y = self.y[ batch_start_index : batch_end_index , :]

        batch_x = self._reshape_data(batch_x)
        batch_y = self._reshape_data(batch_y)

        self.batch_num += 1
        return batch_x, batch_y


