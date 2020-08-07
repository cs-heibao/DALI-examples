import numpy as np
from numpy.random import shuffle
class ExternalInputIterator(object):
    def __init__(self, batch_size, images_path,annotation_path):
        self.images_dir = images_path
        self.annotations_dir = annotation_path
        self.batch_size = batch_size
        with open(self.images_dir, 'r') as f:
            self.img_files = [line.rstrip() for line in f if line is not '']
        shuffle(self.img_files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.img_files)
        return self

    def __next__(self):
        batch = []
        bboxes = []
        id_label = []
        for index in range(self.batch_size):
            f = open(self.img_files[self.i], 'rb')
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            with open(self.annotations_dir + self.img_files[self.i].split('/')[-1].replace('jpg', 'txt')) as fd:
                info = [[line.strip().split(' ')[0], line.strip().split(' ')[1], line.split(' ')[2],
                         line.strip().split(' ')[3], line.strip().split(' ')[4]] for line in fd.readlines()]
                label = [[index, int(line[0])] for line in info]
                boxes = [[float(line[1]), float(line[2]), float(line[3]), float(line[4])] for line in info]
            id_label.append(np.array(label, dtype=np.float32))
            bboxes.append(np.array(boxes, dtype=np.float32))
            self.i = (self.i + 1) % self.n
        return (batch, id_label, bboxes)
    @property
    def size(self):
        return len(self.img_files)

    next = __next__
