from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, eii):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.input = ops.ExternalSource()
        self.id_label = ops.ExternalSource()
        self.boxes = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device="gpu", resize_x=256, resize_y=256)
        self.twist = ops.ColorTwist(device="gpu")
        self.normalize = ops.CropMirrorNormalize(device="gpu", crop=(256, 256),
                                                 mean=[0.0, 0.0, 0.0],
                                                 std=[255.0, 255.0, 255.0],
                                                 mirror=0,
                                                 output_dtype=types.FLOAT,
                                                 output_layout=types.NCHW,
                                                 image_type=types.RGB,)
        # Random variables
        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.5, 0.5])

        self.external_data = eii
        self.iterator = iter(self.external_data)

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        self.jpegs = self.input()
        self.labels = self.id_label()
        self.bboxes = self.boxes()
        # jpegs, boxes, labels = self.source(name="Reader")
        images = self.decode(self.jpegs)
        images = self.resize(images)
        images = self.twist(images.gpu(), saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)
        images = self.normalize(images)
        return (images, self.bboxes.gpu(), self.labels.gpu())

    def iter_setup(self):
        try:
            (images, labels, bboxes) = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
            self.feed_input(self.bboxes, bboxes)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration