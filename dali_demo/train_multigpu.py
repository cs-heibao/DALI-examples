import os
import time
import datetime
import torch
import argparse
from src.data import *
import warnings
import torch.distributed as dist
from src.defined_external_iterator import ExternalInputIterator
from src.defined_external_source import ExternalSourcePipeline
from src.COCOIterator import DALICOCOIterator
# from src.loss_function import Loss
from src.model import model, Loss
from src.utils_time import *
try:
    from apex.parallel.LARC import LARC
    from apex import amp
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4000, help="number of epochs")
    parser.add_argument("--start_epochs", type=int, default=0, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="size of each image batch")
    # prepare for dali module
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='path to test and training data files')
    parser.add_argument('--gpus', type=int, default=4,
                        help='number of the gpus')
    parser.add_argument('--data_pipeline', type=str, default='no_dali', choices=['dali', 'no_dali'],
                        help='data preprocessing pipline to use')
    parser.add_argument('--fp16-mode', type=str, default='off', choices=['off', 'static', 'amp'],
                        help='Half precission mode to use')
    opt = parser.parse_args()

    if opt.fp16_mode != 'off':
        opt.fp16 = True
        opt.amp = (opt.fp16_mode == 'amp')
    else:
        opt.fp16 = False
        opt.amp = False
    if opt.amp:
        amp_handle = amp.init(enabled=opt.fp16)
    model = model(opt)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    if opt.fp16:
        print("INFO: Use Fp16")
        if opt.amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
            # optimizer = amp_handle.wrap_optimizer(optimizer)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.)
    # Prepare dataset
    print("INFO: Prepare Datasets")
    if opt.data_pipeline=='dali':
        eii = ExternalInputIterator(opt.batch_size, opt.img_path, opt.annotation_path)
        train_pipe = [ExternalSourcePipeline(opt.batch_size, num_threads=opt.n_cpu,
                                             device_id=device_id, eii=eii) for device_id in range(opt.gpus)]

        train_loader = DALICOCOIterator(train_pipe, len(eii.img_files))
    # else:
        # # Get dataloader
        # dataset = *****
        # train_loader = torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=opt.batch_size,
        #     shuffle=True,
        #     num_workers=opt.n_cpu,
        #     pin_memory=True,
        #     collate_fn=dataset.collate_fn
        # )

    tmp_lr = opt.lr
    all_time = 0
    for epoch in range(opt.start_epochs, opt.epochs):
        start = time.time()

        model.train()
        for batch_i, datas in enumerate(train_loader):
        # for batch_i, (imgNames, imgs, targets) in enumerate(train_loader):

            for data in datas:
                imgs = data[0][0].cuda()
                targets = data[1][0].cuda()
                label_id = data[2][0].cuda()
                targets = torch.cat([label_id, targets], dim=1)

                _, outputs = model(imgs)
                loss = Loss(outputs, targets)

                optimizer.zero_grad()

                if opt.fp16:
                    if opt.amp:
                        with amp.scale_loss(loss, optimizer) as scale_loss:
                            scale_loss.backward()
                    else:
                        # optimizer.backward(loss)
                        loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                progress_bar(batch_i, int(train_loader._size / (opt.gpus * train_loader.batch_size)))

        train_loader.reset()
        print("Epoch time: {}".format(time.time()-start))

