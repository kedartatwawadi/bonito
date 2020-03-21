#!/usr/bin/env python3

"""
Bonito training.
"""

import os
import csv
from datetime import datetime
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

from bonito.model import Model
from bonito.util import load_data, init
from bonito.training_multi_gpu import ChunkDataSet, train, test
from collections import OrderedDict

import toml
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

try: from apex import amp
except ImportError: pass

def main(args):

    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists." % workdir)
        exit(1)

    init(args.seed, args.device)
    device = torch.device(args.device)

    print("[loading data]")
    chunks, chunk_lengths, targets, target_lengths = load_data(limit=args.chunks, shuffle=True, directory=args.directory)

    if args.test_directory:
        test_chunks, test_chunk_lengths, test_targets, test_target_lengths = load_data(limit=args.test_chunks, shuffle=True, directory=args.test_directory)
        train_dataset = ChunkDataSet(chunks, chunk_lengths, targets, target_lengths)
        test_dataset = ChunkDataSet(test_chunks, test_chunk_lengths, test_targets, test_target_lengths)
    else:
        split = np.floor(chunks.shape[0] * args.validation_split).astype(np.int32)
        train_dataset = ChunkDataSet(chunks[:split], chunk_lengths[:split], targets[:split], target_lengths[:split])
        test_dataset = ChunkDataSet(chunks[split:], chunk_lengths[split:], targets[split:], target_lengths[split:])
                


    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=4, pin_memory=True)

    config = toml.load(args.config)
    argsdict = dict(training=vars(args))

    print("[loading model]")
    model = Model(config)

    weights = os.path.join(workdir, 'weights.tar')

    
    if os.path.exists(weights):
        state_dict = torch.load(weights)
 
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
   
        model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model) #enabling data parallelism
    
    model.to(device)
    model.train()

    os.makedirs(workdir, exist_ok=True)
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    optimizer = AdamW(model.parameters(), amsgrad=True, lr=args.lr)

    if args.amp:
        try:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        except NameError:
            print("[error]: Cannot use AMP: Apex package needs to be installed manually, See https://github.com/NVIDIA/apex")
            exit(1)

    schedular = CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    # initialize duration train_loss
    train_loss = '-1'
    duration = '-1'


    for epoch in range(0, args.epochs + 1):


        # Run val
        try:
            val_loss, val_mean, val_median = test(model, device, test_loader)
        
        except KeyboardInterrupt:
            break

        print("[epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
            epoch, workdir, val_loss, val_mean, val_median
        ))

        torch.save(model.state_dict(), os.path.join(workdir, "weights_%s.tar" % epoch))
        with open(os.path.join(workdir, 'training.csv'), 'a', newline='') as csvfile:
            csvw = csv.writer(csvfile, delimiter=',')
            if epoch == 1:
                csvw.writerow([
                    'time', 'duration', 'epoch', 'train_loss',
                    'validation_loss', 'validation_mean', 'validation_median'
                ])
            csvw.writerow([
                datetime.today(), int(duration), epoch,
                train_loss, val_loss, val_mean, val_median,
            ])

        try:
            train_loss, duration = train(
                model, device, train_loader, optimizer, use_amp=args.amp
            )
        except KeyboardInterrupt:
            break

        schedular.step()


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    parser.add_argument("config")
    parser.add_argument("--directory", default=None)
    parser.add_argument("--test_directory", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=1000000, type=int)
    parser.add_argument("--test_chunks", default=10000, type=int)
    parser.add_argument("--validation_split", default=0.99, type=float)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    return parser
