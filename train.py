import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
import argparse

import duett
import physionet

class WarmUpCallback(pl.callbacks.Callback):
    """Linear warmup over warmup_steps batches, tries to auto-detect the base lr"""
    def __init__(self, steps=1000, base_lr=None, invsqrt=True, decay=None):
        print('warmup_steps {}, base_lr {}, invsqrt {}, decay {}'.format(steps, base_lr, invsqrt, decay))
        self.warmup_steps = steps
        if decay is None:
            self.decay = steps
        else:
            self.decay = decay

        if base_lr is None:
            self.state = {'steps': 0, 'base_lr': base_lr}
        else:
            self.state = {'steps': 0, 'base_lr': float(base_lr)}

        self.invsqrt = invsqrt

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        optimizers = model.optimizers()

        if self.state['steps'] < self.warmup_steps:
            if type(optimizers) == 'list':
                if self.state['base_lr'] is None:
                    self.state['base_lr'] = [o.param_groups[0]['lr'] for o in optimizers]
                for opt,base in zip(optimizers, self.state['base_lr']):
                    self.set_lr(opt, self.state['steps']/self.warmup_steps * base)
            else:
                if self.state['base_lr'] is None:
                    self.state['base_lr'] = optimizers.param_groups[0]['lr']
                self.set_lr(optimizers, self.state['steps']/self.warmup_steps * self.state['base_lr'])
            self.state['steps'] += 1
        elif self.invsqrt:
            if type(optimizers) == 'list':
                if self.state['base_lr'] is None:
                    self.state['base_lr'] = [o.param_groups[0]['lr'] for o in optimizers]
                for opt,base in zip(optimizers, self.state['base_lr']):
                    self.set_lr(opt,base * (self.decay / (self.state['steps'] - self.warmup_steps + self.decay)) ** 0.5)
            else:
                if self.state['base_lr'] is None:
                    self.state['base_lr'] = optimizers.param_groups[0]['lr']
                self.set_lr(optimizers, self.state['base_lr'] * (
                            self.decay / (self.state['steps'] - self.warmup_steps + self.decay)) ** 0.5)
            self.state['steps'] += 1

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()

def average_models(models):
    """Averages model weights and loads the resulting weights into the first model, returning it"""
    models = list(models)
    n = len(models)
    sds = [m.state_dict() for m in models]
    averaged = {}
    for k in sds[0]:
        averaged[k] = sum(sd[k] for sd in sds) / n
    models[0].load_state_dict(averaged)
    return models[0]

seed = 2020
pl.seed_everything(seed)
dm = physionet.PhysioNetDataModule(batch_size=512, num_workers=16, use_temp_cache=True)
dm.setup()
pretrain_model = duett.pretrain_model(d_static_num=dm.d_static_num(),
        d_time_series_num=dm.d_time_series_num(), d_target=dm.d_target(), pos_frac=dm.pos_frac(),
        seed=seed)
checkpoint = pl.callbacks.ModelCheckpoint(save_last=True, monitor='val_loss', mode='min', save_top_k=1, dirpath='checkpoints')
warmup = WarmUpCallback(steps=2000)
trainer = pl.Trainer(gpus=1, logger=False, num_sanity_val_steps=2, max_epochs=300,
        gradient_clip_val=1.0, callbacks=[warmup, checkpoint])
trainer.fit(pretrain_model, dm)

pretrained_path = checkpoint.best_model_path
for seed in range(2020, 2023):
    pl.seed_everything(seed)
    fine_tune_model = duett.fine_tune_model(pretrained_path, d_static_num=dm.d_static_num(),
            d_time_series_num=dm.d_time_series_num(), d_target=dm.d_target(), pos_frac=dm.pos_frac(), seed=seed)
    checkpoint = pl.callbacks.ModelCheckpoint(save_top_k=5, save_last=False, mode='max', monitor='val_ap', dirpath='checkpoints')
    warmup = WarmUpCallback(steps=1000)
    trainer = pl.Trainer(gpus=1, logger=False, max_epochs=50, gradient_clip_val=1.0,
            callbacks=[warmup, checkpoint])
    trainer.fit(fine_tune_model, dm)
    final_model = average_models([duett.fine_tune_model(path, d_static_num=dm.d_static_num(),
            d_time_series_num=dm.d_time_series_num(), d_target=dm.d_target(), pos_frac=dm.pos_frac())
            for path in checkpoint.best_k_models.keys()])
    trainer.test(final_model, dataloaders=dm)
