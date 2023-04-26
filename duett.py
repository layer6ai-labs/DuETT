import torch
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import x_transformers

class BatchNormLastDim(nn.Module):
    def __init__(self, d, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(d, **kwargs)

    def forward(self, x):
        if x.ndim == 2:
            return self.batch_norm(x)
        elif x.ndim == 3:
            return self.batch_norm(x.transpose(1,2)).transpose(1,2)
        else:
            raise NotImplementedError("BatchNormLastDim not implemented for ndim > 3 yet")

def simple_mlp(d_in, d_out, n_hidden, d_hidden, final_activation=False, input_batch_norm=False,
        hidden_batch_norm=False, dropout=0., activation=nn.ReLU):
    # Could add options for different activations, batch norm, etc. as needed
    if n_hidden == 0:
        layers = ([BatchNormLastDim(d_in)] if input_batch_norm else []) + \
                [nn.Linear(d_in, d_out)]
    else:
        layers = ([BatchNormLastDim(d_in)] if input_batch_norm else []) + \
                [nn.Linear(d_in, d_hidden), activation(), nn.Dropout(dropout)] + \
                [l for _ in range(n_hidden-1) for l in ([BatchNormLastDim(d_hidden)] if hidden_batch_norm else []) + \
                    [nn.Linear(d_hidden, d_hidden), activation(), nn.Dropout(dropout)]] + \
				([BatchNormLastDim(d_hidden)] if hidden_batch_norm else []) + \
                [nn.Linear(d_hidden, d_out)]
    if final_activation:
        layers.append(activation())
    return nn.Sequential(*layers)

def pretrain_model(d_static_num, d_time_series_num, d_target, **kwargs):
    return Model(d_static_num, d_time_series_num, d_target, **kwargs)

def fine_tune_model(ckpt_path, **kwargs):
    return Model.load_from_checkpoint(ckpt_path, pretrain=False, aug_noise=0., aug_mask=0.5, transformer_dropout=0.5,
            lr=1.e-4, weight_decay=1.e-5, fusion_method='rep_token', **kwargs)

class Model(pl.LightningModule):
    def __init__(self, d_static_num, d_time_series_num, d_target, lr=3.e-4, weight_decay=1.e-1, glu=False,
            scalenorm=True, n_hidden_mlp_embedding=1, d_hidden_mlp_embedding=64, d_embedding=24, d_feedforward=512,
            max_len=48, n_transformer_head=2, n_duett_layers=2, d_hidden_tab_encoder=128, n_hidden_tab_encoder=1,
            norm_first=True, fusion_method='masked_embed', n_hidden_head=1, d_hidden_head=64, aug_noise=0., aug_mask=0.,
            pretrain=True, pretrain_masked_steps=1, pretrain_n_hidden=0, pretrain_d_hidden=64, pretrain_dropout=0.5,
            pretrain_value=True, pretrain_presence=True, pretrain_presence_weight=0.2, predict_events=True,
            transformer_dropout=0., pos_frac=None, freeze_encoder=False, seed=0, save_representation=None,
            masked_transform_timesteps=32, **kwargs):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.d_time_series_num = d_time_series_num
        self.d_target = d_target
        self.d_embedding = d_embedding
        self.max_len = max_len
        self.pretrain = pretrain
        self.pretrain_masked_steps = pretrain_masked_steps
        self.pretrain_dropout = pretrain_dropout
        self.freeze_encoder = freeze_encoder
        self.set_pos_frac(pos_frac)
        self.rng = np.random.default_rng(seed)
        self.aug_noise = aug_noise
        self.aug_mask = aug_mask
        self.fusion_method = fusion_method
        self.pretrain_presence = pretrain_presence
        self.pretrain_presence_weight = pretrain_presence_weight
        self.predict_events = predict_events
        self.masked_transform_timesteps = masked_transform_timesteps
        self.pretrain_value = pretrain_value
        self.save_representation = save_representation
        self.register_buffer("MASKED_EMBEDDING_KEY", torch.tensor(0)) # For multi-gpu training
        self.register_buffer("REPRESENTATION_EMBEDDING_KEY", torch.tensor(1))

        # For any special timesteps, e.g., masked, static, [CLS], etc.
        self.special_embeddings = nn.Embedding(8, d_embedding)
        self.embedding_layers = nn.ModuleList([
            simple_mlp(2, d_embedding, n_hidden_mlp_embedding, d_hidden_mlp_embedding, hidden_batch_norm=True)
            for _ in range(d_time_series_num)])

        self.n_obs_embedding = nn.Embedding(16, 1)

        if d_feedforward is None:
            d_feedforward = d_embedding * 4

        et_dim = d_embedding*(masked_transform_timesteps+1)
        tt_dim = d_embedding*(d_time_series_num+1)
        self.event_transformers = nn.ModuleList([x_transformers.Encoder(dim=et_dim, depth=1,
                heads=n_transformer_head, pre_norm=norm_first, use_scalenorm=scalenorm,
                attn_dim_head=d_embedding//n_transformer_head, ff_glu=glu,
                ff_mult=d_feedforward/et_dim, attn_dropout=transformer_dropout,
                ff_dropout=transformer_dropout) for _ in range(n_duett_layers)])
        self.full_event_embedding = nn.Embedding(d_time_series_num + 1, et_dim)
        self.time_transformers = nn.ModuleList([x_transformers.Encoder(dim=tt_dim, depth=1,
                heads=n_transformer_head, pre_norm=norm_first, use_scalenorm=scalenorm,
                attn_dim_head=d_embedding//n_transformer_head, ff_glu=glu,
                ff_mult=d_feedforward/tt_dim, attn_dropout=transformer_dropout,
                ff_dropout=transformer_dropout) for _ in range(n_duett_layers)])
        self.full_time_embedding =  self.cve(batch_norm=True, d_embedding=tt_dim)
        self.full_rep_embedding = nn.Embedding(tt_dim, 1)

        d_representation = d_embedding * (d_time_series_num + 1) # time_series + static
        self.head = simple_mlp(d_representation, d_target, n_hidden_head, d_hidden_head,
                hidden_batch_norm=True, final_activation=False, activation=nn.ReLU)
        self.pretrain_value_proj = simple_mlp(d_representation, d_time_series_num,
                pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True)
        if self.pretrain_presence:
            self.pretrain_presence_proj = simple_mlp(d_representation, d_time_series_num,
                    pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True)
        if self.predict_events:
            self.predict_events_proj = simple_mlp(et_dim, masked_transform_timesteps,
                    pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True)
            if self.pretrain_presence:
                self.predict_events_presence_proj = simple_mlp(et_dim, masked_transform_timesteps,
                        pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True)

        self.tab_encoder = simple_mlp(d_static_num, d_embedding, n_hidden_tab_encoder,
                    d_hidden_tab_encoder, hidden_batch_norm=True)

        self.pretrain_loss = F.mse_loss
        self.loss_function = F.binary_cross_entropy_with_logits
        self.pretrain_presence_loss = F.binary_cross_entropy_with_logits
        num_classes = None if d_target == 1 else d_target
        self.train_auroc = torchmetrics.AUROC(num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(num_classes=num_classes)
        self.train_ap = torchmetrics.AveragePrecision(num_classes=num_classes)
        self.val_ap = torchmetrics.AveragePrecision(num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(num_classes=num_classes)
        self.test_ap = torchmetrics.AveragePrecision(num_classes=num_classes)

    def set_pos_frac(self, pos_frac):
        if type(pos_frac) == list:
            pos_frac = torch.tensor(pos_frac, device=torch.device('cuda'))
        self.pos_frac = pos_frac
        if pos_frac != None:
            self.pos_weight = 1 / (2 * pos_frac)
            self.neg_weight = 1 / (2 * (1 - pos_frac))

    def cve(self, d_embedding=None, batch_norm=False):
        if d_embedding == None:
            d_embedding = self.d_embedding
        d_hidden = int(np.sqrt(d_embedding))
        if batch_norm:
            return nn.Sequential(nn.Linear(1, d_hidden), nn.Tanh(), BatchNormLastDim(d_hidden), nn.Linear(d_hidden, d_embedding))
        return nn.Sequential(nn.Linear(1, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_embedding))

    def feats_to_input(self, x, batch_size, limits=None):
        xs_ts, xs_static, times = x
        xs_ts = list(xs_ts)

        for i,f in enumerate(xs_ts):
            n_vars = f.shape[1] // 2
            if f.shape[0] > self.max_len:
                f = f[-self.max_len:]
                times[i] = times[i][-self.max_len:]
            # Aug
            if self.training and self.aug_noise > 0 and not self.pretrain:
                f[:,:n_vars] += self.aug_noise * torch.randn_like(f[:,:n_vars]) * f[:,n_vars:]
            f = torch.cat((f, torch.zeros_like(f[:,:1])), dim=1)
            if self.training and self.aug_mask > 0 and not self.pretrain:
                mask = torch.rand(f.shape[0]) < self.aug_mask
                f[mask,:] = 0.
                f[mask,-1] = 1.
            xs_ts[i] = f
        n_timesteps = [len(ts) for ts in times]

        pad_to = np.max(n_timesteps)
        xs_ts = torch.stack([F.pad(t, (0, 0, 0, pad_to-t.shape[0])) for t in xs_ts]).to(self.device)
        xs_times = torch.stack([F.pad(t, (0, pad_to-t.shape[0])) for t in times]).to(self.device)
        xs_static = torch.stack(xs_static).to(self.device)

        if self.training and self.aug_noise > 0 and not self.pretrain:
            xs_static += self.aug_noise * torch.randn_like(xs_static)

        return xs_static, xs_ts, xs_times, n_timesteps

    def pretrain_prep_batch(self, x, batch_size):
        xs_static, xs_ts, xs_times, n_timesteps = self.feats_to_input(x, batch_size)
        n_steps = xs_ts.shape[1]
        n_vars = (xs_ts.shape[2] - 1) // 2
        y_ts = []
        y_ts_n_obs = []
        y_events = []
        y_events_mask = []
        xs_ts_clipped = xs_ts.clone()
        for batch_i, n in enumerate(n_timesteps):
            if n < 2:
                mask_i = n
            elif self.pretrain_masked_steps > 1:
                if self.pretrain_masked_steps > n:
                    mask_i = np.arange(n)
                else:
                    mask_i = self.rng.choice(np.arange(n), size=self.pretrain_masked_steps)
            else:
                mask_i = self.rng.choice(np.arange(0, n))
            y_ts.append(xs_ts[batch_i,mask_i,:n_vars])
            y_ts_n_obs.append(xs_ts[batch_i,mask_i,n_vars:2*n_vars])

            xs_ts_clipped[batch_i, mask_i, :] = 0.
            xs_ts_clipped[batch_i,mask_i,-1] = 1.

            if self.predict_events:
                event_mask_i = self.rng.choice(np.arange(0, self.d_time_series_num))
                y_events.append(xs_ts[batch_i, :, event_mask_i])
                y_events_mask.append(xs_ts[batch_i, :, event_mask_i + n_vars].clip(0,1))
                xs_ts_clipped[batch_i, :, event_mask_i] = 0
                xs_ts_clipped[batch_i, :, event_mask_i + n_vars] = -1

        y_ts = torch.stack(y_ts)
        y_ts_n_obs = torch.stack(y_ts_n_obs)
        y_ts_masks = y_ts_n_obs.clip(0,1)
        if len(y_events) > 0:
            y_events = torch.stack(y_events)
            y_events_mask = torch.stack(y_events_mask)
        if self.pretrain_dropout > 0:
            keep = self.rng.random((batch_size, n_vars)) > self.pretrain_dropout
            keep = torch.tensor(keep, device=xs_ts.device)
            # Only drop out values that are unmasked in y
            if y_ts_masks.ndim > 2:
                keep = torch.logical_or(1 - y_ts_masks.sum(dim=1).clip(0,1), keep)
            else:
                keep = torch.logical_or(1 - y_ts_masks, keep)
            keep = torch.cat((keep.tile(1,2), torch.ones((batch_size, 1), device=keep.device)), dim=1)
            xs_ts_clipped *= torch.logical_or(keep.unsqueeze(1), xs_ts_clipped == -1)
        return (xs_static, xs_ts_clipped, xs_times, n_timesteps), y_ts, y_ts_masks, y_events, y_events_mask

    def forward(self, x, pretrain=False, representation=False):
        """
        Forward run
        :param x: input to the model
        :return: prediction output (i.e., class probabilities vector)
        """
        xs_static, xs_feats, xs_times, n_timesteps = x
        n_vars = xs_feats.shape[2] // 2
        if self.predict_events:
            event_mask_inds = xs_feats[:,:,n_vars:n_vars*2] == -1
            event_mask_inds = torch.cat((event_mask_inds, torch.zeros(xs_feats.shape[:2] + (1,), device=xs_feats.device, dtype=torch.bool)), dim=2)
            event_mask_inds = torch.cat((event_mask_inds, event_mask_inds[:,:1,:]), dim=1)
        n_obs_inds = xs_feats[:,:,n_vars:n_vars*2].to(int).clip(0, self.n_obs_embedding.num_embeddings - 1)
        xs_feats[:,:,n_vars:n_vars*2] = self.n_obs_embedding(n_obs_inds).squeeze(-1)

        embedding_layer_input = torch.empty(xs_feats.shape[:-1] + (n_vars, 2), dtype=xs_feats.dtype, device=xs_feats.device)
        embedding_layer_input[:,:,:,0] = xs_feats[:,:,:n_vars]
        embedding_layer_input[:,:,:,1] = xs_feats[:,:,n_vars:n_vars*2]
        # dims: batch, time step, var, embedding
        psi = torch.zeros((xs_feats.shape[0], xs_feats.shape[1]+1, n_vars+1, self.d_embedding), dtype=xs_feats.dtype, device=xs_feats.device)
        for i, el in enumerate(self.embedding_layers):
            psi[:,:-1,i,:] = el(embedding_layer_input[:,:,i,:])
        psi[:,:-1,-1,:] = self.tab_encoder(xs_static).unsqueeze(1)
        psi[:,-1,:,:] = self.special_embeddings(self.REPRESENTATION_EMBEDDING_KEY.to(self.device)).unsqueeze(0).unsqueeze(1)
        mask_inds = torch.cat((xs_feats[:,:,-1] == 1, torch.zeros((xs_feats.shape[0], 1), device=xs_feats.device, dtype=torch.bool)), dim=1)
        psi[mask_inds, :, :] = self.special_embeddings(self.MASKED_EMBEDDING_KEY.to(self.device))
        if self.predict_events:
            psi[event_mask_inds, :] = self.special_embeddings(self.MASKED_EMBEDDING_KEY.to(self.device))

        # batch, time step, full embedding
        time_embeddings = self.full_time_embedding(xs_times.unsqueeze(2))
        time_embeddings = torch.cat((time_embeddings,
            self.full_rep_embedding.weight.T.unsqueeze(0).expand(xs_feats.shape[0],-1,-1)),
            dim=1)
        for layer_i, (event_transformer, time_transformer) in enumerate(zip(self.event_transformers, self.time_transformers)):
            et_out_shape = (psi.shape[0], psi.shape[2], psi.shape[1], psi.shape[3])
            embeddings = psi.transpose(1,2).flatten(2) + self.full_event_embedding.weight.unsqueeze(0)
            event_outs = event_transformer(embeddings).view(et_out_shape).transpose(1,2)
            tt_out_shape = event_outs.shape
            embeddings = event_outs.flatten(2) + time_embeddings
            psi = time_transformer(embeddings).view(tt_out_shape)
        transformed = psi.flatten(2)

        if self.fusion_method == 'rep_token':
            z_ts = transformed[:,-1,:]
        elif self.fusion_method == 'masked_embed':
            if self.pretrain_masked_steps > 1:
                masked_ind = F.pad(xs_feats[:,:,-1] > 0, (0,1), value=False)
                z_ts = []
                for i in range(transformed.shape[0]):
                    z_ts.append(F.pad(transformed[i, masked_ind[i],:], (0,0,0,self.pretrain_masked_steps-masked_ind[i].sum()), value=0.))
                z_ts = torch.stack(z_ts) # batch size x pretrain_masked_steps x d_embedding
            else:
                masked_ind = xs_feats[:,:,-1:]
                z_ts = []
                for i in range(transformed.shape[0]):
                    z_ts.append(transformed[i, torch.nonzero(masked_ind[i].squeeze()==1),:])
                z_ts = torch.cat(z_ts, dim=0).squeeze()
        elif self.fusion_method == 'averaging':
            z_ts = torch.mean(transformed[:,:-1,:], dim=1)

        z = z_ts
        if representation:
            return z

        if pretrain:
            rep_token_head = torch.tile(transformed[:,0,:].unsqueeze(1), (1, self.masked_transform_timesteps, 1))
            y_hat_presence = self.pretrain_presence_proj(z).squeeze() if self.pretrain_presence else None
            y_hat_value = self.pretrain_value_proj(z).squeeze(1) if self.pretrain_value else None
            z_events = []
            y_hat_events, y_hat_events_presence = None, None
            if self.predict_events:
                for i in range(event_mask_inds.shape[0]):
                    z_events.append(psi[i][event_mask_inds[i].nonzero(as_tuple=True)].flatten())
                z_events = torch.stack(z_events)
                y_hat_events = self.predict_events_proj(z_events).squeeze()
                y_hat_events_presence = self.predict_events_presence_proj(z_events).squeeze() if self.pretrain_presence else None
            return y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence

        out = self.head(z).squeeze(1)

        if self.save_representation:
            return out, z
        else:
            return out

    def configure_optimizers(self):
        optimizers = [torch.optim.AdamW([p for l in self.modules() for p in l.parameters()],
                lr=self.lr, weight_decay=self.weight_decay)]
        return optimizers

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float64, device=self.device)
        batch_size = y.shape[0]
        if self.pretrain:
            x_pretrain, y, mask, y_events, y_events_mask = self.pretrain_prep_batch(x, batch_size)
            y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence = self.forward(x_pretrain, pretrain=True)

            loss = 0
            if self.pretrain_value:
                if self.pretrain_masked_steps > 1:
                    for i in range(self.pretrain_masked_steps):
                        loss += self.pretrain_loss(y_hat_value[:,i]*mask[:,i], y[:,i]*mask[:,i])
                    loss /= self.pretrain_masked_steps
                else:
                    loss = self.pretrain_loss(y_hat_value*mask, y*mask)
            if self.pretrain_presence:
                if self.pretrain_masked_steps > 1:
                    presence_loss = 0
                    for i in range(self.pretrain_masked_steps):
                        presence_loss += self.pretrain_presence_loss(y_hat_presence[:,i], mask[:,i]) * self.pretrain_presence_weight
                    presence_loss /= self.pretrain_masked_steps
                else:
                    presence_loss = self.pretrain_presence_loss(y_hat_presence, mask) * self.pretrain_presence_weight
                loss += presence_loss
            if self.predict_events:
                if self.pretrain_value:
                    loss += self.pretrain_loss(y_hat_events*y_events_mask, y_events*y_events_mask)
                if self.pretrain_presence:
                    loss += self.pretrain_presence_loss(y_hat_events_presence, y_events_mask) * self.pretrain_presence_weight
        else:
            y_hat = self.forward(self.feats_to_input(x, batch_size))
            if self.pos_frac is not None:
                weight = torch.where(y > 0, self.pos_weight, self.neg_weight)
                loss = self.loss_function(y_hat, y, weight)
            else:
                loss = self.loss_function(y_hat, y)
            self.train_auroc.update(y_hat, y.to(int))
            self.train_ap.update(y_hat, y.to(int))

        # Workaround to fix the loss=nan issue on the train progress bar
        # self.trainer.train_loop.running_loss.append(loss)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float64, device=self.device)
        batch_size = y.shape[0]
        if self.pretrain:
            x_pretrain, y, mask, y_events, y_events_mask = self.pretrain_prep_batch(x, batch_size)
            y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence = self.forward(x_pretrain, pretrain=True)

            loss = 0
            if self.pretrain_value:
                if self.pretrain_masked_steps > 1:
                    for i in range(self.pretrain_masked_steps):
                        loss += self.pretrain_loss(y_hat_value[:,i]*mask[:,i], y[:,i]*mask[:,i])
                    loss /= self.pretrain_masked_steps
                else:
                    loss = self.pretrain_loss(y_hat_value*mask, y*mask)
                self.log('val_next_loss', loss, on_epoch=True, sync_dist=True, rank_zero_only=True)
            if self.pretrain_presence:
                if self.pretrain_masked_steps > 1:
                    presence_loss = 0
                    for i in range(self.pretrain_masked_steps):
                        presence_loss += self.pretrain_presence_loss(y_hat_presence[:,i], mask[:,i]) * self.pretrain_presence_weight
                    presence_loss /= self.pretrain_masked_steps
                else:
                    presence_loss = self.pretrain_presence_loss(y_hat_presence, mask) * self.pretrain_presence_weight
                self.log('val_presence_loss', presence_loss, on_epoch=True, sync_dist=True, rank_zero_only=True)
                loss += presence_loss
            if self.predict_events:
                event_loss = self.pretrain_loss(y_hat_events*y_events_mask, y_events*y_events_mask)
                self.log('val_event_loss', event_loss, on_epoch=True, sync_dist=True, rank_zero_only=True)
                loss += event_loss
        else:
            y_hat = self.forward(self.feats_to_input(x, batch_size))
            if self.pos_frac is not None:
                weight = torch.where(y > 0, self.pos_weight, self.neg_weight)
                loss = self.loss_function(y_hat, y, weight)
            else:
                loss = self.loss_function(y_hat, y)
            self.val_auroc.update(y_hat, y.to(int).to(self.device))
            self.val_ap.update(y_hat, y.to(int).to(self.device))

        if not self.pretrain:
            self.log('val_ap', self.val_ap, on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.log('val_auroc', self.val_auroc, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True, rank_zero_only=True)

    def training_epoch_end(self, training_step_outputs):
        if not self.pretrain:
            self.log('train_auroc', self.train_auroc, sync_dist=True, rank_zero_only=True)
            self.log('train_ap', self.train_ap, sync_dist=True, rank_zero_only=True)

    def validation_epoch_end(self, validation_step_outputs):
        if not self.pretrain:
            print("val_auroc", self.val_auroc.compute(), "val_ap", self.val_ap.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float64, device=self.device)
        batch_size = y.shape[0]
        if self.save_representation:
            y_hat, z = self.forward(self.feats_to_input(x, batch_size))

            print("saving representations...")
            with open(self.save_representation, 'ab') as f:
                if y.ndim == 1:
                    np.savetxt(f,np.concatenate([z.cpu(), y.unsqueeze(1).cpu()], axis=1))
                else:
                    np.savetxt(f,np.concatenate([z.cpu(), y.cpu()], axis=1))
        else:
            y_hat = self.forward(self.feats_to_input(x, batch_size))
        if self.pos_frac is not None:
            weight = torch.where(y > 0, self.pos_weight, self.neg_weight)
            loss = self.loss_function(y_hat, y, weight)
        else:
            loss = self.loss_function(y_hat, y)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.test_auroc.update(y_hat, y.to(int).to(self.device))
        self.log('test_auroc', self.test_auroc, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.test_ap.update(y_hat, y.to(int).to(self.device))
        self.log('test_ap', self.test_ap, on_epoch=True, sync_dist=True, rank_zero_only=True)

        return loss, self.test_auroc, self.test_ap

    def on_load_checkpoint(self, checkpoint):
        # Ignore errors from size mismatches in head, since those might change between pretraining
        # and supervised training
        # Adapted from https://github.com/PyTorchLightning/pytorch-lightning/issues/4690#issuecomment-731152036
        print('Loading from checkpoint')
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in model_state_dict:
            if k not in state_dict:
                state_dict[k] = model_state_dict[k]
                is_changed = True
        for k in state_dict:
            if k in model_state_dict:
                if k.startswith('head') and state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

        if self.freeze_encoder:
            self.freeze()

    def freeze(self):
        print('Freezing')
        for n, w in self.named_parameters():
            if "head" not in n:
                w.requires_grad = False
            else:
                print("Skip freezing:", n)
