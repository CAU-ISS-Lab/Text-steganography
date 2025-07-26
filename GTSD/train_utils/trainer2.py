from random import random

import torch
import copy
import os
import logging
import collections
import enum

from tqdm import tqdm
from torch.optim import AdamW
#from torch.utils.tensorboard import SummaryWriter
from torch import autocast
from torch.cuda.amp import GradScaler

from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModel

from utils import get_group_parameters, load_states_from_checkpoint


logger = logging.getLogger(__name__)
CheckpointState = collections.namedtuple(
    "CheckpointState", ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL
class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

def get_t(batch,schedule_sampler,device):
    t, weights = schedule_sampler.sample(batch['src_input_ids'].shape[0],device,)
    return t
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
def training_losses_s2s(self, model, input_text, t, is_dev=False):
    """
    Compute training losses for a single timestep.

    :param model: the model to evaluate loss on.
    :param x_start: the [N x C x ...] tensor of inputs.
    :param t: a batch of timestep indices.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :param noise: if specified, the specific Gaussian noise to try to remove.
    :return: a dict with the key "loss" containing a tensor of shape [N].
             Some mean or variance settings may also have other keys.
    """

    q_input_ids = input_text['tgt_input_ids'].long().to(t.device)

    x_start_mean = model.get_embeds(q_input_ids)  # because of DDP

    p_input_ids = input_text['src_input_ids'].long().to(t.device)
    p_attention_mask = input_text['src_attention_mask'].long().to(t.device)
    q_attention_mask = input_text['tgt_attention_mask'].long().to(t.device) if self.config.pred_len else None
    tgt_length = input_text['length'].long().to(t.device) if self.config.pred_len else None

    # the variance of x_0 is \sqrt{(1 - \bar{α}_t)} when t=0
    std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                               torch.tensor([0]).to(x_start_mean.device),  # t=0
                               x_start_mean.shape)
    x_start_log_var = 2 * torch.log(std)
    x_start = self.get_x_start(x_start_mean, std, self.knoise)  # (bs, seq_len, hz)

    noise = None
    if noise is None:
        noise = torch.randn_like(x_start)  # self.prompt1

    # reparametrization trick.
    x_t = self.q_sample(x_start, t, noise=noise)

    get_logits = model.get_logits  # passed in is the method

    terms = {}

    if self.loss_type == LossType.E2E_MSE or self.loss_type == LossType.RESCALED_MSE:
        x_self_cond = None
        if self.config.self_condition and random() < 0.5:
            with torch.no_grad():
                prev_output, length_out = model(tgt_emb=x_t,
                                                timesteps=self._scale_timesteps(t),
                                                x_self_cond=x_self_cond,
                                                src_input_ids=p_input_ids,
                                                src_attention_mask=p_attention_mask,
                                                tgt_attention_mask=q_attention_mask,
                                                tgt_length=tgt_length, )
                x_self_cond = self.model_predictions(x_t, t, prev_output)['pred_x_start']
                # beacause of the DDP, the detach_() is unavailable
                # detach and detach_ are all stop gradient, but detach will generate a new tensor
                x_self_cond.detach()

        # model: LM model, input the src, output the tgt
        model_output, length_out = model(tgt_emb=x_t,
                                         timesteps=self._scale_timesteps(t),
                                         x_self_cond=x_self_cond,
                                         src_input_ids=p_input_ids,
                                         src_attention_mask=p_attention_mask,
                                         tgt_attention_mask=q_attention_mask,
                                         tgt_length=tgt_length, )

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,  # choose
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape

        # calculate the MSE loss about \bar{x_0}(contain variance)
        terms["mse"] = ((target - model_output) ** 2).mean(-1)  # [bs, seqlen]
        if self.config.schedule_sampler == 'uniform':
            terms["mse"] = terms["mse"].mean(-1)  # [bs]

        # only when t=0, its distribution is completely close to the distribution of embedding i.e. t0_loss,
        # calculate the MSE loss about x_0(no variance)
        model_out_x_start = self.x0_helper(model_output, x_t, t)['pred_xstart']
        t0_mask = (t == 0)
        t0_loss = ((x_start_mean - model_out_x_start) ** 2).mean(-1)  # [bs, seqlen]
        if self.config.schedule_sampler == 'uniform':
            t0_loss = t0_loss.mean(-1)  # [bs]

        # only when t=0, its distribution is completely close to the distribution of embedding i.e. t0_loss,
        # otherwise it is close to x_{0} i.e. terms["mse"]
        terms["t0_loss"] = t0_loss
        terms["mse_pre"] = terms["mse"]  # [bs, seqlen] / [bs]
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])  # [bs, seqlen] / [bs]

        # let the x_T approximate the Guassian, \mu = 0
        if self.config.predict_x_start:
            x_output = model_out_x_start
        else:
            x_output = x_start

        out_mean, _, _ = self.q_mean_variance(x_output, torch.LongTensor(
            [self.num_timesteps - 1]).to(x_output.device))
        # tT_loss = mean_flat(out_mean ** 2)
        tT_loss = (out_mean ** 2).mean(-1)
        if self.config.schedule_sampler == 'uniform':
            tT_loss = tT_loss.mean(-1)
        terms["tT_loss"] = tT_loss

        # At each step, the cross-entropy with the real data is calculated.
        decoder_nll = self.token_discrete_loss(x_output, get_logits, q_input_ids)
        terms["decoder_nll"] = decoder_nll
        if self.config.schedule_sampler == 'uniform':
            terms["decoder_nll"] = terms["decoder_nll"].mean(-1)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        if self.config.pred_len:
            # length_out: [bs, seq_len] / tgt_length: [bs,]
            tgt_length = tgt_length.view(-1)
            # reduce the loss scale
            terms["length_loss"] = self.config.length_factor * loss_fct(
                length_out, tgt_length).unsqueeze(-1)
            if is_dev:
                top1_len = torch.topk(length_out, 1, dim=-1)[1]
                correct = torch.eq(top1_len.view(-1), tgt_length.view(-1)).sum(0)
                terms["top-1_acc"] = correct / length_out.size(1)

                top5_len = torch.topk(length_out, 5, dim=-1)[1]
                correct = torch.eq(
                    top5_len.view(-1),
                    tgt_length.view(-1, 1).expand_as(top5_len).contiguous().view(-1)
                ).sum(0)
                terms["top-5_acc"] = correct / length_out.size(1)
    else:
        raise NotImplementedError(self.loss_type)

    return terms


class TrainLoop2:
    def __init__(
        self,
        config,
        model,
        diffusion,
        data,
        dev_data,
        schedule_sampler,
    ):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.dev_data = dev_data
        self.schedule_sampler = schedule_sampler
        
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.ema_rate = (
            [config.ema_rate]
            if isinstance(config.ema_rate, float)
            else [float(x) for x in config.ema_rate.split(",")]
        )
        self.log_interval = config.log_interval
        self.eval_interval = config.eval_interval
        self.save_interval = config.save_interval
        self.warmup_steps = config.warmup_steps
        self.weight_decay = config.weight_decay
        self.total_steps = config.total_steps
        self.gradient_clipping = config.grad_clip
        self.gradient_accumulation_steps = config.grad_accum
        self.device = config.device
        self.train_type = config.model.mode

        self.master_params = list(self.model.parameters())
        self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        self.checkpoint_path = config.exp.dir
        #self.writer = SummaryWriter(log_dir=self.checkpoint_path + '/board')
        
        if self.config.use_AMP:
            self.scaler = GradScaler()

        if config.load_bart:
            self.optimizer = AdamW(get_group_parameters(config, self.model))
        else:
            self.optimizer = AdamW(
                self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        
        if config.data.name == 'commongen':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.warmup_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.warmup_steps, 
                num_training_steps=int(config.lr_step)
            )
        self.global_step = 0

        # auto load last checkpoint
        self.model_path = os.path.join(self.checkpoint_path, 'model')
        if (0 == 0) and (not os.path.exists(self.model_path)):
            os.mkdir(self.model_path)
        if config.resume_checkpoint:
            self.check_load()

        # model to DDP
        self.model = self.model

        if config.fix_encoder:
            model_cfg = AutoConfig.from_pretrained(config.model.name)
            self.encoder = AutoModel.from_pretrained(config.model.name, config=model_cfg)
            if config.load_bart:
                self.encoder = self.encoder.encoder
            self.encoder.to(self.device)
            self.encoder = self.encoder
        else:
            self.encoder = None

    def run_loop(self):
        if 0 == 0:
            print("***** Running training *****")
            logger.info(f"  Max steps = {self.total_steps}")
            logger.info(f"  Instantaneous batch size per GPU = {self.batch_size}")
            bs = self.batch_size * self.gradient_accumulation_steps * (1)
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {bs}"
            )
            logger.info(f"  Total warm up steps = {self.warmup_steps}")
            logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
            
        if self.config.continue_train and (
            abs(self.optimizer.param_groups[0]['lr'] - self.scheduler.get_lr()[0]) > 1e-10):
            self.scheduler.step()

        while self.global_step < self.total_steps:            
            epoch_iterator = tqdm(
                self.data, desc="Iteration", disable=0 not in [-1, 0]
            )

            self.model.zero_grad()
            self.model.eval()
            for step, batch in enumerate(epoch_iterator):

                self.forward_backward(batch)

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    
                    if self.gradient_clipping > 0:
                        if self.config.use_AMP:
                            self.scaler.unscale_(self.optimizer)
                        self.grad_clip()
                    
                    if self.config.use_AMP:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.model.zero_grad()

                if self.global_step % self.log_interval == 0:
                    self.writer.add_scalar(
                        tag='learning_rate', 
                        scalar_value=self.optimizer.param_groups[0]['lr'], 
                        global_step=self.global_step
                    )
                    
                # ema schedule. It actually save a shadow model for evalution.
                # It doesn't change the training process.
                for rate, params in zip(self.ema_rate, self.ema_params):
                    self.update_ema(params, self.master_params, rate=rate)
                self.global_step += 1
                
                # dev dataset for evaluation
                if self.dev_data is not None and self.global_step % self.eval_interval == 0:
                    if 0 == 0:
                        print('eval on validation set...')
                        for step, batch in tqdm(enumerate(self.dev_data)):
                            if step > 50:
                                break
                            # self.forward_only(step, batch, step_ratio)
                            self.forward_only(step, batch)

                # save model
                if (self.total_steps - self.global_step) < 30000 and self.global_step % self.save_interval == 0:
                    self.save()
                elif self.global_step % 10000 == 0:
                    self.save()

    # def forward_backward(self, batch, step_ratio):
    def forward_backward(self, batch):

        if self.train_type == 's2s':
            # the timestep t is random sample.
            if self.config.schedule_sampler == 'uniform':
                t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], 
                                                          self.device, 
                                                        #   step_ratio=1.0-step_ratio,
                                                          )
            else:
                t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], 
                                                          self.device, 
                                                          seq_len=batch['length'].to(self.device),
                                                        #   step_ratio=1.0-step_ratio,
                                                          )

            # print("config.use_AMP = ",self.config.use_AMP)
            if self.config.use_AMP:
                with autocast(device_type='cuda', dtype=torch.float16):
                    losses = self.diffusion.training_losses(self.model, batch, t)
                    
                    # loss moment
                    if self.config.schedule_sampler == 'loss-second-moment':
                        self.schedule_sampler.update_with_local_losses(
                            t, losses["loss"].detach()
                        )
                        
                    if self.config.loss_aware:
                        self.schedule_sampler.update_with_local_losses(
                            t, losses["loss"].detach()
                        )
                    
                    if self.config.pred_len:
                        loss = (losses["loss"] * weights * batch['tgt_attention_mask'].to(self.device)).mean()
                    else:
                        loss = (losses["loss"] * weights).mean()
                    loss = loss / self.gradient_accumulation_steps
                
                # if self.config.grad_penalty:
                #     scaled_grad_params = torch.autograd.grad(
                #         outputs=self.scaler.scale(loss), inputs=self.model.parameters(), create_graph=True)

                #     inv_scale = 1. / self.scaler.get_scale()
                #     grad_params = [p * inv_scale for p in scaled_grad_params]

                #     with autocast(device_type='cuda', dtype=torch.float16):
                #         grad_norm = 0
                #         for grad in grad_params:
                #             grad_norm += grad.pow(2).sum()
                #         grad_norm = grad_norm.sqrt()
                #         loss = loss + grad_norm    
                
            else:
                losses = self.diffusion.training_losses(self.model, batch, t)
                
                # loss moment
                if self.config.schedule_sampler == 'loss-second-moment':
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                
                if self.config.loss_aware:
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                loss = loss / self.gradient_accumulation_steps
                
                # if self.config.grad_penalty:
                #     grad_params = torch.autograd.grad(
                #         outputs=loss, inputs=self.model.parameters(), create_graph=True)
                #     grad_norm = 0
                #     for grad in grad_params:
                #         grad_norm += grad.pow(2).sum()
                #     grad_norm = grad_norm.sqrt()
                #     loss = loss + grad_norm

        else:
            return NotImplementedError
        
        if self.config.use_AMP:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.global_step % self.log_interval == 0:
            for key, value in losses.items():
                if self.config.pred_len:
                    losses[key] = (value * weights * batch['tgt_attention_mask'].to(self.device)).mean()
                else:
                    losses[key] = (value * weights).mean()
                self.writer.add_scalar(
                    tag=f'train_loss/{key}', scalar_value=losses[key], global_step=self.global_step)
            logger.info(f" global_step = {self.global_step} ,total_step = {self.total_steps}, train_loss = {loss}")

        # for index, (name, p) in enumerate(self.model.module.named_parameters()):
        #     if p.grad == None:
        #         print(index, name)

    # def forward_only(self, step, batch, step_ratio):
    def forward_only(self, step, batch):
        with torch.no_grad():
            self.model.zero_grad()
            if self.train_type == 's2s':
                if self.config.schedule_sampler == 'uniform':
                    t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], 
                                                              self.device, 
                                                            #   step_ratio=1.0-step_ratio
                                                            )
                else:
                    t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], 
                                                              self.device, 
                                                              seq_len=batch['length'].to(self.device),
                                                            #   step_ratio=1.0-step_ratio,
                                                            )
                if self.config.use_AMP:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        losses = self.diffusion.training_losses(self.model, batch, t, is_dev=True)
                else:
                    losses = self.diffusion.training_losses(self.model, batch, t, is_dev=True)
            else:
                return NotImplementedError

            for key, value in losses.items():
                if 'acc' not in key:
                    if self.config.pred_len:
                        value = (value * weights * batch['tgt_attention_mask'].to(self.device)).mean()
                    else:
                        value = (value * weights).mean()
                self.writer.add_scalar(
                    tag=f'eval_loss/{key}', scalar_value=value, global_step=self.global_step)

    def check_load(self):
        model_checkpoint_files = []
        ema_checkpoint_files = []
        for item in os.scandir(self.model_path):
            if item.is_file():
                if "model_checkpoint" in item.path:
                    model_checkpoint_files.append(item.path)
                if "ema" in item.path:
                    ema_checkpoint_files.append(item.path)

        if not self.config.load_from_ema and len(model_checkpoint_files) != 0:
            model_checkpoint_files.sort(key=lambda f: int(
                f.split('model_checkpoint-')[1]), reverse=True)
            if 0 == 0:
                logger.info("***** load " + model_checkpoint_files[0] + " *****")

            model_saved_state = load_states_from_checkpoint(
                model_checkpoint_files[0], 0)
            self.global_step = self._load_saved_state(model_saved_state)

        elif self.config.load_from_ema and len(ema_checkpoint_files) != 0:
            ema_checkpoint_files.sort(key=lambda f: int(
                f.split('checkpoint-')[-1]), reverse=True)
            if 0 == 0:
                logger.info("***** load " + ema_checkpoint_files[0] + " *****")
            
            ema_saved_state = load_states_from_checkpoint(
                ema_checkpoint_files[0], 0)
            self.ema_params = [
                [ema_saved_state.model_dict[name].to(self.device) 
                 for name, _ in self.model.named_parameters()]
                for _ in range(len(self.ema_rate))
            ]
            self.global_step = self._load_saved_state(ema_saved_state)

        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
            if 0 == 0:
                logger.info("***** there are no checkpoint in " + self.model_path + " *****")
                
    def _load_saved_state(self, saved_state: CheckpointState):
        self.global_step = saved_state.offset
        if 0 == 0:
            logger.info('Loading checkpoint @ step=%s', self.global_step)
            print('Loading saved model state ...')
        if self.config.continue_train:
            saved_state.scheduler_dict['base_lrs'] = [self.config.lr]
            if 0 == 0:
                logger.info('Now the learning rate is %s', saved_state.scheduler_dict['base_lrs'])
        # set strict=False if you use extra projection
        self.model.load_state_dict(saved_state.model_dict)
        self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler.load_state_dict(saved_state.scheduler_dict)
        self.master_params = list(self.model.parameters())
        return self.global_step

    def save(self):
        def save_checkpoint(rate, ema_params):
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            if not rate:
                model_state_dict = model_to_save.state_dict()
            else:
                model_state_dict = model_to_save.state_dict()
                for i, (name, _value) in enumerate(model_to_save.named_parameters()):
                    assert name in model_state_dict
                    model_state_dict[name] = ema_params[i]

            opt_state_dict = self.optimizer.state_dict()
            sch_state_dict = self.scheduler.state_dict()
            offset = self.global_step
            state = CheckpointState(
                model_state_dict, opt_state_dict, sch_state_dict, offset,)
            if not rate:
                ckpt_path = os.path.join(
                    self.model_path, 'model_checkpoint-' + str(offset))
            else:
                ckpt_path = os.path.join(
                    self.model_path, 'ema_' + str(rate) + '_checkpoint-' + str(offset))

            torch.save(state._asdict(), ckpt_path)
            if 0 == 0:
                logger.info('Saved checkpoint at %s', ckpt_path)

        if 0 == 0:
            save_checkpoint(0, None)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)

    def grad_clip(self):
        max_grad_norm = self.gradient_clipping  # 3.0
        if hasattr(self.optimizer, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.optimizer.clip_grad_norm(max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def update_ema(self, target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)
