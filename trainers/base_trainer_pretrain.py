import torch
import wandb
from loguru import logger
from tqdm import tqdm


class BaseTrainerPretrain:
    @staticmethod
    def transform(args):
        return None

    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def train(model, device, loader, optimizer, args, calc_loss, scheduler=None, accelerator = None):
        [m.train() for m in model]
        
        loss_accum = [0 for _ in loader]
        t = [iter(tqdm(l, desc="Train", disable=not accelerator.is_main_process)) for l in loader]

        for step in range(args.steps):
            #batch = batch.to(device)
            # print("the shape of current batch", batch.x.shape)
            # exit()
            loss = [0 for _ in loader]
            valid = [False for _ in loader]
            for i in range(len(loader)):
                batch = next(t[i],None)
                loss = []
                if batch is None or batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                    pass
                else:
                    valid[i] = True
                    optimizer[i].zero_grad()
                    # print("Batch shape in trainer: ", batch.x.shape, batch.y.shape)
                    pred_list = model[i](batch)
                    # print("Batch shape in trainer after model: ", batch.x.shape, batch.y.shape)
                    # print("Pred List shape in trainer: ", pred_list.shape)

                    loss[i] = calc_loss[i](pred_list, batch)

                    # loss.backward()
                    accelerator.backward(loss[i])
                    
            for i in range(len(loader)):
                if valid[i]:
                    torch.nn.utils.clip_grad_norm_(model[i].parameters(), 1.0)
                    optimizer[i].step()

                    if scheduler[i]:
                        scheduler[i].step()

                    detached_loss = loss[i].item()
                    loss_accum[i] += detached_loss
                    # if accelerator.is_main_process:
                    #     t.set_description(f"Train (loss = {detached_loss:.4f}, smoothed = {loss_accum / (step + 1):.4f})")
                    #     wandb.log({"train/iter-loss": detached_loss, "train/iter-loss-smoothed": loss_accum / (step + 1)})
        if accelerator.is_main_process:
            print("in the main process")
            for _ in range(len(loader)):
                logger.info("Average training loss: {:.4f}".format(loss_accum[i] / (step + 1)))
        return loss_accum / (step + 1)

    @staticmethod
    def name(args):
        raise NotImplemented
