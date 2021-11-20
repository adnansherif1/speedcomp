import torch
import wandb
from loguru import logger
from tqdm import tqdm


class BaseTrainer:
    @staticmethod
    def transform(args):
        return None

    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def train(model, device, loader, optimizer, args, calc_loss, scheduler=None, accelerator = None):
        model.train()

        loss_accum = 0
        t = tqdm(loader, desc="Train", disable=not accelerator.is_main_process)
        for step, batch in enumerate(t):
            #batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                optimizer.zero_grad()
                # print("Batch shape in trainer: ", batch.x.shape, batch.y.shape)
                pred_list = model(batch)
                # print("Batch shape in trainer after model: ", batch.x.shape, batch.y.shape)
                # print("Pred List shape in trainer: ", pred_list.shape)

                loss = calc_loss(pred_list, batch)

                # loss.backward()
                accelerator.backward(loss)
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                detached_loss = loss.item()
                loss_accum += detached_loss
                if accelerator.is_main_process:
                    t.set_description(f"Train (loss = {detached_loss:.4f}, smoothed = {loss_accum / (step + 1):.4f})")
                    wandb.log({"train/iter-loss": detached_loss, "train/iter-loss-smoothed": loss_accum / (step + 1)})
        if accelerator.is_main_process:
            print("in the main process")
            logger.info("Average training loss: {:.4f}".format(loss_accum / (step + 1)))
        return loss_accum / (step + 1)

    @staticmethod
    def name(args):
        raise NotImplemented
