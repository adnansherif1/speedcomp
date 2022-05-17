from trainers import register_trainer

from .base_trainer_pretrain import BaseTrainerPretrain


@register_trainer("baseline_pretrain")
class BaselineTrainerPretrain(BaseTrainerPretrain):
    @staticmethod
    def name(args):
        return "baseline_pretrain"
