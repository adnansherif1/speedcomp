from .code import CodeUtil
from .mol import MolUtil
from .tud import TUUtil
from .tud_pretrain import TUUtil_pretrain
from .pcqm import PcqmUtil
DATASET_UTILS = {
    'ogbg-code': CodeUtil,
    'ogbg-code2': CodeUtil,
    'ogbg-molhiv': MolUtil,
    'ogbg-molpcba': MolUtil,
    'NCI1': TUUtil,
    'NCI109': TUUtil,
    'AIDS': TUUtil,
    'IMDB-BINARY': TUUtil,
    'IMDB-MULTI': TUUtil,
    'pretrain':TUUtil_pretrain,
    'COLLAB':TUUtil,
    'PROTEINS':TUUtil,
    'DD':TUUtil,
    'ogbg-pcqm':PcqmUtil,
}
