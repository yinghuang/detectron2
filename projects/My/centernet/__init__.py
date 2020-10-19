from .config import add_centernet_config

# 只有导入FCOS类, 该模型才会被注册到detectron2的META_ARCH里面
from .centernet import CenterNet

