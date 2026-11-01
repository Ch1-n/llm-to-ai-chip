from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    vocab: int = 50257
    width: int = 768
    layers: int = 6
    heads: int = 12
    head_dim: int = 64
    rope_dim: int = 16
    q_rank: int = 128
    o_groups: int = 4
    o_rank: int = 64
    hidden: int = 2048
    window: int = 128
    index_heads: int = 4
    index_dim: int = 32
    index_topk: int = 8
    ratios: tuple = (0, 0, 4, 128, 4, 128)
    max_context: int = 8192
    query_tile: int = 128
    eps: float = 1e-6
    rope_theta: float = 10000.
    compress_rope_theta: float = 40000.

    def __post_init__(self):
        if any(getattr(self, name) < 1 for name in ('vocab', 'width', 'layers', 'heads',
            'head_dim', 'rope_dim', 'q_rank', 'o_groups', 'o_rank', 'hidden', 'window',
            'index_heads', 'index_dim', 'index_topk', 'max_context', 'query_tile')):
            raise ValueError('dimensions must be positive')
        if self.heads % self.o_groups or self.rope_dim % 2:
            raise ValueError('invalid groups or rotary dimension')
        if self.rope_dim > min(self.head_dim, self.index_dim):
            raise ValueError('rotary dimension exceeds head dimension')
        if len(self.ratios) != self.layers or any(r not in (0, 4, 128) for r in self.ratios):
            raise ValueError('one SWA/CSA/HCA compression ratio is required per layer')


VARIANTS = ('shared-dense', 'swa', 'csa', 'hca', 'hybrid')


def preset(name='teaching', variant='hybrid'):
    if name == 'teaching':
        cfg = Config()
    elif name == 'tiny':
        cfg = Config(vocab=97, width=32, layers=3, heads=4, head_dim=16, rope_dim=4,
                     q_rank=16, o_groups=2, o_rank=8, hidden=64, window=8,
                     index_heads=2, index_dim=8, index_topk=2, ratios=(0, 4, 128),
                     max_context=512, query_tile=32)
    else:
        raise ValueError(name)
    if variant == 'hybrid':
        return cfg
    if variant not in VARIANTS:
        raise ValueError(variant)
    ratio = {'shared-dense': 0, 'swa': 0, 'csa': 4, 'hca': 128}[variant]
    return replace(cfg, ratios=(ratio,) * cfg.layers,
                   window=cfg.max_context if variant == 'shared-dense' else cfg.window)
