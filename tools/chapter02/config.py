"""Model configurations; these are teaching models, not DeepSeek replicas."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 50257
    width: int = 768
    layers: int = 6
    heads: int = 12
    hidden: int = 2048
    max_context: int = 8192
    rope_theta: float = 10000.0
    dropout: float = 0.0
    q_rank: int = 128
    kv_rank: int = 128
    nope_dim: int = 32
    rope_dim: int = 32
    value_dim: int = 64
    experts: int = 8
    top_k: int = 2
    expert_hidden: int = 512
    shared_experts: int = 0
    router_float32: bool = False

    def __post_init__(self):
        for name in ('vocab', 'width', 'layers', 'heads', 'hidden', 'max_context',
                     'q_rank', 'kv_rank', 'nope_dim', 'rope_dim', 'value_dim',
                     'experts', 'top_k', 'expert_hidden'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('width / heads must be an even integer')
        if self.rope_dim % 2 or self.top_k > self.experts:
            raise ValueError('rope_dim must be even and top_k <= experts')
        if not 0 <= self.dropout < 1 or self.shared_experts not in (0, 1):
            raise ValueError('invalid dropout or shared_experts')
        if self.rope_theta <= 0:
            raise ValueError('rope_theta must be positive')


def preset(name):
    if name == 'teaching':
        return ModelConfig()
    if name == 'tiny':
        return ModelConfig(vocab=97, width=32, layers=2, heads=4, hidden=64,
                           max_context=256, q_rank=16, kv_rank=16, nope_dim=4,
                           rope_dim=4, value_dim=8, experts=4, top_k=2,
                           expert_hidden=24)
    raise ValueError(f'unknown preset: {name}')
