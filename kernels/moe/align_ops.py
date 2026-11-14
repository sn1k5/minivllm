import os

from common import build

_mod = build("align_ops", os.path.dirname(os.path.abspath(__file__)),
             ["align_ops.cu"])


def moe_align(topk_ids, block_size, num_experts):
    sorted_ids, expert_ids, num_post = _mod.moe_align(topk_ids, num_experts,
                                                      block_size)
    return sorted_ids, expert_ids, num_post
