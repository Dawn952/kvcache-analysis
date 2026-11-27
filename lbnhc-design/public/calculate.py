"""Recalculate the article's per-card byte ledger with Python's standard library.

Run from any working directory: python path/to/calculate.py
This is capacity arithmetic, not an engine or operator execution test.
"""

import json
from pathlib import Path


def cdiv(n, d):
    return (n + d - 1) // d


def local_tokens(total, rank, dcp, interleave):
    stripes, tail = divmod(total, interleave)
    rounds, extra = divmod(stripes, dcp)
    return rounds * interleave + (
        interleave if rank < extra else tail if rank == extra else 0
    )


def workload_example(p, state_ids, long_history_ids):
    """Illustrate quotas under two request mixes; the shared null is excluded."""
    page_coverage = p["cache"]["local_page_tokens"] * p["parallel"]["dcp"]
    short_history_ids = 1
    history_quota = long_history_ids + short_history_ids
    state_quota = 2 * state_ids
    usable = history_quota + state_quota
    two_short_history = 2 * short_history_ids
    two_short_states = 2 * state_ids
    three_short_history = 3 * short_history_ids
    three_short_states = 3 * state_ids
    return {
        "scope": "Illustrative per-card quotas using steady decode occupancy. Histories are private; null is reserved separately.",
        "short_request": {
            "input_tokens": page_coverage - p["sequence"]["output_budget_tokens"],
            "output_budget_tokens": p["sequence"]["output_budget_tokens"],
            "total_tokens": page_coverage,
            "history_ids": short_history_ids,
            "state_ids": state_ids,
            "private_ids": short_history_ids + state_ids,
        },
        "long_request": {
            "input_tokens": p["sequence"]["input_tokens"],
            "output_budget_tokens": p["sequence"]["output_budget_tokens"],
            "history_ids": long_history_ids,
            "state_ids": state_ids,
            "private_ids": long_history_ids + state_ids,
        },
        "fixed_quotas_for_one_long_plus_one_short": {
            "history_ids": history_quota,
            "state_ids": state_quota,
            "usable_ids": usable,
        },
        "two_short_requests": {
            "history_ids": two_short_history,
            "state_ids": two_short_states,
            "fixed_history_free_ids": history_quota - two_short_history,
            "fixed_state_free_ids": state_quota - two_short_states,
        },
        "three_short_requests": {
            "history_ids": three_short_history,
            "state_ids": three_short_states,
            "private_ids": three_short_history + three_short_states,
            "fits_fixed_quotas": (
                three_short_history <= history_quota and three_short_states <= state_quota
            ),
            "fits_shared_pool": three_short_history + three_short_states <= usable,
            "shared_free_ids": usable - three_short_history - three_short_states,
        },
    }


def calculate(p):
    parallel, seq = p["parallel"], p["sequence"]
    model, decode, cache = p["model"], p["decode"], p["cache"]
    block = cache["local_page_tokens"]
    dcp = parallel["dcp"]
    total = seq["input_tokens"] + seq["output_budget_tokens"]
    assert model["global_kda_heads"] % parallel["tp"] == 0
    heads = model["global_kda_heads"] // parallel["tp"]
    dim = model["kda_head_dim"]
    conv_positions = model["conv_kernel_width"] - 1 + decode["speculative_tokens"]
    conv = conv_positions * 3 * heads * dim * cache["conv_element_bytes"]
    recurrent = heads * dim * dim * cache["recurrent_element_bytes"]
    state = conv + recurrent
    mla_page = block * cache["mla_bytes_per_token"]
    indexer_page = block * cache["indexer_bytes_per_token"]
    draft_page = block * cache["draft_bytes_per_token"]
    assert state <= mla_page
    assert block % cache["operator_page_tokens"] == 0
    groups = cdiv(model["kda_layers"], model["main_layers"])
    group_widths = [len(range(i, model["kda_layers"], groups)) for i in range(groups)]
    states_per_layer = (
        cache["base_state_pages"]
        + cache["extra_speculative_state_pages"]
        + cache["extra_prefill_checkpoint_pages"]
    )
    boundary_states_per_layer = states_per_layer + cache["boundary_extra_state_pages"]
    pool_id_bytes = (
        model["main_layers"] * mla_page
        + model["indexer_layers"] * indexer_page
        + model["draft_layers"] * draft_page
    )
    history_ids = cdiv(total, block * dcp)

    def rank_ledger(rank, resident_state_pages):
        local = local_tokens(total, rank, dcp, seq["interleave_tokens"])
        state_ids = groups * resident_state_pages
        total_ids = history_ids + state_ids + cache["null_ids"]
        components = {
            "mla_history": local * model["main_layers"] * cache["mla_bytes_per_token"],
            "indexer_history": local * model["indexer_layers"] * cache["indexer_bytes_per_token"],
            "draft_history": local * model["draft_layers"] * cache["draft_bytes_per_token"],
            "kda_effective_states": model["kda_layers"] * resident_state_pages * state,
            "kda_page_padding": model["kda_layers"] * resident_state_pages * (mla_page - state),
            "kda_unused_layer_slots": (
                groups * model["main_layers"] - model["kda_layers"]
            ) * resident_state_pages * mla_page,
            "kda_id_empty_indexer_draft": state_ids * (
                model["indexer_layers"] * indexer_page + model["draft_layers"] * draft_page
            ),
            "history_tail": (history_ids * block - local) * (
                model["main_layers"] * cache["mla_bytes_per_token"]
                + model["indexer_layers"] * cache["indexer_bytes_per_token"]
                + model["draft_layers"] * cache["draft_bytes_per_token"]
            ),
            "null_id": cache["null_ids"] * pool_id_bytes,
        }
        effective = sum(components[k] for k in (
            "mla_history", "indexer_history", "draft_history", "kda_effective_states"
        ))
        pool = total_ids * pool_id_bytes
        assert sum(components.values()) == pool
        return {
            "dcp_rank": rank,
            "local_history_tokens": local,
            "history_ids": history_ids,
            "state_ids": state_ids,
            "null_ids": cache["null_ids"],
            "total_ids": total_ids,
            "pool_id_bytes": pool_id_bytes,
            "state_bytes": state,
            "components": components,
            "effective_bytes": effective,
            "overhead_bytes": pool - effective,
            "pool_bytes": pool,
            "pool_mib": pool / 2**20,
        }

    rows = [rank_ledger(rank, states_per_layer) for rank in range(dcp)]
    boundary_rows = [rank_ledger(rank, boundary_states_per_layer) for rank in range(dcp)]
    assert sum(r["local_history_tokens"] for r in rows) == total
    query_steps = 1 + decode["speculative_tokens"]
    correction_per_layer_slot = heads * query_steps * dim * 4
    key_gate_per_layer_slot = heads * query_steps * (2 * dim) * 2
    record_bytes_per_slot = model["kda_layers"] * (
        correction_per_layer_slot + key_gate_per_layer_slot
    )
    alignment = p["alignment_example"]
    unit = max(alignment["requested_page_tokens"], cache["operator_page_tokens"])
    chosen = unit * cdiv(state, unit * alignment["generic_mla_bytes_per_token"])
    return {
        "scope": "Per-card capacity arithmetic for the article's stated inputs; not HBM or operator measurement.",
        "parameters": p,
        "derived": {
            "kda_heads_per_rank": heads,
            "conv_state_bytes": conv,
            "recurrent_state_bytes": recurrent,
            "group_widths": group_widths,
            "states_per_kda_layer": states_per_layer,
            "boundary_states_per_kda_layer": boundary_states_per_layer,
            "mla_page_bytes": mla_page,
            "indexer_page_bytes": indexer_page,
            "draft_page_bytes": draft_page,
            "aligned_page_tokens_for_this_example": chosen,
            "correction_bytes_per_layer_slot": correction_per_layer_slot,
            "key_gate_bytes_per_layer_slot": key_gate_per_layer_slot,
            "record_bytes_per_slot": record_bytes_per_slot,
            "records_at_declared_capacity_bytes": record_bytes_per_slot * decode["record_capacity"],
        },
        "ranks": rows,
        "boundary_ranks": boundary_rows,
        "workload_example": workload_example(p, groups * states_per_layer, history_ids),
    }


if __name__ == "__main__":
    parameters = json.loads(Path(__file__).with_name("parameters.json").read_text(encoding="utf-8"))
    print(json.dumps(calculate(parameters), ensure_ascii=False, indent=2))
