
def estimate_mbu(total_param_size, nb_hidden, hidden_size, nb_heads, nb_kv_heads, batch_size, sequence_length, dt, peak_bandwidth=600e9):
    kv_cache_size = batch_size * sequence_length * (hidden_size // nb_heads) * nb_hidden * 2 * nb_kv_heads * 2  # 2 bytes for fp16
    achieved_bandwidth = (total_param_size + kv_cache_size) / dt
    mbu = achieved_bandwidth / peak_bandwidth
    return mbu

def estimate_mfu(nb_parameters, nb_hidden, hidden_size, nb_heads, batch_size, dt, sequence_length, peak_flops=125e12):
    Q = hidden_size // nb_heads
    flops_per_token = 2 * nb_parameters + 4 * nb_hidden * nb_heads * Q * sequence_length
    flops_per_iter = flops_per_token * batch_size * sequence_length
    flops_achieved = flops_per_iter / dt
    mfu = flops_achieved / peak_flops
    return mfu