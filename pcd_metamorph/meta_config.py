operator_configs = {
    'rain_rate': {'range': (0.1, 10), 'decimals': 1},
    'snow_rate': {'range': (0.1, 2.4), 'decimals': 1},
    'visibility': {'range': (200, 1000), 'decimals': 1},
    'latency': {'range': (1, 300), 'decimals': 1},
    'trans_x': {'range': (-0.2, 0.2), 'decimals': 4},
    'trans_y': {'range': (-0.2, 0.2), 'decimals': 4},
    'trans_z': {'range': (-0.2, 0.2), 'decimals': 4},
    'yaw': {'range': (-0.033, 0.033), 'decimals': 4},
    'chlossy_p': {'type': 'random', 'decimals': 1},
    'lossy_p': {'type': 'random', 'decimals': 1}
}