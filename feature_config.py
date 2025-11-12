# Default normalization constants
DEFAULT_TANH_CLIP = 0.999
OBS_EPS = 1e-8
# Default maximum sizes
MAX_NUM_TOKENS = 1  # Changed from 16 to match mediator and lob_state_cython
EXT_NORM_DIM = 21  # Expanded from 16 to 21 to include all technical features (including taker_buy_ratio derivatives)

# Initialize feature layout (to be populated by make_layout)
FEATURES_LAYOUT = []

def make_layout(obs_params=None):
    """Build the observation feature layout list based on parameters."""
    global FEATURES_LAYOUT, N_FEATURES
    if obs_params is None:
        obs_params = {}
    # Determine actual dimensions from parameters or defaults
    max_tokens = obs_params.get('max_num_tokens', MAX_NUM_TOKENS)
    ext_dim = obs_params.get('ext_norm_dim', EXT_NORM_DIM)
    include_fear = obs_params.get('include_fear_greed', obs_params.get('use_dynamic_risk', False))
    # Define feature blocks
    layout = []
    # Bar-level features block
    layout.append({
        "name": "bar",
        "size": 3,
        "dtype": "float32",
        "clip": None,
        "scale": 1.0,
        "bias": 0.0,
        "source": "bars"
    })
    # Derived features block (4h return and volatility proxy for 4h timeframe)
    layout.append({
        "name": "derived",
        "size": 2,
        "dtype": "float32",
        "clip": DEFAULT_TANH_CLIP,
        "scale": 1.0,
        "bias": 0.0,
        "source": "derived"
    })
    # Technical indicators block
    layout.append({
        "name": "indicators",
        "size": 13,
        "dtype": "float32",
        "clip": None,
        "scale": 1.0,
        "bias": 0.0,
        "source": "indicators"
    })
    # Microstructure proxies block
    layout.append({
        "name": "microstructure",
        "size": 3,
        "dtype": "float32",
        "clip": DEFAULT_TANH_CLIP,
        "scale": 1.0,
        "bias": 0.0,
        "source": "micro"
    })
    # Agent state features block
    layout.append({
        "name": "agent",
        "size": 6,
        "dtype": "float32",
        "clip": DEFAULT_TANH_CLIP,
        "scale": 1.0,
        "bias": 0.0,
        "source": "agent"
    })
    # Metadata block (is_high_importance, time_since_event, risk_off_flag, fear_greed_value, fear_greed_indicator)
    # Always 5 features to match obs_builder implementation
    meta_size = 5
    layout.append({
        "name": "metadata",
        "size": meta_size,
        "dtype": "float32",
        "clip": DEFAULT_TANH_CLIP,
        "scale": 1.0,
        "bias": 0.0,
        "source": "meta"
    })
    # External normalized columns block (if any)
    if ext_dim and ext_dim > 0:
        layout.append({
            "name": "external",
            "size": ext_dim,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "external"
        })
    # Token metadata block (num_tokens_norm, token_id_norm)
    if max_tokens > 0:
        layout.append({
            "name": "token_meta",
            "size": 2,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "token_meta"
        })
    # Token one-hot block
    if max_tokens > 0:
        layout.append({
            "name": "token",
            "size": max_tokens,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "token"
        })
    FEATURES_LAYOUT = layout
    # Compute total feature vector length
    N_FEATURES = sum(block["size"] for block in layout)
    return FEATURES_LAYOUT

# Build default layout with default parameters on import
make_layout({})
