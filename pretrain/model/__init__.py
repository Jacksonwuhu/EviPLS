from .TemporalTransformer import TemporalTransformer


def build_model(args, model_name, clip_event_feats=None, clap_event_feats=None):
    model_dict = {
        'TemporalTransformer': TemporalTransformer
    }
    assert model_name in model_dict.keys(), f'invalid model_name: {model_name}'

    if model_name == 'TemporalTransformer':
        model = TemporalTransformer(
                    input_v_2d_dim = args.input_v_2d_dim,
                    input_v_3d_dim = args.input_v_3d_dim,
                    input_a_dim = args.input_a_dim,
                    hidden_dim = args.hidden_dim,
                    nhead = args.nhead,
                    ff_dim = args.ff_dim,
                    num_layers = args.num_layers,
                    dropout = args.dropout,
                    pre_norm = args.pre_norm,
                    clap_event_feats = clap_event_feats,
                    clip_event_feats = clip_event_feats
                )
    else:
        raise ValueError(f'The {model_name} model is not supported.')

    return model
