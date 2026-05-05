def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm

    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():  # temp for some ckpt
        from .QWen2_5 import _QWen_VL_Interface

        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif "Qwen3.5" in vlm_name:
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    elif "gemma-4" in vlm_name.lower() or "gemma4" in vlm_name.lower():
        from .Gemma4 import _Gemma4_VL_Interface

        return _Gemma4_VL_Interface(config)
    elif "internvl" in vlm_name.lower():
        # Covers InternVL3 / InternVL3.5 (HF format, e.g. OpenGVLab/InternVL3_5-2B-HF).
        # The non-HF OpenGVLab variants ship with `trust_remote_code` modeling files
        # and a different .chat() API — point your config at the *-HF checkpoints.
        from .InternVL import _InternVL3_Interface
    
        return _InternVL3_Interface(config)
    elif "florence" in vlm_name.lower():  # temp for some ckpt
        from .Florence2 import _Florence_Interface

        return _Florence_Interface(config)
    elif "cosmos-reason2" in vlm_name.lower():
        # Cosmos-Reason2 is architecturally Qwen3-VL (VLM), but implemented
        # in world_model/ for historical reasons. Import directly.
        from starVLA.model.modules.vlm.CosmosReason2 import _CosmosReason2_Interface

        return _CosmosReason2_Interface(config)
    
    elif "InternVL" in vlm_name or "internvl" in vlm_name.lower():
        from .InternVL3 import _InternVL3_Interface
        return _InternVL3_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")