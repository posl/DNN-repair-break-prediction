def check_safety_prop(outputs, pid):
    """
    Check safety property of the model outputs.
    
    Parameters
    ------------------
    outputs: modelの出力. prob, pred, pred_min, pred_maxを含むdict.
    pid: safety propertyのid.

    Returns
    ------------------
    list of bool: 各サンプルがsafety propertyを満たすかどうか. 満たす場合にFalse. そうでなければTrue.
    """
    if pid == 2:
        is_safe = outputs["pred_max"] != 0
    elif pid == 7:
        is_safe = (outputs["pred_min"] != 3) & (outputs["pred_min"] != 4)
    elif pid == 8:
        is_safe = (outputs["pred_min"] == 0) | (outputs["pred_min"] == 1)
    elif pid == 9:
        is_safe = outputs["pred_min"] == 3
    else:
        raise ValueError(f"pid {pid} is not supported.")
    return ~is_safe