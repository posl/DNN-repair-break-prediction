# utility function(s)
def setWeights(model, weight_list):
    """
    Set the weights of the model to the given list of weights.
    """
    for weight, (name, v) in zip(weight_list, model.named_parameters()):
        attrs = name.split(".")
        obj = model
        for attr in attrs:
            obj = getattr(obj, attr)
        obj.data = weight


def AdjustWeights(baseWeights, corrDiff, incorrDiff, a, b, strategy="both-org", lr=1e-3):
    if "org" in strategy:
        sign = 1
    else:
        sign = -1

    p_corr, p_incorr = a / (a + b), b / (a + b)

    if "both" in strategy:
        return [
            b_w + sign * lr * (p_corr * cD - p_incorr * iD) for b_w, cD, iD in zip(baseWeights, corrDiff, incorrDiff)
        ]
    elif "corr" in strategy:
        return [b_w + sign * lr * p_corr * cD for b_w, cD in zip(baseWeights, corrDiff)]
    elif "incorr" in strategy:
        return [b_w - sign * lr * p_incorr * iD for b_w, iD in zip(baseWeights, incorrDiff)]
    else:
        raise ValueError(f"Unrecognized strategy {strategy}")
