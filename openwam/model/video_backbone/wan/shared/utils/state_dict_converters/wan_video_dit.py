def WanVideoDiTStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.split(".")[0] in ["pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"]:
            continue
        name_ = name
        if name_.startswith("model."):
            name_ = name_[len("model.") :]
        state_dict_[name_] = state_dict[name]
    return state_dict_
