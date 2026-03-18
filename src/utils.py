channels = ["C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6"]
alternative_channels = ["FC3", "FCZ", "FC4", "C3", "C1", "CZ", "C2", "C4"]
csp_params = {
    "n_components": 4
}

# raw = mne.io.read_raw_edf(r[1], preload=True)
# if raw.info['sfreq'] != 160.0:
#     raw.resample(sfreq=160.0)
# mne.datasets.eegbci.standardize(raw)
# raw.set_montage("standard_1005")

# tmin = -.500  # start of each epoch (in sec)
#     tmax = 1.000  # end of each epoch (in sec)

# ex["epochs"] = mne.concatenate_epochs(ex["epochs"])
# ex["epochs"] = balance_classes(ex["epochs"])

#             ex['y'] = ex["epochs"].events[:, -1]
# super epochs qui sont ma moyenne de N epoches (N = 30) pour réduire le bruit
# ex["X_avg"], ex["y_avg"] = average_over_epochs(
#     ex["epochs"],
#     ex["y"],
#     event_id
# )

    #         dump_model(ex["clf"], ex["name"], amount_of_subjects)

    # ex["crossval_scores"] = cross_val_score(
    #     ex["clf"], ex["X_avg"], ex["y_avg"], cv=cv,
    #     error_score='raise')


experiments = [
    {
        "name": "Left_right_fist",
        "description": "open and close left or right fist",
        "runs": [3, 7, 11],
        "mapping": {
            0: "Rest",
            1: "Left fist",
            2: "Right fist"
        },
    },
    {
        "name": "Imagine_left_right_fist",
        "description": "imagine opening and closing left or right fist",
        "runs": [4, 8, 12],
        "mapping": {
            0: "Rest",
            1: "Imagine left fist",
            2: "Imagine right fist"
        },
    },
    {
        "name": "Fists_feet",
        "description": "open and close both fists or both feet",
        "runs": [5, 9, 13],
        "mapping": {
            0: "Rest",
            1: "Both fists",
            2: "Both feet"
        },
    },
    {
        "name": "Imagine_fists_feet",
        "description": "imagine opening and closing both fists or both feet",
        "runs": [6, 10, 14],
        "mapping": {
            0: "Rest",
            1: "Imagine both fists",
            2: "Imagine both feet"
        },
    },
    {
        "name": "Movement_of_fists",
        "description": "movement (real or imagined) of fists",
        "runs": [3, 7, 11, 4, 8, 12],
        "mapping": {
            0: "Rest",
            1: "Left fist",
            2: "Right fist"
        },
    },
    {
        "name": "Movement_fists_feet",
        "description": "movement (real or imagined) of fists or feet",
        "runs": [5, 9, 13, 6, 10, 14],
        "mapping": {
            0: "Rest",
            1: "Both fists",
            2: "Both feet"
        },
    }
]