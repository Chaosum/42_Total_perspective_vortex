event_id = {
    'T0': 1,
    'T1': 2,
    'T2': 3
}
id_event = {
    1 : "T0",
    2 : "T1",
    3 : "T2"
}

T_meaning = {
    "T0" : {
        1 : "rest",
        2 : "rest",
        3 : "rest",
        4 : "rest",
        5 : "rest",
        6 : "rest",
        7 : "rest",
        8 : "rest",
        9 : "rest",
        10 : "rest",
        11 : "rest",
        12 : "rest",
        13 : "rest",
        14 : "rest",
    },
    "T1" : {
        1 : "",
        2 : "",
        3 : "left fist",
        4 : "left fist",
        5 : "both fists",
        6 : "both fists",
        7 : "left fist",
        8 : "left fist",
        9 : "both fists",
        10 : "both fists",
        11 : "left fist",
        12 : "left fist",
        13 : "both fists",
        14 : "both fists"
    },
    "T2" : {
        1 : "",
        2 : "",
        3 : "right fist",
        4 : "right fist",
        5 : "both feet",
        6 : "both feet",
        7 : "right fist",
        8 : "right fist",
        9 : "both feet",
        10 : "both feet",
        11 : "right fist",
        12 : "right fist",
        13 : "both feet",
        14 : "both feet"
    }

}
# useful_runs = {run_id: {"desc": "description", "task": <1-4>, "modal": "real"/"imagery", "T1": "label_T1", "T2": "label_T2"}}
# Mapping mis au propre à partir de la description officielle PhysioNet.
useful_runs = {
    # Baselines (runs 1–2) ne sont pas utilisés pour la classification

    # Task 1 : gauche/droite, mouvement RÉEL des mains
    3:  {"desc": "Task 1 - real left/right fist",      "task": 1, "modal": "real",    "T1": "left_fist",   "T2": "right_fist"},
    7:  {"desc": "Task 1 - real left/right fist",      "task": 1, "modal": "real",    "T1": "left_fist",   "T2": "right_fist"},
    11: {"desc": "Task 1 - real left/right fist",      "task": 1, "modal": "real",    "T1": "left_fist",   "T2": "right_fist"},

    # Task 2 : gauche/droite, mouvement IMAGINAIRE des mains
    4:  {"desc": "Task 2 - imagery left/right fist",   "task": 2, "modal": "imagery", "T1": "left_fist",   "T2": "right_fist"},
    8:  {"desc": "Task 2 - imagery left/right fist",   "task": 2, "modal": "imagery", "T1": "left_fist",   "T2": "right_fist"},
    12: {"desc": "Task 2 - imagery left/right fist",   "task": 2, "modal": "imagery", "T1": "left_fist",   "T2": "right_fist"},

    # Task 3 : mains vs pieds, mouvement RÉEL
    5:  {"desc": "Task 3 - real hands vs feet",        "task": 3, "modal": "real",    "T1": "both_fists",  "T2": "both_feet"},
    9:  {"desc": "Task 3 - real hands vs feet",        "task": 3, "modal": "real",    "T1": "both_fists",  "T2": "both_feet"},
    13: {"desc": "Task 3 - real hands vs feet",        "task": 3, "modal": "real",    "T1": "both_fists",  "T2": "both_feet"},

    # Task 4 : mains vs pieds, mouvement IMAGINAIRE
    6:  {"desc": "Task 4 - imagery hands vs feet",     "task": 4, "modal": "imagery", "T1": "both_fists",  "T2": "both_feet"},
    10: {"desc": "Task 4 - imagery hands vs feet",     "task": 4, "modal": "imagery", "T1": "both_fists",  "T2": "both_feet"},
    14: {"desc": "Task 4 - imagery hands vs feet",     "task": 4, "modal": "imagery", "T1": "both_fists",  "T2": "both_feet"},
}