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
useful_runs = {
    4:  {"desc": "imagine opening/closing left or right fist", "T1": "left_fist",   "T2": "right_fist"},
    6:  {"desc": "imagine opening/closing both fists or feet", "T1": "both_fists",  "T2": "both_feet"},
    7:  {"desc": "imagine left or right fist",                 "T1": "left_fist",   "T2": "right_fist"},
    8:  {"desc": "imagine left or right foot",                 "T1": "left_foot",   "T2": "right_foot"},
    9:  {"desc": "imagine fists vs feet",                      "T1": "both_fists",  "T2": "both_feet"},
    10: {"desc": "imagine left or right fist",                 "T1": "left_fist",   "T2": "right_fist"},
}