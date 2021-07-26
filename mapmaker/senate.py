class_1_and_3 = {
    "WA",
    "ND",
    "CA",
    "NV",
    "UT",
    "AZ",
    "HI",
    "MO",
    "WI",
    "IN",
    "OH",
    "PA",
    "NY",
    "VT",
    "CT",
    "MD",
    "FL",
}

class_2_and_3 = {
    "AK",
    "AL",
    "OR",
    "ID",
    "CO",
    "SD",
    "KS",
    "OK",
    "IA",
    "AR",
    "LA",
    "IL",
    "KY",
    "SC",
    "GA",
    "NC",
    "NH",
    "NE",
}

class_1_and_2 = {
    "MT",
    "UT",
    "NM",
    "WY",
    "TX",
    "MN",
    "MI",
    "TN",
    "MS",
    "WV",
    "VA",
    "NJ",
    "RI",
    "MA",
    "ME",
    "DE",
}

senate_2022 = class_1_and_3 | class_2_and_3

r_incumbents_up_2022 = {
    "ID",
    "UT",
    "ND",
    "SD",
    "NE",
    "OK",
    "IA",
    "AR",
    "LA",
    "WI",
    "IN",
    "KY",
    "FL",
    "SC",
    "AK",
    "KS",
}

r_open_2022 = {
    "AL",
    "MO",
    "NC",
    "OH",
    "PA",
}

d_incumbents_up_2022 = {
    "CA",
    "OR",
    "WA",
    "NV",
    "AZ",
    "CO",
    "IL",
    "GA",
    "MD",
    "NY",
    "CT",
    "VT",
    "NH",
    "HI",
}

d_open_2022 = set()

r_lock_2022 = 50 - len(r_open_2022) - len(r_incumbents_up_2022)
d_lock_2022 = 50 - len(d_open_2022) - len(d_incumbents_up_2022)

assert (
    r_incumbents_up_2022 | d_incumbents_up_2022 | r_open_2022 | d_open_2022
) == senate_2022
