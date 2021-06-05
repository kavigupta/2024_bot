import sys
import tqdm
from mapmaker.challenge import generate_challenge_maps

CHALLENGES = {
    1: ("By Population", lambda x: x["CVAP"]),
    2: (
        "Non-mainline prot",
        lambda x: x["evangelical"] + x["mormon"] + x["black_protestant"],
    ),
    3: ("2016 margin", lambda x: x["past_pres_partisanship"]),
    4: ("Bachelor %", lambda x: x["bachelor %"]),
    5: ("Number of Colleges", lambda x: x["college"]),
    6: ("White %", lambda x: x["white %"]),
}

challenges_to_generate = [int(x) for x in sys.argv[1:]]

for i in tqdm.tqdm(challenges_to_generate):
    generate_challenge_maps(i, *CHALLENGES[i])
