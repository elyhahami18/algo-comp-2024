import numpy as np
from typing import List, Tuple
# Helper functions to pre-process the scores list to set scores of invalid combinations to 0
def pre_process(scores: List[List[int]], gender_id: List[str], gender_pref: List[str]) -> List[List[int]]:
    n = len(scores)
    for i in range(n):
        for j in range(n):
            if not is_compatible(i, j, gender_id, gender_pref):
                scores[i][j] = 0
    return scores

# Helper function to address incompatible gender identity/preference combinations
def is_compatible(proposer, receiver, gender_id, gender_pref):
    # Since no clear directions were given, I assume that nonbinary can match to anybody
    preference_to_identity = {
        'Men': ['Male', 'Nonbinary'],
        'Women': ['Female', 'Nonbinary'],
        'Bisexual': ['Male', 'Female', 'Nonbinary']
    }

    proposer_gender_pref = gender_pref[proposer]
    receiver_gender_id = gender_id[receiver]

    # Check if the receiver's gender identity is in the list of identities that the proposer's preference finds acceptable
    return receiver_gender_id in preference_to_identity.get(proposer_gender_pref, [])

def better_proposal(new_proposer, current_proposer, top_choice, scores):
    return scores[new_proposer][top_choice] > scores[current_proposer][top_choice]

def next_best_choice(proposer, scores, proposers_choices):
    # Skip over choices that have already been made
    for choice in range(len(scores)):
        if choice not in proposers_choices[proposer]:
            proposers_choices[proposer].append(choice)
            return choice
    return None

def run_matching(scores: List[List], gender_id: List, gender_pref: List) -> List[Tuple]:

    # Preprocess scores for compatibility
    scores = pre_process(scores, gender_id, gender_pref)

    # Randomly split into proposers and acceptors
    n = len(scores)
    proposers = np.random.choice(n, n//2, replace=False)
    acceptors = list(set(range(n)) - set(proposers))

    # Initialize all proposers as free
    free_proposers = set(proposers)

    # Each acceptor keeps track of the best proposal they've received
    acceptor_proposals = {acceptor: None for acceptor in acceptors}

    proposers_choices = {proposer: [] for proposer in proposers}

    # Implement the Gale-Shapley algo that we went over in comp
    while free_proposers:
        for proposer in list(free_proposers):
            top_choice = next_best_choice(proposer, scores, proposers_choices)
            if top_choice is None:
                # If there are no more choices, remove the proposer from free_proposers
                free_proposers.remove(proposer)
                continue  # Skip to the next proposer

            current_proposer = acceptor_proposals.get(top_choice)  # Use .get to avoid KeyError
            if current_proposer is None or better_proposal(proposer, current_proposer, top_choice, scores):
                if current_proposer is not None:
                    free_proposers.add(current_proposer)
                free_proposers.remove(proposer)
                acceptor_proposals[top_choice] = proposer

    matches = [(proposer, acceptor) for acceptor, proposer in acceptor_proposals.items() if proposer is not None]
    return matches

if __name__ == "__main__":
    raw_scores = np.loadtxt('raw_scores.txt').tolist()
    genders = []
    with open('genders.txt', 'r') as file:
        for line in file:
            curr = line[:-1]
            genders.append(curr)

    gender_preferences = []
    with open('gender_preferences.txt', 'r') as file:
        for line in file:
            curr = line[:-1]
            gender_preferences.append(curr)

    gs_matches = run_matching(raw_scores, genders, gender_preferences)
    print(gs_matches)
    # the result of running the above print statement is '[(5, 0), (7, 1), (6, 3), (9, 4), (2, 2)]'
