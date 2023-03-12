import numpy as np
import itertools
import pickle
import random
import copy
import time

DATA_PATH = "votes.txt"
VERBOSE = True
INITIAL_COALITION_SIZE = 55
ATTEMPTS_PER_COALITION_SIZE = 75
GENERATE_ORDERS = True

# Generates a pickle file containing all possible elimination orders
def generate_elimination_orders():
    candidates = list(range(1,12))
    orders = list(itertools.permutations(candidates))
    
    with open('orders.pickle', 'wb') as file:
        pickle.dump(orders, file)

    print("INFO: Orders saved!")

# Loads the pickle file with all elimination orders
def get_elimination_orders():
    with open('orders.pickle', 'rb') as file:
        orders = pickle.load(file)

    print("INFO: Elimination orders have been loaded.")

    return orders

# Parses the votes file and returns list of votes
def get_profile(path=DATA_PATH):
    profile = []

    # Open file and remove whitespaces:
    with open(path) as file:
        lines = [line.rstrip() for line in file]
    
    # Remove redundant text:
    lines = lines[23:]

    for line in lines:
        # Extract number of voters that had this preference order:
        number = int(line.split(':')[0])

        # Only keep part after ":":
        vote = line.split(':')[1].strip()

        # Add votes in the form of singular voters:
        for i in range(number):
            profile.append(vote.split(','))

    if VERBOSE:
        print("INFO: Profile has been loaded.")

    return profile

# Counts the votes for a candidate in the profile
def count_votes(candidate, profile):
    votes = 0

    for vote in profile:
        # Check if vote has been emptied:
        if vote == None or len(vote) == 0:
            continue
        
        # Check if voter has a tie in their preference:
        if '{' in str(vote[0]):
            vote = str(vote[0]).replace('{', '')

            if candidate == vote:
                votes += 0.5
        elif '}' in str(vote[0]):
            vote = str(vote[0]).replace('}', '')

            if candidate == vote:
                votes += 0.5
        else:
            preference = int(vote[0])
        
            if candidate == preference:
                votes += 1

    return votes

# Determine STV winner
def STV(profile):
    STV_profile = copy.deepcopy(profile)
    no_winner = True
    candidates = list(range(1, 12))
    removed = []

    while no_winner:
        tally = np.zeros(11)

        # Highlight which candidates are eliminated:
        for lost in removed:
            tally[lost - 1] = float("inf")

        # Count votes for each candidate at current stage:
        for candidate in candidates:
            tally[candidate - 1] = count_votes(candidate, STV_profile)

        # Find and eliminate the loser:
        loser = np.argmin(tally) + 1
        STV_profile = eliminate_candidate(loser, STV_profile)
        candidates.remove(loser)
        removed.append(loser)

        # Check if winner is found:
        if len(candidates) == 1:
            winner = candidates[0]
            no_winner = False

    return winner

# Returns a new profile where the specified candidate has been removed
def eliminate_candidate(candidate, profile):
    new_profile = []

    # Remove specified candidate from the votes:
    for vote in profile:
        if str(candidate) in vote:
            vote.remove(str(candidate))

        new_profile.append(vote)

    return new_profile

# Returns struct with profile containing k random emptied votes and the corresponding markings
def get_profile_with_k_empty(profile, k):
    markings = np.ones(len(profile))
    emptied = random.sample(range(len(profile)), k)

    # Empty votes and mark as unused:
    for vote in emptied:
        profile[vote] = []
        markings[vote] = 0

    if VERBOSE:
        print("INFO: {} votes in profile emptied.".format(k))

    return {
        "profile": profile,
        "markings": markings
    }

# Main algorithm which finds a possible manipulation:
def manipulate(order, profile, k):
    starting_point = get_profile_with_k_empty(profile, k)
    profile = starting_point["profile"]
    profile = copy.deepcopy(profile)
    original_profile = copy.deepcopy(profile)
    markings = starting_point["markings"]

    # Record which votes are unused:
    unmarked = []
    for i in range(len(markings)):
        if markings[i] == 0:
            unmarked.append(i)

    order_works = True
    count = 0

    if VERBOSE:
        print("INFO: The manipulation algorithm has started.")

    count += 1
    order_works = True

    if VERBOSE:
        print("INFO: Testing order {}".format(count))

    for i in range(1, (len(order) - 1) + 1):
        if not order_works:
            break
        # Tally the current votes:
        votes = [count_votes(cand, profile) for cand in order]

        for j in range(2, len(order) + 1):
            if not order_works:
                break

            while votes[j - 1] < votes[i - 1]:
                if 0 in markings:
                    # Find unused vote:
                    vote_number = np.where(markings == 0)[0][0]
                    markings[vote_number] = 1
                    current_vote = profile[vote_number]

                    # Move cj to highest position:
                    if str(j) in current_vote:
                        current_vote.remove(str(j))
                        current_vote.insert(0, str(j))
                    else:
                        current_vote.insert(0, str(j))
                    
                    profile[vote_number] = current_vote
                else:
                    order_works = False
                    break

        # for k in range(len(profile)):
        #     if markings[k] == 1:
        #         if not len(profile[k]) == 0:
        #             if profile[k][-1] == i:
        #                 markings[k] = 0

        #                 if VERBOSE:
        #                     print("INFO: Vote {} has been unmarked.".format(k))
        
        # Eliminate ci:
        profile = eliminate_candidate(i, profile)
    
    # Add coalition to the original profile:
    for index in unmarked:
        str_order = [str(candidate) for candidate in order]
        original_profile[index] = list(reversed(str_order))
    
    if VERBOSE:
        print("INFO: A successful manipulation was found.")

    return original_profile

# Checks the number of votes in both profiles as well as how many votes are different
def compare_profiles(path_original, path_manipulated):
    with open(path_original, 'r') as file:
        original = [line.rstrip() for line in file]
    with open(path_manipulated, 'r') as file:
        manipulated = [line.rstrip() for line in file]

    print("\nINFO: Original profile has {} votes.".format(len(original)))

    print("INFO: Manipulated profile has {} votes.".format(len(manipulated)))

    # Count the number of lines where the two profiles differ:
    changed = 0
    for i in range(len(original)):
        if original[i] != manipulated[i]:
            changed += 1
    
    print("INFO: There were {} changed votes".format(changed))

# Runs algorithm until a different winner than the original winner is found
def find_other_winner(coalition_size, num_attempts, orders):
    other_winner = False
    count = 0

    while other_winner == False and count < num_attempts:
        profile = get_profile()
        original_winner = STV(profile)
        new_profile = copy.deepcopy(profile)
        
        # Pick a random potential elimination order:
        order = random.choice(orders)

        # Run algorithm:
        result = manipulate(order=order, profile=new_profile, k=coalition_size)
        count += 1

        if len(result) == 0:
            continue
        else:
            winner = STV(result)
            
            print("INFO: Testing manipulation {}".format(count), end="\r")

            if VERBOSE:
                print("INFO: The winner after manipulation {} is candidate {}".format(count, winner))

            if winner != original_winner:
                other_winner = True

                if VERBOSE:
                    print("INFO: Another winner was found. The winner is candidate {}\n".format(winner))

    if other_winner:
        with open("manipulated.txt", 'w') as file:
            for vote in result:
                file.write("%s\n" % str(vote))

    return other_winner

def main():
    start = time.time()

    if GENERATE_ORDERS:
        generate_elimination_orders()
    
    coalition_size = INITIAL_COALITION_SIZE
    num_attempts = ATTEMPTS_PER_COALITION_SIZE
    profile = get_profile()
    orders = get_elimination_orders()
    manipulation_found = False

    # Run algorithm until a manipulation is found with the current coalition size:
    while not manipulation_found:
        print("INFO: Testing with coalition size {}:".format(coalition_size))

        manipulation_found = find_other_winner(
            coalition_size=coalition_size,
            num_attempts=num_attempts,
            orders=orders
            )
        
        coalition_size += 1

    end = time.time()

    print("\nINFO: Manipulation found with a coalition size of {}".format(coalition_size - 1))

    with open("original.txt", 'w') as file:
        for vote in profile:
            file.write("%s\n" % str(vote))

    compare_profiles(path_original="original.txt", path_manipulated="manipulated.txt")

    print("INFO: Time taken: {} seconds".format(end - start))

if __name__ == "__main__":
    main()