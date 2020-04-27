
class State():
    """
    A class representing the election results for a given state. 
    Assumes there are no ties between dem and gop votes. The party with a 
    majority of votes receives all the Electoral College (EC) votes for 
    the given state.
    """
    def __init__(self, name, dem, gop, ec):
        """
        Parameters:
        name - the 2 letter abbreviation of a state
        dem - number of Democrat votes cast
        gop - number of Republican votes cast
        ec - number of EC votes a state has 

        Attributes:
        self.name - str, the 2 letter abbreviation of a state
        self.winner - str, the winner of the state, "dem" or "gop"
        self.margin - int, difference in votes cast between the two parties, a positive number
        self.ec - int, number of EC votes a state has
        """
        self.name = name
        self.ec = ec
        self.margin = abs(dem-gop)
        if dem>gop:
            self.winner = "dem"
        if gop>dem: 
            self.winner = "gop"

    def get_name(self):
        """
        Returns:
        str, the 2 letter abbreviation of the state  
        """
        return self.name

    def get_num_ecvotes(self):
        """
        Returns:
        int, the number of EC votes the state has 
        """
        return self.ec

    def get_margin(self):
        """
        Returns: 
        int, difference in votes cast between the two parties, a positive number
        """
        return self.margin

    def get_winner(self):
        """
        Returns:
        str, the winner of the state, "dem" or "gop"
        """
        return self.winner

    def __str__(self):
        """
        Returns:
        str, representation of this state in the following format,
        "In <state>, <ec> EC votes were won by <winner> by a <margin> vote margin."
        """
        statement = "In " + self.get_name() + ", " + str(self.get_num_ecvotes()) +\
        " EC votes were won by " + self.get_winner() + " by a " + str(self.get_margin())\
        + " vote margin."
        return statement
        
    def __eq__(self, other):
        """
        Determines if two State instances are the same.
        They are the same if they have the same state name, winner, margin and ec votes.
        Be sure to check for instance type equality as well! 

        Param:
        other - State object to compare against  

        Returns:
        bool, True if the two states are the same, False otherwise
        """
        if type(self)==type(other)\
        and self.get_name()==other.get_name()\
        and self.get_winner()==other.get_winner()\
        and self.get_margin()==other.get_margin()\
        and self.get_num_ecvotes()==other.get_num_ecvotes():
            return True
        else:
            return False

def load_election_results(filename):
    """
    Reads the contents of a file, with data given in the following tab-delimited format,
    State   Democrat_votes    Republican_votes    EC_votes 

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a list of State instances
    """
    L = []
    f = open(filename)
    for line in f:
        newline2 = line.split("\t")
        if newline2[1]=='Democrat':
            pass
        else:
            L.append(State(newline2[0], int(newline2[1]), int(newline2[2]), int(newline2[3])))
    return L

def find_winner(election):
    """
    Finds the winner of the election based on who has the most amount of EC votes.
    Note: In this simplified representation, all of EC votes from a state go
    to the party with the majority vote.

    Parameters:
    election - a list of State instances 

    Returns:
    a tuple, (winner, loser) of the election i.e. ('dem', 'gop') if Democrats won, else ('gop', 'dem')
    """
    dem_electoral_votes = 0
    gop_electoral_votes = 0
    for individual_state in election:
        electoral_votes = individual_state.get_num_ecvotes()
        state_winner = individual_state.get_winner()
        if state_winner == "dem":
            dem_electoral_votes += electoral_votes
        elif state_winner == "gop":
            gop_electoral_votes += electoral_votes
    if gop_electoral_votes>dem_electoral_votes:
        return ('gop','dem')
    if gop_electoral_votes<dem_electoral_votes:
        return ('dem','gop')

def winner_states(election):
    """
    Finds the list of States that were won by the winning candidate (lost by the losing candidate).
    
    Parameters:
    election - a list of State instances 

    Returns:
    A list of State instances won by the winning candidate
    """
    Winners_States = []
    winner_loser = find_winner(election)
    president = winner_loser[0]
    for individual_state in election:
        state_winner = individual_state.get_winner()
        if state_winner == president:
            Winners_States.append(individual_state)
    return Winners_States

def ec_votes_reqd(election, total=538):
    """
    Finds the number of additional EC votes required by the loser to change election outcome.
    Note: A party wins when they earn half the total number of EC votes plus 1.

    Parameters:
    election - a list of State instances 
    total - total possible number of EC votes

    Returns:
    int, number of additional EC votes required by the loser to change the election outcome
    """
    Min_EC = 538/2+1
    Losers_States = []
    Losers_Total = 0
    winner_loser = find_winner(election)
    not_president = winner_loser[1]
    for individual_state in election:
        state_winner = individual_state.get_winner()
        if state_winner == not_president:
            Losers_States.append(individual_state)
    for state in Losers_States:
        electoral_votes = state.get_num_ecvotes()
        Losers_Total += electoral_votes
    return Min_EC-Losers_Total

                                                 
def greedy_election(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states. First chooses the states with the smallest 
    win margin, i.e. state that was won by the smallest difference in number of voters. 
    Continues to choose other states up until it meets or exceeds the ec_votes_needed. 
    Should only return states that were originally won by the winner in the election.

    Parameters:
    winner_states - a list of State instances that were won by the winner 
    ec_votes_needed - int, number of EC votes needed to change the election outcome
    
    Returns:
    A list of State instances such that the election outcome would change if additional
    voters relocated to those states (also can be referred to as our swing states)
    The empty list, if no possible swing states
    """
    ec_votes_switching = 0
    swing_states = []
    Sorted_winner_states = sorted(winner_states, key = State.get_margin)
    for individual_state in Sorted_winner_states:
        if ec_votes_needed>ec_votes_switching:
            electoral_votes = individual_state.get_num_ecvotes()
            ec_votes_switching += electoral_votes
            swing_states.append(individual_state)
    return swing_states
              
def dp_move_max_voters(winner_states, ec_votes, memo = None):
    """
    Finds the largest number of voters needed to relocate to get at most ec_votes
    for the election loser. 

    Parameters:
    winner_states - a list of State instances that were won by the winner 
    ec_votes - int, the maximum number of EC votes 
    memo - dictionary, an OPTIONAL parameter for memoization (don't delete!).
    Note: If you decide to use the memo make sure to override the default value when it's first called.

    Returns:
    A list of State instances such that the maximum number of voters need to be relocated
    to these states in order to get at most ec_votes 
    The empty list, if every state has a # EC votes greater than ec_votes
    """
    # Maximizing the number of voters you can move without going above a certain number 
    # of electoral votes. 
    # Inspired by Lecture 2 Code
    def helper_function(winner_states, ec_votes, memo = {}): 
        if (len(winner_states), ec_votes) in memo:
            result = memo[(len(winner_states), ec_votes)]
        elif winner_states == [] or ec_votes == 0:
            result = (0, ())
        elif winner_states[0].get_num_ecvotes() > ec_votes:
            #Explore right branch only
            result = helper_function(winner_states[1:], ec_votes, memo)
        else:
            nextItem = winner_states[0]
            #Explore left branch
            withVal, withToTake =\
                     helper_function(winner_states[1:],
                                ec_votes - nextItem.get_num_ecvotes(), memo)
            withVal += nextItem.get_margin()
            #Explore right branch
            withoutVal, withoutToTake = helper_function(winner_states[1:],
                                                    ec_votes, memo)
            #Choose better branch
            if withVal > withoutVal:
                result = (withVal, withToTake + (nextItem,))
            else:
                result = (withoutVal, withoutToTake)
        memo[(len(winner_states), ec_votes)] = result
        return result
    return list(helper_function(winner_states, ec_votes)[1])
    
    
def move_min_voters(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states. Should minimize the number of voters being relocated. 
    Only return states that were originally won by the winner (lost by the loser)
    of the election.


    Parameters:
    winner_states - a list of State instances that were won by the winner 
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    A list of State instances such that the election outcome would change if additional
    voters relocated to those states (also can be referred to as our swing states)
    The empty list, if no possible swing states
    """
    # Minimizing the number of voters you can move to go above a certain number 
    # of electoral votes. 
    winner_ec_votes = 0
    swing_states = []
    for individual_state in winner_states:
        winner_ec_votes+= individual_state.get_num_ecvotes()
    complement = dp_move_max_voters(winner_states, 268)
    for state in winner_states:
        if state not in complement:
            swing_states.append(state)
    return swing_states
    

def flip_election(election, swing_states):
    """
    Finds a way to shuffle voters in order to flip an election outcome. 
    Moves voters from states that were won by the losing candidate (any state not in winner_states), 
    to each of the states in swing_states. 
    To win a swing state, must move (margin + 1) new voters into that state. Any state that voters are
    moved from should still be won by the loser even after voters are moved.
    
    Also finds the number of EC votes gained by this rearrangement, as well as the minimum number of 
    voters that need to be moved.

    Parameters:
    election - a list of State instances representing the election 
    swing_states - a list of State instances where people need to move to flip the election outcome 
                   (result of move_min_voters or greedy_election)
    
    Return:
    A tuple that has 3 elements in the following order:
        - a dictionary with the following (key, value) mapping: 
            - Key: a 2 element tuple, (from_state, to_state), the 2 letter abbreviation of the State 
            - Value: int, number of people that are being moved 
        - an int, the total number of EC votes gained by moving the voters 
        - an int, the total number of voters moved 
    None, if it is not possible to sway the election
    """
    dict_flip = {}
    votes_gained = sum(n.get_num_ecvotes() for n in swing_states)
    total_votes_moved = sum(n.get_margin() + 1 for n in swing_states) # +1 because you wanna win
    winner_copy = winner_states(election)
    loser_states = []
    for state in election:
        if state not in winner_copy:
            loser_states.append(state)   
    loser_copy = sorted(loser_states, key=State.get_margin, reverse=True)
    swing_copy = sorted(swing_states, key=State.get_margin, reverse=True)
    loser_margins = [n.get_margin() for n in loser_copy]
    swing_margins = [n.get_margin() for n in swing_copy]
    if sum(loser_margins) - len(loser_margins) < sum(swing_margins) + len(swing_margins):
        return None
    l = 0
    s = 0
    while swing_margins[len(swing_margins)-1] != -1: # Checks if last swing state has been reduced
        if l < len(loser_margins):
            if loser_margins[l] > swing_margins[s]:
                votes_transfered = min(loser_margins[l]-1, swing_margins[s] + 1)
                dict_flip[(loser_copy[l].get_name(), swing_copy[s].get_name())] = votes_transfered
                loser_margins[l] = loser_margins[l] - swing_margins[s] - 1
                swing_margins[s] = -1
                s += 1
        if s < len(swing_margins):
            if loser_margins[l] < swing_margins[s]:
                votes_transfered = min(loser_margins[l]-1, swing_margins[s])
                dict_flip[(loser_copy[l].get_name(), swing_copy[s].get_name())] = votes_transfered
                swing_margins[s] = swing_margins[s] - loser_margins[l] + 1
                loser_margins[l] = 1
                l += 1
    return (dict_flip, votes_gained, total_votes_moved)

if __name__ == "__main__":
    pass

    year = 2012 # Update the election year to debug
    election = load_election_results("%s_results.txt" % year) 

    winner, loser = find_winner(election)
    won_states = winner_states(election)
    names_won_states = [state.get_name() for state in won_states]
    ec_votes_needed = ec_votes_reqd(election)
    print("Winner:", winner, "\nLoser:", loser)
    print("EC votes needed:",ec_votes_needed)
    print("States won by the winner: ", names_won_states, "\n")

    print("greedy_election")
    greedy_swing = greedy_election(won_states, ec_votes_needed)
    names_greedy_swing = [state.get_name() for state in greedy_swing]
    voters_greedy = sum([state.get_margin()+1 for state in greedy_swing])
    ecvotes_greedy = sum([state.get_num_ecvotes() for state in greedy_swing])
    print("Greedy swing states results:", names_greedy_swing)
    print("Greedy voters displaced:", voters_greedy, "for a total of", ecvotes_greedy, "Electoral College votes.", "\n")

    print("dp_move_max_voters")
    total_lost = sum(state.get_num_ecvotes() for state in won_states)
    move_max = dp_move_max_voters(won_states, total_lost-ec_votes_needed)
    max_states_names = [state.get_name() for state in move_max]
    max_voters_displaced = sum([state.get_margin()+1 for state in move_max])
    max_ec_votes = sum([state.get_num_ecvotes() for state in move_max])
    print("States with the largest margins:", max_states_names)
    print("Max voters displaced:", max_voters_displaced, "for a total of", max_ec_votes, "Electoral College votes.", "\n")

    print("move_min_voters")
    swing_states = move_min_voters(won_states, ec_votes_needed)
    swing_state_names = [state.get_name() for state in swing_states]
    min_voters = sum([state.get_margin()+1 for state in swing_states])
    swing_ec_votes = sum([state.get_num_ecvotes() for state in swing_states])
    print("Complementary knapsack swing states results:", swing_state_names)
    print("Min voters displaced:", min_voters, "for a total of", swing_ec_votes, "Electoral College votes. \n")

    print("flip_election")
    flipped_election = flip_election(election, swing_states)
    print("Flip election mapping:", flipped_election)