"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import logging

# Setting logging level and config
logging.basicConfig(format='%(asctime)s: %(message)s')
logging.getLogger().setLevel(logging.ERROR)

depth_cumul = 0
number_of_calls = 0

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

def progressive_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    board_blank_space_ratio = len(game.get_blank_spaces())/(game.width*game.height)
    logging.info("progressive_score: Blank space ratio: %d."%board_blank_space_ratio)
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    # try to minimize oponent moves when the board is umpty 
    # and turn to defense mode as the board gets full
    return float(own_moves*(1-board_blank_space_ratio) - opp_moves*(board_blank_space_ratio))

def agressive_score(game, player):
    """Rewards using one of the opponent's available moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    diff = tuple(abs(x-y) for x,y in zip(game.get_player_location(player),game.get_player_location(game.get_opponent(player))))
    
    if diff in ((1,2),(2,1)):
        # The player is standing on one of the opponent's moves
        return 1
    # Else maximize moves
    return len(game.get_legal_moves(player)) - 8


def as_deep_as_possible_score(game, player):
    """Rewards using one of the opponent's available moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    return -len(game.get_blank_spaces())

def open_2nd_moves_score(game, player):
    """Maximizes second moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    

    r, c = game.get_player_location(player)

    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1),
                  (0, -2),  (0, 2), (2, 0),  (-2, 0)]

    second_moves = [(r+dr,c+dc) for dr, dc in directions if (r+dr, c+dc) in game.get_blank_spaces()]
    
    return  len(second_moves)*(len(game.get_legal_moves(player))>0) + len(game.get_legal_moves(player))

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=20., name='no name'):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.name = name

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        global depth_cumul
        global number_of_calls
        
        number_of_calls += 1

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        logging.info("get_move: Called with method: %s."%self.method)
        
        if len(legal_moves) > 0:
            # Prepare a random move to be returned in case we run out of time
            move = legal_moves[random.randint(0,len(legal_moves)-1)]
        else:
            logging.info("get_move: No legal move available.")
            return (-1,-1)
        
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            # Start with depth = 1 
            depth = 1

            while self.time_left() > self.TIMER_THRESHOLD:
                
                logging.info("get_move: Working on depth: %d."%depth)

                if self.method == 'minimax':
                    s, move = self.minimax(game, depth)
                    
                elif self.method == 'alphabeta':
                    s, move = self.alphabeta(game, depth)
                  
                # We found a winning move no need to search deeper
                if s == float("inf") or s == float('-inf'):
                    logging.warning("Player %s: Depth reached : %d. With score: %s",self.name,depth,s)
                    break
                # Go to next depth when iterative search is activated
                elif self.iterative:
                    depth += 1
                    logging.warning("Player %s: Depth reached : %d. With score: %s",self.name,depth,s)
                else:
                    break 
                
                if depth > 100:
                    input(s)
                    logging.getLogger().setLevel(logging.WARNING)
                
        except Timeout:
            logging.warning("Player %s: Max depth reached : %d.",self.name,depth)
            pass

        depth_cumul += depth
        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        # Go back if timeout is close to the threshold
        if self.time_left() < self.TIMER_THRESHOLD:
            logging.info("minimax: Timeout reached.")
            raise Timeout()
        
        # Test cases where search should stop
        if depth <= 0 or len(game.get_legal_moves()) == 0:
            return self.score(game, self), (-1,-1)

        # Expand to the next level of the tree
        next_level = [((self.minimax(game.forecast_move(m), depth-1, not maximizing_player))[0],m) for m in game.get_legal_moves()]

        # Return max or min
        if maximizing_player:
            return max(next_level)
        else:
            return min(next_level)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if maximizing_player:
            return self.max_val(game, depth, alpha, beta)
        else:
            return self.min_val(game, depth, alpha, beta)
    
    def max_val(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        # Go back if timeout is close to the threshold
        if self.time_left() < self.TIMER_THRESHOLD:
            logging.info("max_val: Timeout reached.")
            raise Timeout()
        
        # Test cases where search should stop
        if depth <= 0  or len(game.get_legal_moves()) == 0:
            logging.info("max_val: Search ended.")
            return self.score(game, self), game.__last_player_move__[self]
        
        v = float("-inf")
        move = (-1, -1)
        for m in game.get_legal_moves():
            v, move = max ((v,move), (self.min_val(game.forecast_move(m), depth-1, alpha, beta)[0],m))
            if v >= beta :
                return v, move
            alpha = max(alpha, v)
        return v, move
    
    def min_val(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        
        # Go back if timeout is close to the threshold
        if self.time_left() < self.TIMER_THRESHOLD:
            logging.info("min_val: Timeout reached.")
            raise Timeout()
        
        # Test cases where search should stop
        if depth <= 0  or len(game.get_legal_moves()) == 0: 
            logging.info("min_val: Search ended.")
            return self.score(game, self), game.__last_player_move__[game.get_opponent(self)]
                
        v = float("inf")
        move = (-1, -1)
        for m in game.get_legal_moves():
            v, move = min ((v, move), (self.max_val(game.forecast_move(m), depth-1, alpha, beta)[0],m))
            if v <= alpha :
                return v, m
            beta = min(beta, v) 
        return v, move

            
            
            
            