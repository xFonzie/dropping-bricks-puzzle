import streamlit as st
from collections import defaultdict
from typing import List, Tuple, Set
from random import randint

class Game:
    def __init__(self, H, D) -> None:
        """
        Initial game settings.

        args:
            H: Height of the tower.
            D: Upper bound of allowed number of drops.
        """
        self.H = H
        self.D = D

    def _initialize(self) -> None:
        """
        Overwrite winning positions to initial state.
        """
        self.winning = {
            (i, i, d, k) for d in range(self.D + 1) for k in range(3) for i in range(1, self.H + 1) 
        }
        
    def search_winning_positions(self, optimized=True) -> Set[int]:
        """
        Finds the fix-point of set of winning positions for this game.

        If optimized, performs serach on the game tree in topological order, which considerably saves computation time
        while not losing any winning state in the game.
        Complexitity optimized: O(H^3 * D)

        If not optimized, checks every possible position in the game if it is winning untill fix-point reached.
        Complexity not optimizd: O(H^5 * D^2)

        kwargs:
            optimized: True to use topological sort, else use brute-force algorithm.
        """

        self._initialize()

        if optimized:
            def _topological_sort(n) -> List[Tuple[int]]:
                """
                Performs the topological sort of the graph based on game tree.
                This algorithm utilizes depth-first-search and saves the order of positions. 
                Complexity: O(n^2) 

                args:
                    n:  upper bound of initial 
                """
                
                used = defaultdict(bool)
                ans = []

                def _dfs(v):
                    x, y = v
                    used[v] = True

                    for m in range(x + 1, y + 1):
                        to = (x, m - 1)
                        if not used[to]:
                            _dfs(to)
                        to = (m, y)
                        if not used[to]:
                            _dfs(to)

                    ans.append(v)

                _dfs((1, n))
                return ans
            
            order = _topological_sort(self.H)
            for target in order: # H^2
                x, y = target
                for d in range(self.D): # D
                    for k in range(1, 3): # const
                        for m in range(x + 1, y + 1): # H
                            if (x, m - 1, d + 1, k - 1) in self.winning and\
                                (m, y, d + 1, k) in self.winning:
                                self.winning.add((x, y, d, k))

        if not optimized:
            current_winning_size = -1

            while current_winning_size != len(self.winning): # H^2*D    at most according to Corollary from Knaster–Tarski fixed point theorem
                current_winning_size = len(self.winning)

                for x in range(1, self.H + 1): # H
                    for y in range(x, self.H + 1): # H
                        for d in range(self.D): # D
                            for k in range(1, 3): # const
                                for m in range(x + 1, y + 1): # H
                                    if (x, m - 1, d + 1, k - 1) in self.winning and\
                                        (m, y, d + 1, k) in self.winning:
                                        self.winning.add((x, y, d, k))
        return self.winning
    
    def update_parameters(self, H, D):
        """
        Update parameters for the game.

        args:
            H: Height of the tower.
            D: Upper bound of allowed number of drops.
        """
        self.H = H
        self.D = D
        self._initialize()
    
    def play_console(self):
        """
        Plays the game using console input-output.
        """
        self.search_winning_positions()
        current = (1, self.H, 0, 2)
        x, y, d, k = current
        num_drops = 0
        while x < y:
            for m in range(x + 1, y + 1):
                if (x, m - 1, d + 1, k - 1) in self.winning and\
                    (m, y, d + 1, k) in self.winning:
                    respond = input(f'Computer drops at {m} floor. Please, reply ("safe" or "broken"):\n')
                    num_drops += 1
                    if respond.startswith('s'):
                        current = (m, y, d + 1, k)
                    else:
                        current = (x, m - 1, d + 1, k - 1)
                    break
            x, y, d, k = current

        print(f'The answer is {x}. The game ended in {num_drops} drops.')
    
    def has_stategy(self):
        """
        Check if the game has winning strategy
        """
        self.search_winning_positions()
        return (1, self.H, 0, 2) in self.winning
            

class GUIGame(Game):
    """
    Class that uses streamlit functionality
    to help integrate the game into website.
    """
    def __init__(self, H, D) -> None:
        super().__init__(H, D)
        self.state = (1, self.H, 0, 2)
        self.num_drops = 0

    def _initialize(self) -> None:
        super()._initialize()
        self.state = (1, self.H, 0, 2)
        self.num_drops = 0
        
    def step_ai(self) -> None:
        x, y, d, k = self.state

        for m in range(x + 1, y + 1)[::-1]:
            if (x, m - 1, d + 1, k - 1) in self.winning and\
                    (m, y, d + 1, k) in self.winning:
                self.m = m
                self.num_drops += 1
                return m
    
    def step_random(self) -> None:
        x, y, d, k = self.state

        m = randint(x + 1, y)

        self.m = m
        self.num_drops += 1
        return m
    
    def safe_step(self) -> None:
        # set next state
        x, y, d, k = self.state
        self.state = (self.m, y, d + 1, k)

        # check if game ends
        if self.m == y:
            st.session_state.complete = (y, self.num_drops, True)
            st.session_state.webstate = 'result'
            return
        
        if d >= self.D or k <= 0:
            st.session_state.complete = (self.m, self.num_drops, False)
            st.session_state.webstate = 'result'
            return
    
    def broken_step(self) -> None:
        # set next state
        x, y, d, k = self.state
        self.state = (x, self.m - 1, d + 1, k - 1)
        
        # check if game ends
        if x == self.m - 1:
            st.session_state.complete = (x, self.num_drops, True)
            st.session_state.webstate = 'result'
            return

        if d >= self.D or k <= 1:
            st.session_state.complete = (x, self.num_drops, False)
            st.session_state.webstate = 'result'
            return

def handle_game(game, container, H, D):
    with container:
        st.session_state.webstate = 'choose'
        game.update_parameters(H, D)
        with st.spinner("Calculating if computer have winning strategy..."):
            strategy = game.has_stategy()

        st.session_state.strategy = strategy
        st.session_state.complete = None

def start_ai_game():
    st.session_state.mode = 'ai'
    st.session_state.webstate = 'play'

def start_random_game():
    st.session_state.mode = 'random'
    st.session_state.webstate = 'play'

if __name__ == '__main__':
    # Initialize session_state variables
    if 'complete' not in st.session_state:
        # Preserves the brick stability, number of drops, and is game was won or not
        st.session_state.complete = None
    if 'mode' not in st.session_state:
        # Preserves mode of the game 'ai' or 'random'
        st.session_state.mode = None
    if 'strategy' not in st.session_state:
        # Preserves does the computer have strategy or not
        st.session_state.strategy = None
    
    if 'webstate' not in st.session_state:
        # Preserves the state of the web page 'select', 'choose', 'play' or 'result'
        st.session_state.webstate = 'select'

    st.set_page_config(
        page_title="Dropping Bricks Puzzle",
    )

    st.title('Dropping Bricks Puzzle Solver')
    st.write('By Danil Kuchukov, IU student')

    col1, col2 = st.columns(2)
    with col1:
        H = st.number_input('Insert the height of the tower (H):', min_value=1, value=125, step=1)
    with col2:
        D = st.number_input('Insert the upper bound of allowed number of drops (D):', min_value=0, max_value=H, value=min(16, H), step=1)
    
    if 'game' not in st.session_state:
        st.session_state.game = GUIGame(H, D)

    with (playground := st.container()):
        game = st.session_state.game
        
        start = st.button("Start game", on_click=lambda: handle_game(game, playground, H, D), type='primary')

        if st.session_state.webstate == 'choose':
            if st.session_state.strategy:
                st.write("Computer **has** winning strategy.")
            else:
                st.write("Comuter **does not** have winning strategy.")
            st.write("Please, choose the prefered game mode:")
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                st.button("Strategy", on_click=start_ai_game)
            with fcol2:
                st.button("Random", on_click=start_random_game)

        if st.session_state.webstate == 'play':
            if st.session_state.mode == 'ai':
                m = game.step_ai()
            elif st.session_state.mode == 'random':
                m = game.step_random()

            st.metric(label="Computer drops", value=m, delta_color='off', delta=f'Total drops: {game.num_drops}, Bricks remaining: {game.state[2]}')
            st.write("Your response:")
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                st.button("Safe", on_click=game.safe_step)
            with fcol2:
                st.button("Broken", on_click=game.broken_step)
            
        if st.session_state.webstate == 'result':
            s, n, won = st.session_state.complete
            if won:
                st.write(f"""
                        The stability of the bricks is **{s}**. The game completed in **{n}** moves.\n
                        *You can start the game again.*
                        """)
            else:
                st.write(f"""
                        The computer failed to find the stability of the bricks in the allowed number
                         of drops. Maximum *safe* height is {s}.
                        """)
