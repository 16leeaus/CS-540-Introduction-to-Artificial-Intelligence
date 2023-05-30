import random
import copy

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        # Detect drop phase
        drop_phase = self.is_drop(state)

        if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            pass

        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better
        move = []
        (row, col) = (random.randint(0,4), random.randint(0,4))
        while not state[row][col] == ' ':
            (row, col) = (random.randint(0,4), random.randint(0,4))

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))
        return move

    # UNTESTED
    def is_drop(self, state):
        count = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                # Counts number of items, returns true if it exceeds 8
                if not(state[i][j] ==' '):
                    count += 1
                if count >= 8:
                    return True
        
        return False

    # UNTESTED
    def adjacent_spaces(self, color, state, current_space):
        i = current_space[0]
        j = current_space[1]
        iRange = [i-1, i+1]
        jRange = [j-1, j+1]
        
        # Checks to make sure nothing is out of bounds
        if i-1 < 0:
            iRange[0] = i
        if i+1 > len(state):
            iRange[1] = i
        if j-1 < 0:
            jRange[0] = j
        if j+1 > len(state[0]):
            jRange[1] = j

        results = []

        # Goes through all valid spots around the current space
        for x in range(iRange[0], iRange[1]):
            for y in range(jRange[0], jRange[1]):
                if state[x][y] == ' ':
                    tempResult = copy.deepcopy(state)
                    tempResult[x][y] = color
                    tempResult[i][j] = ' '
                    results.append(tempResult)
        return results

    # UNTESTED
    def succ(self, color, state):
        results = []
        drop_state = self.is_drop(state)

        # Does first section of the game
        if drop_state == True:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    if state[i][j] == ' ':
                        tempResult = copy.deepcopy(state)
                        tempResult[i][j] = color
                        results.append(tempResult)
        # Does the next section of the game
        if drop_state == False:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    if not(state[i][j] == ' '):
                        current_space = [i, j]
                        results.append(self.adjacent_spaces(color, state, current_space))
        return results


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    # UNTESTED
    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and 3x3 square corners wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # UNTESTED: check \ diagonal wins
        for row in range(1):
            for col in range(1):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[row][col]==self.my_piece else -1
        
        # UNTESTED: check / diagonal wins
        for row in range(1):
            for col in range(3, 4):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col-1] == state[row+2][col-2] == state[row+3][col-3]:
                    return 1 if state[row][col]==self.my_piece else -1
        
        # UNTESTED: check 3x3 square corners wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row][col+2] == state[row+2][col+2] == state[row+2][col] != state[row+1][col+1]:
                    return 1 if state[row][col]==self.my_piece else -1
    
        return 0 # no winner yet

        # UNTESTED
        def heuristic_game_value(self, state, d, turn):
            d -= 1
            gv = self.game_value(state)
            if turn == 1:
                turn = 0
            else:
                turn = 1
            sts = self.succ(self.my_piece, state)
            if gv != 0:
                return self.heuristic_game_value(state, 0, 0) if gv == 1 else self.heuristic_game_value(state, 0, 1)
            elif d == 0:
                return self.eX(state, self.my_piece)
            elif turn == 1:
                return max(self.heuristic_game_value(s, d, turn) for s in sts)
            elif turn == 0:
                return min(self.heuristic_game_value(s, d, turn) for s in sts)

        def eX(self, state, color):
            count = 0

            #Sum horizontal
            for row in range(state):
                for i in range(2):
                    for num in range(0,3):
                        if state[row][i+num] == color:
                            count += 1

            #Sum verticle
            for col in range(5):
                for i in range(2):
                    for num in range(0, 3):
                        if state[i + num][col] == color:
                            count += 1

            #Sum for diagonal \
            for row in range(1):
                for col in range(1):
                    for num in range(0, 3):
                        if state[row + num][col + num] == color:
                            count += 1

            #Sum for diagonal /
            for row in range(1):
                for col in range(1):
                    for num in range(0, 3):
                        if state[row+num][col-num]:
                            count += 1
            
            #Sum for 3x3 square
            for row in range(2):
                for col in range(2):
                    if state[row][col] == color:
                        count += 1
                    if state[row][col+2] == color:
                        count += 1
                    if state[row+2][col] == color:
                        count += 1
                    if state[row+2][col+2] == color:
                        count += 1
                    if state[row+1][col+1] == color:
                        count += 1

            return -1 + ((2 ** count) / 100)
            

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
