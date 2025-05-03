import chess
from enum import Enum

import chess.polyglot

class NodeType(Enum):
    EXACT = 0
    UPPER_BOUND = 1
    LOWER_BOUND = 2

class TTEntry():
    def __init__(self, hash_key, depth, score, node_type, best_move):
        self.hash_key = hash_key    # Position hash
        self.depth = depth          # Search depth
        self.score = score          # Position score
        self.node_type = node_type  # Type of node (EXACT, UPPERBOUND, LOWERBOUND)
        self.best_move = best_move  # Best move found

MATE_SCORE = 1000000


pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
    'KE' : (
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0,   0, -10, -20, -30,
        -30, -10, 20,  30,  30,  20,  -10, -30,
        -30, -10, 30,  40,  40,  30,  -10, -30,
        -30, -10, 30,  40,  40,  30,  -10, -30,
        -30, -10, 20,  30,  30,  20,  -10, -30,
        -30, -30, 0,   0, 0,   0,   -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50)

}

# Create mirrored tables for black pieces
pst_mirrored = {}
for piece, table in pst.items():
    pst_mirrored[piece.lower()] = tuple(reversed(table))

# print(pst_mirrored['p'][45])
# print( pst['P'][45])

piece_to_pst = {
    chess.PAWN: 'P',
    chess.KNIGHT: 'N',
    chess.BISHOP: 'B',
    chess.ROOK: 'R',
    chess.QUEEN: 'Q',
    chess.KING: 'K',
}


class Engine():
    def __init__(self):
        self.board = chess.Board()
        self.nodes_searched = 0
        self.transposition_table = {}
        self.tt_size = 1000000
        self.tt_hits = 0
        pass

    def set_fen(self, fen:str):
        self.board = chess.Board(fen)

    def set_moves(self, moves):
        for move in moves:
            self.board.push(move)

    def get_fen(self):
        return self.board.fen()
    
    def print_board(self):
        print(self.board)
    
    def piece_value(self, piece):
        """Assign simple values to pieces."""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King has infinite value, but not useful in evaluation
        }
        return values[piece.piece_type] if piece.color == chess.WHITE else -values[piece.piece_type]
    
    def is_endgame(self):
        """Check if we're in an endgame position"""
        white_queens = len(self.board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = len(self.board.pieces(chess.QUEEN, chess.BLACK))
        white_minors = len(self.board.pieces(chess.KNIGHT, chess.WHITE)) + \
                    len(self.board.pieces(chess.BISHOP, chess.WHITE))
        black_minors = len(self.board.pieces(chess.KNIGHT, chess.BLACK)) + \
                    len(self.board.pieces(chess.BISHOP, chess.BLACK))
        
        return white_queens + black_queens <= 1 and white_minors <= 2 and black_minors <= 2

    def can_apply_null_move(self):
        """Determine if it's safe to apply null move"""
        # Don't use null move if:
        # 1. In check
        # 2. In endgame
        # 3. Side to move has very few pieces (risk of zugzwang)
        return not (self.board.is_check() or self.is_endgame())

    def get_piece_position_score(self, piece, square, is_endgame):
        """Get the position score for a piece on a square."""
        # For white we are getting smaller numbers ie R on 0 so we flip it
        piece_symbol = piece_to_pst[piece.piece_type]

        # print(piece, piece.color, square)

        # If endgame we will serach the endgame table instead
        if is_endgame and piece == chess.KING:
            piece_symbol += "E"

        if piece.color == chess.WHITE:
            return pst[piece_symbol][63-square]
        else:
            # For black pieces, we can trust the table return negative score as white score is +ive
            return -pst[piece_symbol][square]

    def king_endgame_score(self, white_king_sq, black_king_sq, our_color, op_color):
        """In the endgame the engine should prefer pusshing the opponent king to the side of the board and bring its King closer so it can checkmate"""
        score = 0

        # There are 4 ccorners on the board
        corners = [0, 7, 63, 56]

        black_on_corner = black_king_sq in corners
        white_on_corner = white_king_sq in corners
        
        # *There are 4 sides taht correspond to 4 sides
        # Following numbers represent the piece positions on the sides of the board: the top and bottom rank and the left and right sides if the enemy is in this territory then we are in buisness
        files = [
                0, 1, 2, 3, 4, 5, 6, 7,
                8,                   15,
                16,                  23,
                24,                  31,
                32,                  39,
                40,                  47,
                48,                  55,
                56, 57, 58, 59, 60, 61, 62, 63
                ]

        black_on_side = black_king_sq in files
        white_on_side = white_king_sq in files

        if our_color == chess.WHITE:
            if black_on_corner:
                score += 30
            if black_on_side:
                score += 50

        if our_color == chess.BLACK:
            if white_on_corner:
                score += 30
            if white_on_side:
                score += 50
        
        # Todo:- Find the correct wayt o calculate manhatan distance
        # distance = abs(black_king_sq - white_king_sq)
        # score += (64-distance) *6

        return score

    def evaluate(self, ply=0):
        """Evaluate based on material difference, mate, and draw"""
        # Check for checkmate
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return -MATE_SCORE + ply # Black wins and played the last move
            else:
                return MATE_SCORE - ply  # White wins and played the last move
            
        
        # Check for stalemate
        if self.board.is_stalemate():
            return 0 

        # Check for draw by insufficient material or fifty-move rule
        if self.board.is_insufficient_material() or self.board.is_fifty_moves():
            return 0  
    
        score = 0

        white_king_sq = None
        black_king_sq = None

        # ! testing faster move generateion
        # Pieces cache
        white_pieces = self.board.occupied_co[chess.WHITE]
        black_pieces = self.board.occupied_co[chess.BLACK]

        is_endgame = self.is_endgame()
        
        # Evaluate White pieces
        for square in chess.scan_forward(white_pieces):
            piece = self.board.piece_at(square)
            score += self.piece_value(piece)
            score += self.get_piece_position_score(piece, square, is_endgame)

            if(piece.piece_type == chess.KING):
                white_king_sq = square
           
        
        # Evaluate black pieces
        for square in chess.scan_forward(black_pieces):
            piece = self.board.piece_at(square)
            score += self.piece_value(piece)
            score += self.get_piece_position_score(piece, square, is_endgame)

            # This will be used later for manhattan distance
            if(piece.piece_type == chess.KING):
                black_king_sq = square
        
        # * manhattan distance
        if is_endgame:
            # First calculate as our side is white and op is black and add
            # Calculate as our side is black ans op is white and add -ive of that
            end_game_score = self.king_endgame_score(white_king_sq, black_king_sq, chess.WHITE, chess.BLACK)
            - self.king_endgame_score(white_king_sq, black_king_sq, chess.BLACK, chess.WHITE)

            print(self.board, end_game_score)

            score += end_game_score

        # Todo add pos based eval like open files and pawns bishop pair etc
            
        return score
    
    def score_move(self, move, tt_move=None):
        """Scores for moves, The higher the better"""

        # TT move gets highest priority
        if tt_move and move.uci() == tt_move:
            return 30000

        # Captues, promotion and then checks ->
        if self.board.is_capture(move):
            victim = self.board.piece_at(move.to_square)
            attacker = self.board.piece_at(move.from_square)
            if victim and attacker:
                # _ is numerical literal has no affect on value
                return 10_000 +10000 + (abs(self.piece_value(victim)) * 10 - abs(self.piece_value(attacker)))
        
        if move.promotion:
            return 9_000 + (self.piece_value(chess.Piece(move.promotion, self.board.turn)))

        if self.board.gives_check(move):
            return 5_000
        
        
        # Else it is quiet move 
        return 0

    def ordered_moves(self):
        """Orders Moves based on priority"""

        # Search the table 
        pos_hash = chess.polyglot.zobrist_hash(self.board)

        tt_entry = self.transposition_table.get(pos_hash)
        tt_move = None

        if(tt_entry):
            tt_move = tt_entry.best_move

        moves = list(self.board.legal_moves)
        moves.sort(key=lambda move: -self.score_move(move, tt_move))
        return moves

    def store_position(self, hash_key, depth, score, node_type, best_move):
        """Store position in transposition table with simple size management"""
        if len(self.transposition_table) >= self.tt_size:
            # Remove the shalowest entry
            min_depth = float('inf')
            oldest_key = None
            for key, entry in self.transposition_table.items():
                if entry.depth < min_depth:
                    min_depth = entry.depth
                    oldest_key = key

            if oldest_key:
                self.transposition_table.pop(oldest_key)
            
        self.transposition_table[hash_key] = TTEntry(
            hash_key, depth, score, node_type, best_move
        )

    def quiescence(self, alpha, beta, depth=0, max_depth=10, ply=0):
        """Quiescence search to evaluate tactical sequences"""
        self.nodes_searched += 1

        stand_pat  = self.evaluate()

        if stand_pat >= beta:
            return beta
        
         # Update alpha if standing pat is better
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Stop if we've gone too deep
        # if depth >= max_depth:
        #     return alpha
        
        # Loop through legal moves captures only
        for move in self.ordered_moves():
            if not self.board.is_capture(move):
                continue

            # Get pieces involved
            victim = self.board.piece_at(move.to_square)
            attacker = self.board.piece_at(move.from_square)
    
            if not victim or not attacker:
                continue
        
            # Simple material comparison
            # Could be enhanced with more sophisticated SEE
            victim_value = abs(self.piece_value(victim))
            attacker_value = abs(self.piece_value(attacker))
    
            # Only search if we're not losing material
            # or capturing with pawn
            if victim_value < attacker_value and attacker.piece_type != chess.PAWN:
                continue

            # Delta Pruning
            if stand_pat + victim_value + 200 < alpha:
                continue
            
            self.board.push(move)
            score = -self.quiescence(-beta, -alpha, depth + 1, max_depth, ply+1)
            self.board.pop()
        
            if score >= beta:
                return beta
            
            if score > alpha:
                alpha = score
            
        return alpha

    # ! I am trying a lot of things but the engine's speed is not improving there is sure some major bottle neck
    def minmax(self, depth, alpha, beta, maximizing_player, ply=0, allow_null=True):
        self.nodes_searched += 1

        alpha_orig = alpha
        
        if(depth == 0 or self.board.is_game_over()):
            # return  self.quiescence(alpha, beta, ply=ply), []
            return  self.evaluate(ply), []

        # Search the table 
        pos_hash = chess.polyglot.zobrist_hash(self.board)
        tt_entry = self.transposition_table.get(pos_hash)

        # I have partially understood the upperbound and lower bound logic
        # ! the deeper we got the value of the depth var decreases
        # ! I think shalow evals will have a high depth calue in table as they
        if tt_entry and tt_entry.depth >= depth:
            self.tt_hits +=1
            if tt_entry.node_type == NodeType.EXACT:
                return tt_entry.score, [tt_entry.best_move]
            elif tt_entry.node_type == NodeType.LOWER_BOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.node_type == NodeType.UPPER_BOUND:
                beta = min(beta, tt_entry.score)
            
            if alpha >= beta:
                return tt_entry.score, [tt_entry.best_move]
        
        # Heard somethinng called fulfility pruning

        # Alpha is the best eval so far for the maximizing player
        if maximizing_player:
            max_eval = -float('inf')
            best_line = []
            best_move = None

            for move in self.ordered_moves():
                self.board.push(move)
                eval, line = self.minmax(depth -1, alpha, beta, False, ply + 1)
                self.board.pop()

                # From the best eval so far and the curr eval we will choose the best one
                if eval > max_eval:
                    max_eval = eval
                    best_line = [move.uci()] + line
                    best_move = move.uci()

                alpha = max(max_eval, alpha) # This will update the best eval so far if possible

                #* If at any point the score returned by the minimizing player is greater than or equal to beta, the maximizing player knows that the minimizing player will avoid this branch, so it can stop exploring further down this path (beta cutoff). Means there is a better move somewhere else for black
                if beta <= alpha:
                    break
            # Determine node type and store in transposition table
            node_type = NodeType.EXACT
            if max_eval <= alpha_orig:
                node_type = NodeType.UPPER_BOUND
            elif max_eval >= beta:
                node_type = NodeType.LOWER_BOUND

            # Store position in transposition table
            self.store_position(pos_hash, depth, max_eval, node_type, best_move)

            return max_eval, best_line
            
        # Opposite to that of maximizing player
        else:
            min_eval = float('inf')
            best_line = []
            best_move = None
            for move in self.ordered_moves():
                self.board.push(move)
                eval, line = self.minmax(depth -1, alpha, beta, True, ply + 1)
                self.board.pop()

                # Update the beta and eval values
                if eval < min_eval:
                    min_eval = eval
                    best_line = [move.uci()] + line
                    best_move = move.uci()

                beta = min(min_eval, beta) # Minimizing player wants the lowest value

                # If at any point the score returned by the maximizing player is less than or equal to alpha, the minimizing player knows that the maximizing player will avoid this branch, so it can stop exploring further down this path (alpha cutoff).
                if alpha >= beta:
                    break
            
            # Determine node type and store in transposition table
            node_type = NodeType.EXACT
            if min_eval <= alpha_orig:
                node_type = NodeType.UPPER_BOUND
            elif min_eval >= beta:
                node_type = NodeType.LOWER_BOUND

            # Store position in transposition table
            self.store_position(pos_hash, depth, min_eval, node_type, best_move)
            
            return min_eval, best_line

    def search(self, depth):
        """Do simple search for now"""

        # eval = self.evaluate()
        # print(eval)


        # If white to move best score is the worst that can happen so any other move will be evaluated greater than that
        best_score = -float('inf') if self.board.turn == chess.WHITE else float('inf')
        best_move = None
        best_line = []
        self.nodes_searched = 0

        # Loop and evaluate each move
        for move in self.ordered_moves():
            self.board.push(move)
            score, line = self.minmax(depth -1, -float('inf'), float('inf'), self.board.turn, ply=1)
            self.board.pop()

            if(self.board.turn == chess.WHITE):
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_line = [move.uci()] + line
            
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    best_line = [move.uci()] + line


        # print("Nodes Searched: ", self.nodes_searched)


        if abs(best_score) > 900_000:
            mate_in = (MATE_SCORE - abs(best_score)) // 2
            mate_in = -mate_in if best_score < 0 else mate_in
            print(f"info depth {depth} score mate {mate_in} pv {' '.join(best_line)}")
        else:
            print(f"info depth {depth} nodes {self.nodes_searched} score cp {best_score} pv {' '.join(best_line)}")

        print("tt hits: ", self.tt_hits)

        return best_move.uci() if best_move else None