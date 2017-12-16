""" Player agent using minmax with pruning before reaching target tile value
"""

import sys
from random import randint
from BaseAI_3 import BaseAI
from Grid_3 import directionVectors, UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC
from Grid_3 import vecIndex, UP, DOWN, LEFT, RIGHT

class PlayerAI(BaseAI):
    def __init__(self):
        self.TARGET_TILE_VALUE = 2048

        self.TILE_VALUE_PROBA = {
            2: 9,
            # Here the rare case is omitted for speedup:
            # 4: 1
        }

        self.MAX_DEPTH = 4
        self.MAX_UTILITY = sys.maxsize

    def getMove(self, grid):
        # Before reaching target tile value--use minmax with pruning:
        move, _ = self.__maximize(grid, 0, -self.MAX_UTILITY, self.MAX_UTILITY)
        # After reaching target tile value--use random move to accelerate game termination:
        if move is None:
            moves = grid.getAvailableMoves()
            move = moves[randint(0, len(moves) - 1)]
        return move

    def __maximize(self, grid, depth, alpha, beta):
        if (
            depth == self.MAX_DEPTH or
            self.__is_terminal_state(grid)
        ):
            return (None, self.__eval(grid))

        max_move, max_utility = None, -self.MAX_UTILITY

        for move in grid.getAvailableMoves():
            # Move:
            grid_copy = grid.clone()
            grid_copy.move(move)
            # Minimize on new state:
            _, utility = self.__minimize(grid_copy, depth + 1, alpha, beta)
            # Update node statistics:
            if (utility > max_utility):
                max_move, max_utility = move, utility
            # Prune first--feasibility does matter:
            if (utility >= beta):
                break
            # Update search statistics:
            if (utility > alpha):
                alpha = utility

        return (max_move, max_utility)

    def __minimize(self, grid, depth, alpha, beta):
        if (
            depth == self.MAX_DEPTH or
            self.__is_terminal_state(grid)
        ):
            return (None, self.__eval(grid))

        min_move, min_utility = None, self.MAX_UTILITY

        for move in grid.getAvailableCells():
            # Put:
            grid_copy = grid.clone()

            # Expected utility:
            utility = 0
            freq = 0
            for value, proba in self.TILE_VALUE_PROBA.items():
                grid_copy.setCellValue(move, value)
                _, cond_utility = self.__maximize(grid_copy, depth + 1, alpha, beta)
                utility += proba * cond_utility
                freq += proba
            utility //= freq
            # Update node statistics:
            if (utility < min_utility):
                min_move, min_utility = move, utility
            # Prune first:
            if (utility <= alpha):
                break
            # Update search statistics:
            if (utility < beta):
                beta = utility

        return (min_move, min_utility)

    def __is_terminal_state(self, grid):
        return grid.getMaxTile() == self.TARGET_TILE_VALUE

    def __eval(self, grid):
        SMOOTH_WEIGHT = 10
        MONO_WEIGHT = 20
        EMPTY_WEIGHT = 80
        MAX_WEIGHT = 10

        # Initialize statistics:
        tile_stats = {
            'max': 0,
            'empty_count': 0
        }

        for pos in (
            (i, j) for i in range(grid.size) for j in range(grid.size)
        ):
            tile_value = grid.getCellValue(pos)

            if (tile_stats['max'] < tile_value):
                tile_stats['max'] = tile_value

            if (tile_value == 0):
                tile_stats['empty_count'] += 1

        utility = (
            SMOOTH_WEIGHT * self.__smoothness(grid) +
            MONO_WEIGHT * self.__monotonicity(grid) +
            EMPTY_WEIGHT * tile_stats['empty_count'] +
            MAX_WEIGHT * tile_stats['max']
        )

        return utility

    def __log2(self, value):
        return bin(value ^ (value - 1)).count("1") if value != 0 else 0

    def __find_farthest_pos(self, grid, start_pos, direction_vec):
        next_pos = (start_pos[0]+direction_vec[0], start_pos[1]+direction_vec[1])

        while (not grid.crossBound(next_pos) and grid.canInsert(next_pos)):
            next_pos = (next_pos[0]+direction_vec[0], next_pos[1]+direction_vec[1])

        return next_pos

    def __smoothness(self, grid):
        """ Evaluate smoothness of tile value distribution
        """
        smoothness = 0

        for start_pos in (
            (i, j) for i in range(grid.size) for j in range(grid.size)
        ):
            # Current cell is occupied:
            if (not grid.canInsert(start_pos)):
                start_value = self.__log2(grid.map[start_pos[0]][start_pos[1]])

                for direction_vec in (DOWN_VEC, RIGHT_VEC):
                    terminal_pos = self.__find_farthest_pos(grid, start_pos, direction_vec)
                    terminal_value = grid.getCellValue(terminal_pos)

                    if (not terminal_value is None):
                        terminal_value = self.__log2(terminal_value)
                        smoothness -= abs(start_value - terminal_value)

        return smoothness

    def __monotonicity(self, grid):
        """ Evaluate monotonicity of  tile value distribution
        """
        monotonicity = {
            UP: 0,
            DOWN: 0,
            LEFT: 0,
            RIGHT: 0
        }

        for col in range(grid.size):
            current_row, next_row = 0, 0

            while (next_row < grid.size):
                next_row, _ = self.__find_farthest_pos(grid, (current_row, col), DOWN_VEC)

                current_value = self.__log2(grid.map[current_row][col])
                next_value = self.__log2(grid.map[grid.size - 1 if next_row == grid.size else next_row][col])

                if (current_value < next_value):
                    monotonicity[DOWN] += current_value - next_value
                else:
                    monotonicity[UP] += next_value - current_value

                current_row = next_row

        for row in range(grid.size):
            current_col, next_col = 0, 0

            while (next_col < grid.size):
                _, next_col = self.__find_farthest_pos(grid, (row, current_col), RIGHT_VEC)

                current_value = self.__log2(grid.map[row][current_col])
                next_value = self.__log2(grid.map[row][grid.size - 1 if next_col == grid.size else next_col])

                if (current_value < next_value):
                    monotonicity[RIGHT] += current_value - next_value
                else:
                    monotonicity[LEFT] += next_value - current_value

                current_col = next_col

        return max(monotonicity[UP], monotonicity[DOWN]) + max(monotonicity[LEFT], monotonicity[RIGHT])

if __name__ == '__main__':
    from PlayerAI_3 import PlayerAI
    from Grid_3 import Grid

    player_ai = PlayerAI()
    grid = Grid()

    grid.map = [[0,0,2,4], [0,0,2,4], [0,2,2,2], [0,2,2,1024]]
    # Test case for PlayerAI.__maximize:
    utility = player_ai._PlayerAI__maximize(grid, 3, 0, 0)
    print("[Maximized Utility]: {}".format(utility))
    # Test case for PlayerAI.__eval:
    utility = player_ai._PlayerAI__eval(grid)
    print("[Utility]: {}".format(utility))
