"""TD(0) afterstate learning with n-tuple networks for 2048."""
from .board import move, new_board, spawn_tile, seed, render, to_grid, from_grid, ACTION_NAMES
from .ntuple import NTupleNetwork, NETWORKS
from .agent import play_game, run_games, run_random_games, greedy_action, summarize
