import chess
import chess.pgn
import pandas as pd


def pgnProcessing(pgn_file, player, move_limit):
    """
    Process a PGN file to extract game data for a specific player and limit the number of moves.

    @param pgn_file: str - Path to the PGN file to be processed.
    @param player: list - Name(s) of the player(s) whose games are to be extracted.
    @param move_limit: int - Maximum number of moves to include in the DataFrame.

    @return: pd.DataFrame - A DataFrame containing processed game data for the specified player(s).
    """
    # Opening the pgn file
    f = open(pgn_file)

    # Storing all the games relevant datas
    games = []

    # Parse all games
    while True:
        game = chess.pgn.read_game(f)

        if game is None:
            break

        # Get relevant data from the pgn : player, date, color, result and moves
        # Targeted player's color
        player_color = 1.0 if game.headers["White"] in player else 0.0
        # Targeted player's name
        player_name = game.headers["White"] if player_color == 1.0 else game.headers["Black"]
        # Current game result
        result = game.headers["Result"]
        # Current game result from the targeted player's pov
        if (result == "1-0" and player == 1.0) or (result == "0-1" and player == 0.0):
            result = 1
        elif result == "1/2-1/2":
            result = 0.5
        else:
            result = 0
        # Game date (.07.01 if days/months not available)
        date = pd.to_datetime(game.headers["Date"].split(".")[0] + ".07.01") if "?" in game.headers[
            "Date"] else pd.to_datetime(game.headers["Date"])
        # First move_limit moves played during the current game
        moves = str(game.mainline_moves()).split(" ")
        moves = [moves[i] for i in range(len(moves)) if (i % 3 != 0)][:move_limit]

        # All datas related to the current game
        current_game = [date, player_name, player_color, result] + moves

        # Updating the full games list
        games.append(current_game)

    # Creating the final pandas dataframe
    columns = ["date", "player", "color", "result"] + [f'move_{i + 1}' for i in range(move_limit)]
    game_df = pd.DataFrame(games, columns=columns)

    return game_df


pgnProcessing("../../data/test_set.pgn", "Menkaoure", 6)
