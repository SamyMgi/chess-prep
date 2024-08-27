import chess
import chess.pgn
import numpy as np
import pandas as pd
from datetime import datetime

class FressinAI:
  """
    Input :
      * pgn_file : path of pgn games
      * player_name : name of the targeted player
      * depth : length of preparation
  """
  def __init__(self, pgn_file, player_name, depth):
    self.pgn_file = pgn_file
    self.player_name = player_name
    self.depth = depth
    self.gameProcessing()
    self.training()

  """
    Process games as a dataframe.
  """
  def gameProcessing(self):
    f = open(self.pgn_file)
    opp_games = []
    # Parse all games
    while True:
      game = chess.pgn.read_game(f)

      if game is None :
        break

      # Initialize an empty board
      board = chess.Board()

      # Get the result
      result = game.headers["Result"]

      # Identifying if the targeted player is white or black in the current game
      player = 1.0 if game.headers["White"] in self.player_name else 0.0

      # Determining the winner based on the result
      if (result == "1-0" and player == 1) or (result == "0-1" and player == 0) :
        result = 1
      elif result == "1/2-1/2":
        result = 0.5
      else :
        result = 0

      # Get the date
      if "?" in game.headers["Date"] :
        date = pd.to_datetime(game.headers["Date"].split(".")[0] + ".01.01")
      else :
        date = pd.to_datetime(game.headers["Date"])

      current_game = [player, result, None, date]

      # Count number of moves in the game
      move_count = 1

      # Parse move of the current game
      for move in game.mainline_moves():
        if move_count <= self.depth :
          move_san = board.san(move)
          current_game.append(move_san)
          board.push(move)
          move_count+=1

      opp_games.append(current_game)

    # Dataframe containing all the games
    self.games_df = pd.DataFrame(opp_games)

  """
    Training.
    Applying the algorithm on targeted player's games.
  """
  def training(self):
    # Treating both colors separately
    for i in range(2):
      target = i

      # Using the targeted color
      opp = self.games_df
      opp = opp.loc[opp[0] == target]

      if target == 1:
        depth = self.depth
      else :
        depth=self.depth-1

      best = []

      current_date = datetime.now()

      opp["time"] = (current_date.year - opp[opp.columns[3]].dt.year)*12 + current_date.month - opp[opp.columns[3]].dt.month

      for i in range(depth):
        # Choice for PLAYER
        if i%2==target:
          player_choice_df = opp.groupby(opp.columns[4]).agg(
          loss_perc=(1, 'mean'),
          game_played=(1, 'count'),
          results_sum=(1, 'sum'),
          ).reset_index()

          player_choice_df["log_game_played"] = np.log(player_choice_df["game_played"])
          threshold = player_choice_df["log_game_played"].max() - player_choice_df["log_game_played"].std()

          # From win percentage to loss percentage
          player_choice_df["loss_perc"] = 1 - player_choice_df["loss_perc"]

          # Std is NaN if only one move, so we use the threshold only if we have more than 1 value
          if player_choice_df.shape[0] > 1:
            player_choice_df = player_choice_df.loc[player_choice_df["log_game_played"] >= threshold].reset_index(drop=True)
            
          best.append(list(player_choice_df[player_choice_df.columns[0]])[player_choice_df["loss_perc"].idxmax()])
        else :
          # Intermediary df grouped by move and month.
          opp_choice_df = opp.groupby([opp.columns[4], "time"]).agg(
          game_played=(1, 'count')).reset_index()
          opp_choice_df["score_t"] = opp_choice_df["game_played"] * (1/(1+opp_choice_df["time"])**3)
          opp_choice_df = opp_choice_df.groupby(opp_choice_df.columns[0]).agg(
          game_played=("game_played", 'sum'),
          score_opp=("score_t", 'sum')).reset_index()
          best.append(list(opp_choice_df[opp_choice_df.columns[0]])[opp_choice_df["score_opp"].idxmax()])

        opp = opp.loc[opp[opp.columns[4]] == best[-1]]
        opp = opp.drop(opp.columns[4], axis=1)

      prepa = movelistToSan(best)

      if target == 1:
        self.black_prep = prepa
      else :
        self.white_prep = prepa

  """
    Evaluating the prep generated.
    Input :
      * evaluation_pgn : pgn games to be used for the evaluation.
      * target : color we want to prepare against.
  """
  def evaluation(self, evaluation_pgn, target):

    f = open(evaluation_pgn)

    if target == 1:
      prep = self.black_prep
    else :
      prep = self.white_prep

    prep = sanToMovelist(prep)

    move_check = [0] * len(prep)
    move_result = [0] * len(prep)

    total = 0 # Number of game where prep can work or fail
    usability = 0 # Number of game where the prep can be used
    success = 0 # Number of game won with the prep
    new = []
    # Parse the player game
    while True:

        game = chess.pgn.read_game(f)

        if game is None :
          break

        # Player's color in the current game
        opp_color = chess.WHITE if game.headers["White"] in self.player_name else chess.BLACK

        # Counter for move in prep
        c = 0

        board = chess.Board()
        not_applicable = False
        # Only looking games where the player plays with the color we want to prepare against
        if opp_color == target :
          for move in game.mainline_moves():
            # If P1 move
            if board.turn != opp_color:
              # If the prep is currently working
              if prep[c] == board.san(move) :
                board.push_san(prep[c])
              # Last player move is new
              elif c == len(prep)-1 : 
                board.push_san(prep[c])
                new.append(True)
              # Prep is not applicable. Ex: I want to play with white 1.e4 e5 2.Nf3, but 2.d4 was played during this game.
              else :
                not_applicable = True
                break
            # Else P2 move
            else :
              # Opponent is responding correctly according to prediction
              if prep[c] == board.san(move):
                board.push_san(prep[c])
              # Prep failed, opponent didn't respond correctly according to prediction
              else :
                total+=1
                break

            c+=1

            # If prep was fully applied
            if c == len(prep):
              total+=1
              usability+=1
              # Result of the prep
              result = game.headers.get("Result")
              if result == "1-0" and opp_color!=chess.WHITE:
                  success+=1
              elif result == "0-1" and opp_color!=chess.BLACK:
                  success+=1
              elif result == "1/2-1/2":
                  success+=0.5
              break

          # The game is applicable, we can take its result into account
          if not not_applicable :
            result = game.headers.get("Result")
            if result == "1-0" and opp_color!=chess.WHITE:
                s = 1
            elif result == "0-1" and opp_color!=chess.BLACK:
                s = 1
            elif result == "1/2-1/2":
                s = 0.5
            else :
              s = 0

            # Tracking indicators evolution accross the moves
            for i in range(c) :
              move_check[i]+=1
              move_result[i]+=s

    # Evaluation with indicators
    print("Prep :", prep)
    print("Games found:", usability, "/", total)
    if total !=0 :
      print("Usability rate :",round(usability/total,2)*100, "%")
      if len(new) == total :
        print("Success rate : 100% (NEW move)")
      elif usability != 0 :
        print("Success rate :", round(success/usability,2)*100, "%")

    print("Details :", move_check)
    print("Success details :", move_result)
    print()

    # Detailed evaluation
    if target == 1:
      self.black_usability = np.array(move_check)/total
      self.black_success = np.divide(np.array(move_result), np.array(move_check), out=np.zeros_like(np.array(move_result), dtype=float), where=np.array(move_check)!=0)
      print("Usability %:", self.black_usability)
      print("Success %:", self.black_success)
    else :
      self.white_usability = np.array(move_check)/total
      self.white_success = np.divide(np.array(move_result), np.array(move_check), out=np.zeros_like(np.array(move_result), dtype=float), where=np.array(move_check)!=0)
      print("Usability %:", self.white_usability)
      print("Success %:", self.white_success)


"""
  Convert san to a list of moves.
  Input :
    * san : moves in san format. Ex : "1.e4 c5 2.Nf3"]
  Output :
    * movelist : moves in list format. Ex : ["e4", "c5", "Nf3"]
"""
def sanToMovelist(san):
  movelist = san.split(" ")
  movelist = [move.split('.', 1)[1] if '.' in move else move for move in movelist]
  return movelist

"""
  Convert list to a san moves.
  Input :
    * movelist : moves in list format. Ex : ["e4", "c5", "Nf3"]
  Output :
    * san : moves in san format. Ex : "1.e4 c5 2.Nf3"]
"""
def movelistToSan(movelist):
  m = 1
  san = ""
  turn_color = 1
  for move in movelist:
    if turn_color == 1:
      san+=(str(int(m))+'.'+move)
      turn_color = 0
    else :
      san+=(' '+move+' ')
      turn_color = 1
    m+=0.5

  if san[-1] == " ":
    san = san[:-1]

  return san
