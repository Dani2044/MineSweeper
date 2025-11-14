## =========================================================================
## @author Gabriel Riaño (rianogabriel@javeriana.edu.co)
## @author Daniel Castro (castro-df@javeriana.edu.co)
## =========================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

"""
"""
class Player:

  model = None

  '''
  '''
  def __init__( self, args ):
    self.train()
  # end def

  def train( self ):
    #Read data
    features = pd.read_csv('X.csv', header=None)
    labels = pd.read_csv('y.csv', header=None)

    #Plain data
    labels = labels.values.ravel()

    #Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    #Model
    model = LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced")

    #Train
    model.fit(X_train, y_train)

    #Test
    y_pred = model.predict(X_test)

    #Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    self.model = model
  # end def

  '''
  '''
  def choose_cell( self, w, h, n, board ):
    
    best_i = None
    best_j = None
    min_p_mine = float('inf')

    for i in range(w):
      for j in range(h):
        if not board.m_Patches[i][j]:
          x = self.collect_neighbors(board, i, j)
          p = self.model.predict_proba([x])[0][1]
          if p < min_p_mine:
            min_p_mine = p
            best_i = i
            best_j = j
    
    if min_p_mine == float('inf'):
      print("Random")
      best_i = random.randint(0, w - 1)
      best_j = random.randint(0, h - 1)

    return best_i, best_j
  # end def

  def collect_neighbors( self, board, i, j ):
    neighbors = []
    rows = len(board.m_Mines)
    cols = len(board.m_Mines[0])
    
    # 8 possible directions (up, down, left, right, and diagonals)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]
    
    for dr, dc in directions:
      r, c = i + dr, j + dc
      # Check boundaries
      if 0 <= r < rows and 0 <= c < cols:
        #Check if its revealed
        if board.m_Patches[r][c]:
          neighbors.append(board.m_Mines[r][c])
        else:
          neighbors.append(9)
        # end if
      else:
        neighbors.append(0)
      # end if
    # end for
    
    return neighbors
# end class
## eof - SmartPlayer.py