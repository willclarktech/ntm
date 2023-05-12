module Utils where

import Math (Matrix, Vector)

chunksOf :: Int -> Vector -> Matrix
chunksOf width =
  foldr
    ( \value (h : t) ->
        if length h == width
          then [value] : (h : t)
          else (value : h) : t
    )
    [[]]
