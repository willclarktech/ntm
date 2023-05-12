module Utils where

chunksOf :: Int -> [a] -> [[a]]
chunksOf width =
  foldr
    ( \value (h : t) ->
        if length h == width
          then [value] : (h : t)
          else (value : h) : t
    )
    [[]]
