module Utils where

import Control.Monad (join)
import Data.Bifunctor (bimap)

chunksOf :: Int -> [a] -> [[a]]
chunksOf width =
  foldr
    ( \value (h : t) ->
        if length h == width
          then [value] : (h : t)
          else (value : h) : t
    )
    [[]]

mapPair :: (a -> b) -> (a, a) -> (b, b)
mapPair = join bimap
