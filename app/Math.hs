module Math where

import Data.List (elemIndex, transpose)
import Foreign (fromBool)
import System.Random (Random (randomRs), StdGen)
import Utils (chunksOf)

-- Elementwise vector math
instance (Num a) => Num [a] where
  fromInteger n = [fromInteger n]
  abs = map abs
  (+) = zipWith (+)
  (-) = zipWith (-)
  (*) = error "* is not defined for []"
  signum = error "signum is not defined for []"

-- TODO: Consider richer types. See https://blog.jle.im/entry/fixed-length-vector-types-in-haskell.html
type Vector = [Double]

type Matrix = [Vector]

-- Initialisation

zeroVector :: Int -> Vector
zeroVector l = replicate l 0.0

zeroMatrix :: Int -> Int -> Matrix
zeroMatrix l n = replicate n $ zeroVector l

initVector :: StdGen -> Int -> Vector
initVector gen length = take length $ randomRs (-0.1, 0.1) gen

initMatrix :: StdGen -> Int -> Int -> Matrix
initMatrix gen width height = chunksOf width $ initVector gen (width * height)

-- Transformation

oneHotEncode :: Int -> Int -> Vector
oneHotEncode size v
  | v >= size = error $ "Invalid value for one-hot encoding (width " ++ show size ++ "): " ++ show v
oneHotEncode size v = map (fromBool . (== v)) [0 .. (size - 1)]

oneHotDecode :: Vector -> Int
oneHotDecode v = case elemIndex 1.0 v of
  Just idx -> idx
  Nothing -> error "Invalid one-hot encoded vector"

-- Vector math

argMax :: Vector -> Int
argMax v =
  let m = maximum v
   in case elemIndex m v of
        Nothing -> error "argmax failed"
        Just x -> x

mean :: Vector -> Double
mean ns =
  let n = length ns
   in sum ns / fromIntegral n

complement :: Vector -> Vector
complement = map (1 -)

l2Norm :: Vector -> Double
l2Norm v = sqrt $ sum $ map (^ 2) v

dotProduct :: Vector -> Vector -> Double
dotProduct a b = sum $ zipWith (*) a b

outerProduct :: Vector -> Vector -> Matrix
outerProduct u v = map (\u' -> map (u' *) v) u

multiply :: Vector -> Matrix -> Vector
multiply vector = map (dotProduct vector)

multiplyMatrices :: Matrix -> Matrix -> Matrix
multiplyMatrices m1 m2 = map (\row -> map (dotProduct row) (transpose m2)) m1

cosineSimilarity :: Vector -> Vector -> Double
cosineSimilarity a b = dotProduct a b / (l2Norm a * l2Norm b)

cosineSimilarity' :: Vector -> Vector -> Vector
cosineSimilarity' a b =
  let na = l2Norm a
      nb = l2Norm b
      dp = dotProduct a b
      term1 = map (/ (na * nb)) b
      term2 = map (* (dp / (na ** 3 * nb))) a
   in term1 - term2

softmax :: Vector -> Vector
softmax v =
  let exps = map exp v
      sigma = sum exps
   in map (/ sigma) exps

logSoftmax :: Vector -> Vector
logSoftmax v =
  let maxValue = maximum v
      expValues = map (exp . subtract maxValue) v
      expSum = sum expValues
   in map (\x -> log x - log expSum) expValues

jacobian :: Vector -> Matrix
jacobian x =
  let n = length x
   in [[x !! i * ((if i == j then 1 else 0) - x !! j) | j <- [0 .. n - 1]] | i <- [0 .. n - 1]]

softmax' :: Vector -> Vector
softmax' = map sum . jacobian . softmax

-- Loss functions

crossEntropyLoss :: Vector -> Vector -> Double
crossEntropyLoss target output =
  let logits = logSoftmax output
      nllLoss = dotProduct target logits
   in negate nllLoss / fromIntegral (length target)

crossEntropyLossSequence :: Matrix -> Matrix -> Double
crossEntropyLossSequence target output =
  let losses = zipWith crossEntropyLoss target output
   in mean losses
