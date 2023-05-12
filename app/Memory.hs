module Memory where

import Control.Monad (join)
import Data.Bifunctor (bimap)
import Data.List (transpose)
import Data.Tuple (swap)
import Math (Matrix, Vector, complement, cosineSimilarity, cosineSimilarity', dotProduct, initVector, multiply, softmax, softmax', zeroMatrix, zeroVector)
import Parameter (Parameter (Parameter, gradient, value))
import System.Random (RandomGen (split), StdGen)
import Utils (mapPair)

data ReadHeadInternal = ReadHeadInternal
  { contentInput :: (Matrix, Vector),
    interpolationInput :: (Double, Vector, Vector),
    shiftInput :: (Vector, Vector),
    focusInput :: (Double, Vector)
  }
  deriving (Show)

data ReadHead = ReadHead
  { addressingWeights :: Parameter Vector,
    keyVector :: Parameter Vector,
    shiftVector :: Parameter Vector,
    blending :: Parameter Double,
    sharpening :: Parameter Double,
    readOutput :: Parameter Vector,
    readHeadInternal :: ReadHeadInternal
  }
  deriving (Show)

data ReadHeadInput = ReadHeadInput
  { parsedKeyVector :: Vector,
    parsedShiftVector :: Vector
  }
  deriving (Show)

data WriteHead = WriteHead
  { writeReadHead :: ReadHead,
    eraseVector :: Vector,
    addVector :: Vector
  }
  deriving (Show)

data WriteHeadInput = WriteHeadInput
  { parsedHeadData :: ReadHeadInput,
    parsedEraseVector :: Vector,
    parsedAddVector :: Vector
  }
  deriving (Show)

-- Initialisation

initBlending :: Double
initBlending = 0.5

initSharpening :: Double
initSharpening = 1.5

initReadHead :: StdGen -> Int -> Int -> ReadHead
initReadHead gen l n =
  let (gen1, gen2) = split gen
      zN = zeroVector n
      zL = zeroVector l
   in ReadHead
        { addressingWeights = Parameter zN zN,
          keyVector = Parameter (initVector gen1 l) zL,
          shiftVector = Parameter (initVector gen2 n) zN,
          blending = Parameter initBlending 0,
          sharpening = Parameter initSharpening 0,
          readOutput = Parameter zL zL,
          readHeadInternal =
            ReadHeadInternal
              { contentInput = (zeroMatrix l n, zL),
                interpolationInput = (0, zN, zN),
                shiftInput = (zN, zN),
                focusInput = (0, zN)
              }
        }

initWriteHead :: StdGen -> Int -> Int -> WriteHead
initWriteHead gen l n =
  let (gen1, gen') = split gen
      (gen2, gen3) = split gen'
   in WriteHead
        { writeReadHead = initReadHead gen1 l n,
          eraseVector = initVector gen2 l,
          addVector = initVector gen3 l
        }

-- Parsing

parseReadHeadInput :: Int -> Vector -> ReadHeadInput
parseReadHeadInput l input =
  let (keyVector, shiftVector) = splitAt l input
   in ReadHeadInput
        { parsedKeyVector = keyVector,
          parsedShiftVector = shiftVector
        }

parseWriteHeadInput :: Int -> Int -> Vector -> WriteHeadInput
parseWriteHeadInput l n input =
  let (readHead, input') = splitAt (l + n) input
      (eraseVector, addVector) = splitAt l input'
      parsedHeadData = parseReadHeadInput l readHead
   in WriteHeadInput
        { parsedHeadData = parsedHeadData,
          parsedEraseVector = eraseVector,
          parsedAddVector = addVector
        }

-- Addressing

contentAddressing :: Matrix -> Vector -> Vector
contentAddressing memoryMatrix keyVector = softmax $ map (cosineSimilarity keyVector) memoryMatrix

contentAddressing' :: (Matrix, Vector) -> Vector
contentAddressing' (m, v) =
  let dOdCS = softmax' $ map (cosineSimilarity v) m
      dCSdV = map (sum . cosineSimilarity' v) m
   in zipWith (*) dCSdV dOdCS

interpolate :: Double -> Vector -> Vector -> Vector
interpolate g = zipWith (\c l -> (c * g) + (l * (1 - g)))

interpolate' :: (Double, Vector, Vector) -> (Double, Vector, Vector)
interpolate' (g, c, l) =
  let dIdG = sum $ zipWith (-) c l
      dIdC = replicate (length c) g
      dIdL = replicate (length l) (1 - g)
   in (dIdG, dIdC, dIdL)

cycleForward :: Vector -> Int -> Vector
cycleForward vector n = uncurry (++) $ swap $ splitAt n vector

shift :: Vector -> Vector -> Vector
shift addressingWeights shiftVector =
  let n = length shiftVector
      probabilities = softmax shiftVector
      shifted =
        map
          (cycleForward probabilities)
          [0 .. n - 1]
   in map (dotProduct addressingWeights) shifted

shift' :: (Vector, Vector) -> (Vector, Vector)
shift' (addressingWeights, shiftVector) =
  let n = length shiftVector
      probabilities = softmax shiftVector
      shifted =
        map
          (cycleForward probabilities)
          [0 .. n - 1]
      shiftedOutput = map (dotProduct addressingWeights) shifted

      dSdAW = map sum shifted

      dSdP = map (cycleForward addressingWeights) [0 .. n - 1]
      dPdSV = softmax' shiftVector
      dSdSV = multiply dPdSV dSdP
   in (dSdAW, dSdSV)

focus :: Double -> Vector -> Vector
focus gamma addressingWeights =
  let raised = map (** gamma) addressingWeights
      sigma = sum raised
   in map (/ sigma) raised

focus' :: (Double, Vector) -> (Double, Vector)
focus' (gamma, addressingWeights) =
  let raised = map (** gamma) addressingWeights
      sigma = sum raised
      focused = map (/ sigma) raised
      dFdA = map (\x -> (gamma * x ** (gamma - 1)) / sigma) addressingWeights
      dFdG = zipWith (*) raised $ map log addressingWeights
   in (sum dFdG, dFdA)

-- Data flow

readOp :: ReadHead -> Matrix -> Vector
readOp readHead = multiply (value $ addressingWeights readHead)

writeOp :: WriteHead -> Matrix -> Matrix
writeOp (WriteHead readHead eraseVector addVector) memoryMatrix =
  let addressVector = value $ addressingWeights readHead
      eraseComplements = map (\w -> complement $ map (* w) eraseVector) addressVector
      erased = zipWith (zipWith (*)) memoryMatrix eraseComplements
      weightedAddVectors = map (\w -> map (* w) addVector) addressVector
   in zipWith (+) erased weightedAddVectors

propagateForwardReadHead :: ReadHead -> Matrix -> ReadHeadInput -> ReadHead
propagateForwardReadHead readHead@(ReadHead {addressingWeights, blending, sharpening}) memoryMatrix (ReadHeadInput newKeyVector newShiftVector) =
  let contentWeights = contentAddressing memoryMatrix newKeyVector
      interpolated = interpolate (value blending) contentWeights (value addressingWeights)
      shifted = shift interpolated newShiftVector
      focused = focus (value sharpening) shifted
      newReadHead =
        readHead
          { addressingWeights = Parameter focused (zeroVector $ length focused),
            keyVector = Parameter newKeyVector (zeroVector $ length newKeyVector),
            shiftVector = Parameter newShiftVector (zeroVector $ length newShiftVector),
            readHeadInternal =
              ReadHeadInternal
                { contentInput = (memoryMatrix, newKeyVector),
                  interpolationInput = (value blending, contentWeights, value addressingWeights),
                  shiftInput = (interpolated, newShiftVector),
                  focusInput = (value sharpening, shifted)
                }
          }
      newReadOutput = readOp newReadHead memoryMatrix
   in newReadHead
        { readOutput = Parameter newReadOutput (zeroVector (length newReadOutput))
        }

propagateForwardWriteHead :: WriteHead -> Matrix -> WriteHeadInput -> (WriteHead, Matrix)
propagateForwardWriteHead (WriteHead readHead _ _) memoryMatrix (WriteHeadInput readHeadInput newEraseVector newAddVector) =
  let newReadHead = propagateForwardReadHead readHead memoryMatrix readHeadInput
      newWriteHead =
        WriteHead
          { writeReadHead = newReadHead,
            eraseVector = newEraseVector,
            addVector = newAddVector
          }
      newMemoryMatrix = writeOp newWriteHead memoryMatrix
   in (newWriteHead, newMemoryMatrix)

propagateBackwardReadHead :: ReadHead -> Matrix -> Vector -> ReadHead
propagateBackwardReadHead readHead@(ReadHead addressingWeights keyVector shiftVector blending sharpening readOutput internalInputs) memoryMatrix outputGrads =
  let fGrads = multiply outputGrads (transpose memoryMatrix)
      (gGrad, sGrads) = bimap (\n -> sum $ map (* n) fGrads) (* fGrads) $ focus' (focusInput internalInputs)
      (iGrads, svGrads) = mapPair (zipWith (*) sGrads) $ shift' (shiftInput internalInputs)
      (gammaGrad, cGrads, aGrads) = (\(x, y, z) -> (sum $ map (* x) iGrads, zipWith (*) iGrads y, zipWith (*) iGrads z)) $ interpolate' (interpolationInput internalInputs)
      kGrads = zipWith (*) cGrads $ contentAddressing' (contentInput internalInputs)
   in readHead
        { addressingWeights = addressingWeights {gradient = gradient addressingWeights + aGrads},
          keyVector = keyVector {gradient = gradient keyVector + kGrads},
          shiftVector = shiftVector {gradient = gradient shiftVector + svGrads},
          blending = blending {gradient = gradient blending + gammaGrad},
          sharpening = sharpening {gradient = gradient sharpening + gGrad},
          readOutput = readOutput {gradient = gradient readOutput + outputGrads}
        }
