module Memory where

import Control.Monad (join)
import Data.Bifunctor (bimap)
import Data.List (transpose)
import Data.Tuple (swap)
import Math (Matrix, Vector, complement, cosineSimilarity, cosineSimilarity', dotProduct, initVector, multiply, multiplyMatrices, softmax, softmax', zeroMatrix, zeroVector)
import Parameter (Parameter (Parameter, gradient, value))
import System.Random (RandomGen (split), StdGen)
import Utils (mapPair)

data ReadHeadInternal = ReadHeadInternal
  { contentInput :: (Matrix, Vector),
    interpolationInput :: (Double, Vector, Vector),
    shiftInput :: (Vector, Vector),
    focusInput :: (Double, Vector),
    readOpInput :: Matrix
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

newtype WriteHeadInternal = WriteHeadInternal
  { memoryMatrixInput :: Matrix
  }
  deriving (Show)

data WriteHead = WriteHead
  { readHeadW :: ReadHead,
    eraseVector :: Parameter Vector,
    addVector :: Parameter Vector,
    writeHeadInternal :: WriteHeadInternal
  }
  deriving (Show)

data WriteHeadInput = WriteHeadInput
  { parsedReadHeadInput :: ReadHeadInput,
    parsedEraseVector :: Vector,
    parsedAddVector :: Vector
  }
  deriving (Show)

-- Initialisation

initBlending :: Double
initBlending = 0.5

initSharpening :: Double
initSharpening = 1.5

initReadHeadInternal :: Int -> Int -> ReadHeadInternal
initReadHeadInternal l n =
  let zN = zeroVector n
      zL = zeroVector l
      zM = zeroMatrix l n
   in ReadHeadInternal
        { contentInput = (zeroMatrix l n, zL),
          interpolationInput = (0, zN, zN),
          shiftInput = (zN, zN),
          focusInput = (0, zN),
          readOpInput = zM
        }

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
          readHeadInternal = initReadHeadInternal l n
        }

initWriteHead :: StdGen -> Int -> Int -> WriteHead
initWriteHead gen l n =
  let (gen1, gen') = split gen
      (gen2, gen3) = split gen'
      zL = zeroVector l
      zN = zeroVector n
      zM = zeroMatrix l n
   in WriteHead
        { readHeadW = initReadHead gen1 l n,
          eraseVector = Parameter (initVector gen2 l) zL,
          addVector = Parameter (initVector gen3 l) zL,
          writeHeadInternal =
            WriteHeadInternal
              { memoryMatrixInput = zM
              }
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
      parsedReadHeadInput = parseReadHeadInput l readHead
   in WriteHeadInput
        { parsedReadHeadInput = parsedReadHeadInput,
          parsedEraseVector = eraseVector,
          parsedAddVector = addVector
        }

-- Reading

contentAddressing :: Matrix -> Vector -> Vector
contentAddressing memoryMatrix keyVector = softmax $ map (cosineSimilarity keyVector) memoryMatrix

contentAddressing' :: (Matrix, Vector) -> Vector
contentAddressing' (m, v) =
  let dOdCS = softmax' $ map (cosineSimilarity v) m
      dCSdV = map (sum . cosineSimilarity' v) m
   in zipWith (*) dCSdV dOdCS

interpolate :: Double -> Vector -> Vector -> Vector
interpolate g = zipWith (\c l -> (c * g) + (l * (1 - g)))

interpolate' :: (Double, Vector, Vector) -> (Double, Vector)
interpolate' (g, c, l) =
  let dIdG = sum $ zipWith (-) c l
      dIdC = replicate (length c) g
   in (dIdG, dIdC)

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

readOp :: ReadHead -> Matrix -> Vector
readOp readHead = multiply (value $ addressingWeights readHead)

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
                  focusInput = (value sharpening, shifted),
                  readOpInput = memoryMatrix
                }
          }
      newReadOutput = readOp newReadHead memoryMatrix
   in newReadHead
        { readOutput = Parameter newReadOutput (zeroVector (length newReadOutput))
        }

propagateBackwardReadHead :: ReadHead -> Vector -> ReadHead
propagateBackwardReadHead readHead@(ReadHead {addressingWeights, keyVector, shiftVector, blending, sharpening, readOutput, readHeadInternal}) outputGrads =
  let fGrads = multiply outputGrads (transpose $ readOpInput readHeadInternal)
      (gGrad, sGrads) = bimap (\n -> sum $ map (* n) fGrads) (* fGrads) $ focus' (focusInput readHeadInternal)
      (iGrads, svGrads) = mapPair (zipWith (*) sGrads) $ shift' (shiftInput readHeadInternal)
      (gammaGrad, cGrads) = (\(x, y) -> (sum $ map (* x) iGrads, zipWith (*) iGrads y)) $ interpolate' (interpolationInput readHeadInternal)
      kGrads = zipWith (*) cGrads $ contentAddressing' (contentInput readHeadInternal)
   in readHead
        { addressingWeights = addressingWeights {gradient = gradient addressingWeights + fGrads},
          keyVector = keyVector {gradient = gradient keyVector + kGrads},
          shiftVector = shiftVector {gradient = gradient shiftVector + svGrads},
          blending = blending {gradient = gradient blending + gammaGrad},
          sharpening = sharpening {gradient = gradient sharpening + gGrad},
          readOutput = readOutput {gradient = gradient readOutput + outputGrads}
        }

-- Writing

erase :: Matrix -> Vector -> Vector -> Matrix
erase memoryMatrix addressVector eraseVector =
  let eraseComplements = map (\w -> complement $ map (* w) eraseVector) addressVector
   in zipWith (zipWith (*)) memoryMatrix eraseComplements

add :: Matrix -> Vector -> Vector -> Matrix
add memoryMatrix addressVector addVector =
  let weightedAddVectors = map (\w -> map (* w) addVector) addressVector
   in zipWith (+) memoryMatrix weightedAddVectors

writeOp :: WriteHead -> Matrix -> (Matrix, WriteHeadInternal)
writeOp (WriteHead {readHeadW, eraseVector, addVector}) memoryMatrix =
  let addressVector = value $ addressingWeights readHeadW
      erased = erase memoryMatrix addressVector (value eraseVector)
      newMemoryMatrix = add erased addressVector (value addVector)
      writeHeadInternal =
        WriteHeadInternal
          { memoryMatrixInput = memoryMatrix
          }
   in (newMemoryMatrix, writeHeadInternal)

propagateForwardWriteHead :: WriteHead -> Matrix -> WriteHeadInput -> (WriteHead, Matrix)
propagateForwardWriteHead writeHead@(WriteHead {readHeadW}) memoryMatrix (WriteHeadInput readHeadInput newEraseVector newAddVector) =
  let newReadHead = propagateForwardReadHead readHeadW memoryMatrix readHeadInput
      newWriteHead =
        writeHead
          { readHeadW = newReadHead,
            eraseVector = Parameter newEraseVector (zeroVector $ length newEraseVector),
            addVector = Parameter newAddVector (zeroVector $ length newAddVector)
          }
      (newMemoryMatrix, newWriteHeadInternal) = writeOp newWriteHead memoryMatrix
   in (newWriteHead {writeHeadInternal = newWriteHeadInternal}, newMemoryMatrix)

propagateBackwardWriteHead :: WriteHead -> Matrix -> WriteHead
propagateBackwardWriteHead writeHead@(WriteHead {readHeadW, eraseVector, addVector, writeHeadInternal = (WriteHeadInternal {memoryMatrixInput})}) outputGrads =
  let addressVector = value $ addressingWeights readHeadW
      addVectorGrads = multiply addressVector $ transpose outputGrads
      aGrad1 = multiply (value addVector) outputGrads
      eraseGrads = multiplyMatrices (transpose memoryMatrixInput) outputGrads
      eraseVectorGrads = multiply (complement addressVector) $ transpose eraseGrads
      aGrad2 = multiply (complement $ value eraseVector) eraseGrads
      newReadHead = propagateBackwardReadHead readHeadW (aGrad1 + aGrad2)
   in writeHead
        { readHeadW = newReadHead,
          eraseVector = eraseVector {gradient = gradient eraseVector + eraseVectorGrads},
          addVector = addVector {gradient = gradient addVector + addVectorGrads}
        }
