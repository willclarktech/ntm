module Memory where

import Data.List (transpose)
import Data.Tuple (swap)
import Math (Matrix, Vector, complement, cosineSimilarity, dotProduct, initVector, multiply, softmax, zeroVector)
import Parameter (Parameter (Parameter, value))
import System.Random (RandomGen (split), StdGen)

data ReadHead = ReadHead
  { addressingWeights :: Parameter Vector,
    keyVector :: Parameter Vector,
    shiftVector :: Parameter Vector,
    blending :: Parameter Double,
    sharpening :: Parameter Double,
    readOutput :: Parameter Vector
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
   in ReadHead
        { addressingWeights = Parameter (zeroVector n) (zeroVector n),
          keyVector = Parameter (initVector gen1 l) (zeroVector l),
          shiftVector = Parameter (initVector gen2 n) (zeroVector n),
          blending = Parameter initBlending 0,
          sharpening = Parameter initSharpening 0,
          readOutput = Parameter (zeroVector l) (zeroVector l)
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

interpolate :: Double -> Vector -> Vector -> Vector
interpolate g = zipWith (\c l -> (c * g) + (l * (1 - g)))

shift :: Vector -> Vector -> Vector
shift addressingWeights shiftVector =
  let l = length shiftVector
      probabilities = softmax shiftVector
      shifted =
        map
          (\i -> uncurry (++) $ swap $ splitAt i probabilities)
          [0 .. (l - 1)]
   in map (dotProduct addressingWeights) shifted

focus :: Double -> Vector -> Vector
focus gamma addressingWeights =
  let raised = map (** gamma) addressingWeights
      sigma = sum raised
   in map (/ sigma) raised

locationAddressing :: Double -> Double -> Matrix -> Vector -> Vector -> Vector -> Vector
locationAddressing g gamma memoryMatrix keyVector previousLocationWeights shiftVector =
  let contentWeights = contentAddressing memoryMatrix keyVector
      interpolated = interpolate g contentWeights previousLocationWeights
      shifted = shift interpolated shiftVector
   in focus gamma shifted

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
propagateForwardReadHead readHead@(ReadHead addressingWeights keyVector _ blending sharpening _) memoryMatrix (ReadHeadInput newKeyVector newShiftVector) =
  let newAddressingWeights = locationAddressing (value blending) (value sharpening) memoryMatrix newKeyVector (value addressingWeights) newShiftVector
      newReadHead =
        readHead
          { addressingWeights = Parameter newAddressingWeights (zeroVector $ length newAddressingWeights),
            keyVector = Parameter newKeyVector (zeroVector $ length newKeyVector),
            shiftVector = Parameter newShiftVector (zeroVector $ length newShiftVector)
          }
      newReadOutput = readOp newReadHead memoryMatrix
   in newReadHead
        { readOutput = Parameter newReadOutput (zeroVector (length newReadOutput))
        }

prepareWriteHead :: WriteHead -> Matrix -> WriteHeadInput -> WriteHead
prepareWriteHead (WriteHead readHead _ _) memoryMatrix (WriteHeadInput readHeadInput newEraseVector newAddVector) =
  let newReadHead = propagateForwardReadHead readHead memoryMatrix readHeadInput
   in WriteHead
        { writeReadHead = newReadHead,
          eraseVector = newEraseVector,
          addVector = newAddVector
        }

propagateBackwardReadHead :: ReadHead -> Vector -> ReadHead
propagateBackwardReadHead (ReadHead addressingWeights keyVector shiftVector blending sharpening readOutput) outputGrads =
  let blah = 123
   in ReadHead
        { addressingWeights,
          keyVector,
          shiftVector,
          blending,
          sharpening,
          readOutput
        }
