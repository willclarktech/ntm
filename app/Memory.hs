module Memory where

import Data.List (transpose)
import Data.Tuple (swap)
import Math (Matrix, Vector, complement, cosineSimilarity, dotProduct, multiply, softmax)

data ReadHead = ReadHead
  { addressingWeights :: Vector,
    keyVector :: Vector,
    shiftVector :: Vector,
    blending :: Double,
    sharpening :: Double,
    readOutput :: Vector
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

contentAddressing :: Matrix -> Vector -> Vector
contentAddressing memoryMatrix keyVector = softmax $ map (cosineSimilarity keyVector) memoryMatrix

interpolate :: Double -> Vector -> Vector -> Vector
interpolate g = zipWith (\c l -> (c * g) + (l * (1 - g)))

shift :: Vector -> Vector -> Vector
shift addressingWeights shiftVector =
  let l = length shiftVector
      shifted =
        map
          (\i -> uncurry (<>) $ swap $ splitAt i shiftVector)
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

prepareReadHead :: ReadHead -> Matrix -> ReadHeadInput -> ReadHead
prepareReadHead (ReadHead addressingWeights keyVector _ blending sharpening readOutput) memoryMatrix (ReadHeadInput newKeyVector newShiftVector) =
  let newAddressingWeights = locationAddressing blending sharpening memoryMatrix newKeyVector addressingWeights newShiftVector
   in ReadHead
        { addressingWeights = newAddressingWeights,
          keyVector = newKeyVector,
          shiftVector = newShiftVector,
          blending = blending,
          sharpening = sharpening,
          readOutput = readOutput
        }

prepareWriteHead :: WriteHead -> Matrix -> WriteHeadInput -> WriteHead
prepareWriteHead (WriteHead readHead _ _) memoryMatrix (WriteHeadInput readHeadInput newEraseVector newAddVector) =
  let newReadHead = prepareReadHead readHead memoryMatrix readHeadInput
   in WriteHead
        { writeReadHead = newReadHead,
          eraseVector = newEraseVector,
          addVector = newAddVector
        }

readOp :: ReadHead -> Matrix -> Vector
readOp readHead memoryMatrix = multiply (addressingWeights readHead) (transpose memoryMatrix)

writeOp :: WriteHead -> Matrix -> Matrix
writeOp (WriteHead readHead eraseVector addVector) memoryMatrix =
  let addressVector = addressingWeights readHead
      eraseComplements = map (\w -> complement $ map (* w) eraseVector) addressVector
      erased = zipWith (zipWith (*)) memoryMatrix eraseComplements
      weightedAddVectors = map (\w -> map (* w) addVector) addressVector
   in zipWith (+) erased weightedAddVectors

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
