module Ntm where

import Data.Char (intToDigit)
import Data.List (transpose)
import Data.Tuple (swap)
import Debug.Trace (trace)
import Math (Matrix, Vector, argMax, complement, cosineSimilarity, crossEntropyLossSequence, dotProduct, initMatrix, initVector, multiply, outerProduct, softmax, zeroMatrix, zeroVector)
import Memory (ReadHead (ReadHead, addressingWeights, blending, keyVector, readOutput, sharpening, shiftVector), ReadHeadInput, WriteHead (WriteHead, addVector, eraseVector, writeReadHead), WriteHeadInput, initReadHead, initWriteHead, parseReadHeadInput, parseWriteHeadInput, prepareWriteHead, propagateForwardReadHead, writeOp)
import NeuralNetwork (Layer (RecurrentLayer, biases, input, output, weights), NeuralNetwork (NeuralNetwork, layers), initNeuralNetwork, propagateBackward, propagateForward)
import Parameter (Parameter (Parameter, value))
import System.Random (StdGen, randomRs, split)
import Utils (chunksOf)

data ControllerInput = ControllerInput
  { readHeadOutput :: Vector,
    dataInput :: Vector
  }

data ControllerOutput = ControllerOutput
  { readHeadInput :: ReadHeadInput,
    writeHeadInput :: WriteHeadInput,
    networkOutput :: Vector
  }
  deriving (Show)

data Ntm = Ntm
  { controller :: NeuralNetwork,
    memoryMatrix :: Matrix,
    readHead :: ReadHead,
    writeHead :: WriteHead
  }
  deriving (Show)

-- Utils

ntms2String :: [Ntm] -> String
ntms2String ntms =
  -- TODO: Generalize
  let inputWidth = 3
      memorySize = 8
      outputs = map (networkOutput . parseControllerOutput inputWidth memorySize . controller) ntms
   in map (intToDigit . argMax) outputs

-- Parsing

parseControllerInput :: Int -> NeuralNetwork -> ControllerInput
parseControllerInput l (NeuralNetwork layers) =
  let v = input $ head layers
      (readHeadOutput, dataInput) = splitAt l v
   in ControllerInput readHeadOutput dataInput

parseControllerOutput :: Int -> Int -> NeuralNetwork -> ControllerOutput
parseControllerOutput l n (NeuralNetwork layers) =
  let readHeadInputWidth = l + n
      writeHeadInputWidth = readHeadInputWidth + 2 * l
      o = output $ last layers
      (readHeadInput, o') = splitAt readHeadInputWidth $ value o
      (writeHeadInput, networkOutput) = splitAt writeHeadInputWidth o'
      parsedReadHeadInput = parseReadHeadInput l readHeadInput
      parsedWriteHeadInput = parseWriteHeadInput l n writeHeadInput
   in ControllerOutput parsedReadHeadInput parsedWriteHeadInput networkOutput

-- Data flow

forwardPass :: Ntm -> Vector -> Ntm
forwardPass (Ntm controller memoryMatrix readHead writeHead) input =
  let l = length input
      n = length memoryMatrix
      controllerInput = value (readOutput readHead) ++ input
      newController = propagateForward controller controllerInput
      (ControllerOutput readHeadInput writeHeadInput networkOutput) = parseControllerOutput l n newController
      newReadHead = propagateForwardReadHead readHead memoryMatrix readHeadInput
      newWriteHead = prepareWriteHead writeHead memoryMatrix writeHeadInput
      newMemoryMatrix = writeOp newWriteHead memoryMatrix
   in Ntm
        { controller = newController,
          memoryMatrix = newMemoryMatrix,
          readHead = newReadHead,
          writeHead = newWriteHead
        }

forwardPassSequence :: Ntm -> Matrix -> [Ntm]
forwardPassSequence ntm inputSequence =
  tail $
    foldl
      ( \ntms -> (ntms ++) . (: []) . forwardPass (last ntms)
      )
      [ntm]
      inputSequence

-- backwardPass :: Ntm -> Vector -> Ntm
-- backwardPass (Ntm) target =

-- Initialisation

initNtm :: StdGen -> Int -> Int -> Ntm
initNtm gen inputWidth memorySize =
  let (gen1, gen') = split gen
      (gen2, gen'') = split gen'
      (gen3, gen4) = split gen''
   in Ntm
        { controller = initNeuralNetwork gen1 inputWidth memorySize,
          memoryMatrix = initMatrix gen2 inputWidth memorySize,
          readHead = initReadHead gen3 inputWidth memorySize,
          writeHead = initWriteHead gen4 inputWidth memorySize
        }
