module Ntm where

import Data.Char (intToDigit)
import Data.List (transpose)
import Data.Tuple (swap)
import Debug.Trace (trace)
import Math (Matrix, Vector, argMax, complement, cosineSimilarity, crossEntropyLossSequence, dotProduct, multiply, outerProduct, softmax, zeroMatrix, zeroVector)
import Memory (ReadHead (ReadHead, addressingWeights, blending, keyVector, readOutput, sharpening, shiftVector), ReadHeadInput, WriteHead (WriteHead, addVector, eraseVector, writeReadHead), WriteHeadInput, parseReadHeadInput, parseWriteHeadInput, prepareReadHead, prepareWriteHead, writeOp)
import NeuralNetwork (Layer (RecurrentLayer, biases, input, output, weights), NeuralNetwork (NeuralNetwork, layers), propagateBackward, propagateForward)
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

-- Parsers

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

-- NTM

forwardPass :: Ntm -> Vector -> Ntm
forwardPass (Ntm controller memoryMatrix readHead writeHead) input =
  let l = length input
      n = length memoryMatrix
      controllerInput = readOutput readHead <> input
      newController = propagateForward controller controllerInput
      (ControllerOutput readHeadInput writeHeadInput networkOutput) = parseControllerOutput l n newController
      newReadHead = prepareReadHead readHead memoryMatrix readHeadInput
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

initBlending :: Double
initBlending = 0.5

initSharpening :: Double
initSharpening = 1.5

initVector :: StdGen -> Int -> Vector
initVector gen length = take length $ randomRs (-0.1, 0.1) gen

initMatrix :: StdGen -> Int -> Int -> Matrix
initMatrix gen height width = chunksOf width $ initVector gen (width * height)

initReadHead :: StdGen -> Int -> Int -> ReadHead
initReadHead gen l n =
  let (gen1, gen2) = split gen
   in ReadHead
        { addressingWeights = zeroVector n,
          keyVector = initVector gen1 l,
          shiftVector = initVector gen2 n,
          blending = initBlending,
          sharpening = initSharpening,
          readOutput = zeroVector l
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

initRecurrentLayer :: StdGen -> Int -> Int -> Layer
initRecurrentLayer gen inputWidth outputWidth =
  let (gen1, gen2) = split gen
      w = initMatrix gen1 outputWidth (inputWidth + outputWidth)
      wGrad = zeroMatrix outputWidth (inputWidth + outputWidth)
      b = initVector gen2 outputWidth
      i = zeroVector inputWidth
      o = zeroVector outputWidth
   in --  in RecurrentLayer weights biases i o (zeroMatrix outputWidth (inputWidth + outputWidth)) o o
      RecurrentLayer
        { input = i,
          weights = Parameter w wGrad,
          biases = Parameter b o,
          output = Parameter o o
        }

initController :: StdGen -> Int -> Int -> NeuralNetwork
initController gen l n =
  let -- key vector + shift vector
      readHeadInputWidth = l + n
      -- read head input + erase vector + add vector
      writeHeadInputWidth = readHeadInputWidth + (2 * l)
      -- read head output + sequence element
      inputWidth = 2 * l
      -- read head + write head + sequence element
      outputWidth = readHeadInputWidth + writeHeadInputWidth + l
      recurrentLayer = initRecurrentLayer gen inputWidth outputWidth
   in -- activationLayer = initActivationLayer l tanh
      NeuralNetwork
        { layers =
            [ recurrentLayer
            -- , activationLayer
            ]
        }

initNtm :: StdGen -> Int -> Int -> Ntm
initNtm gen inputWidth memorySize =
  let (gen1, gen') = split gen
      (gen2, gen'') = split gen'
      (gen3, gen4) = split gen''
   in Ntm
        { controller = initController gen1 inputWidth memorySize,
          memoryMatrix = initMatrix gen2 inputWidth memorySize,
          readHead = initReadHead gen3 inputWidth memorySize,
          writeHead = initWriteHead gen4 inputWidth memorySize
        }
