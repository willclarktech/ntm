module Ntm where

import Data.Char (intToDigit)
import Data.List (transpose)
import Data.Tuple (swap)
import Math (Matrix, Vector, argMax, complement, cosineSimilarity, crossEntropyLossSequence, dotProduct, initMatrix, initVector, multiply, outerProduct, softmax, zeroMatrix, zeroVector)
import Memory (ReadHead (ReadHead, addressingWeights, blending, keyVector, readOutput, sharpening, shiftVector), ReadHeadInput, WriteHead (WriteHead, addVector, eraseVector, readHeadW), WriteHeadInput, initReadHead, initWriteHead, parseReadHeadInput, parseWriteHeadInput, propagateBackwardReadHead, propagateBackwardWriteHead, propagateForwardReadHead, propagateForwardWriteHead, writeOp)
import NeuralNetwork (Layer (RecurrentLayer, biases, input, output, weights), NeuralNetwork (NeuralNetwork, layers), initNeuralNetwork, propagateBackwardNeuralNetwork, propagateForward)
import Parameter (Parameter (Parameter, gradient, value))
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

ntm2Char :: Ntm -> Char
ntm2Char ntm =
  let mm = memoryMatrix ntm
      n = length mm
      l = length $ head mm
   in intToDigit . argMax . networkOutput . parseControllerOutput l n . controller $ ntm

ntms2String :: [Ntm] -> String
ntms2String = map ntm2Char

-- Parsing

parseControllerInput :: Int -> NeuralNetwork -> ControllerInput
parseControllerInput l (NeuralNetwork layers) =
  let v = value $ input $ head layers
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

-- Data flow

forwardPass :: Ntm -> Vector -> Ntm
forwardPass (Ntm {controller, memoryMatrix, readHead, writeHead}) input =
  let l = length input
      n = length memoryMatrix
      controllerInput = value (readOutput readHead) ++ input
      newController = propagateForward controller controllerInput
      (ControllerOutput readHeadInput writeHeadInput networkOutput) = parseControllerOutput l n newController
      newReadHead = propagateForwardReadHead readHead memoryMatrix readHeadInput
      (newWriteHead, newMemoryMatrix) = propagateForwardWriteHead writeHead memoryMatrix writeHeadInput
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

-- NOTE: Not proper BPTT but a simpler approximation
backwardPass :: Ntm -> Vector -> Ntm
backwardPass ntm@(Ntm {controller, readHead, writeHead, memoryMatrix}) outputGrad =
  let n = length memoryMatrix
      l = length $ head memoryMatrix
      readHeadInputWidth = l + n
      writeHeadInputWidth = readHeadInputWidth + 2 * l
      controllerOutputGrad1 = zeroVector readHeadInputWidth ++ zeroVector writeHeadInputWidth ++ outputGrad
      newController1 = propagateBackwardNeuralNetwork controller controllerOutputGrad1
      readHeadOutputGrad = take l $ gradient $ input $ head $ layers newController1
      newReadHead = propagateBackwardReadHead readHead readHeadOutputGrad
      memoryMatrixGrad = outerProduct (value $ addressingWeights readHead) readHeadOutputGrad
      newWriteHead = propagateBackwardWriteHead writeHead memoryMatrixGrad
      controllerOutputGrad2 =
        foldl
          (\g -> (++ g) . gradient)
          []
          [ keyVector newReadHead,
            shiftVector newReadHead,
            keyVector $ readHeadW newWriteHead,
            shiftVector $ readHeadW newWriteHead,
            eraseVector newWriteHead,
            addVector newWriteHead
          ]
          ++ zeroVector (length outputGrad)

      newController2 = propagateBackwardNeuralNetwork newController1 controllerOutputGrad2
   in ntm
        { controller = newController2,
          readHead = newReadHead,
          writeHead = newWriteHead
        }

-- backwardPassSequence ::

-- updateParametersNtm :: Float -> Ntm -> Ntm
-- updateParametersNtm learningRate ntm@(Ntm {controller, readHead, writeHead}) =
--   ntm
--     { controller = updateParametersController controller,
--       readHead = updateParametersReadHead readHead,
--       writeHead = updateParametersWriteHead writeHead
--     }
