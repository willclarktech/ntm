module NeuralNetwork where

import Data.List (transpose)
import Math (Matrix, Vector, multiply, outerProduct, zeroVector)
import Parameter (Parameter (Parameter, gradient, value))

data Layer
  = InputLayer
      { output :: Parameter Vector
      }
  | RecurrentLayer
      { input :: Vector,
        weights :: Parameter Matrix,
        biases :: Parameter Vector,
        output :: Parameter Vector
      }

instance Show Layer where
  show (InputLayer output) = "InputLayer (" ++ show (length (value output)) ++ ")"
  show (RecurrentLayer {input, output}) = "RecurrentLayer (" ++ show (length input) ++ "->" ++ show (length (value output)) ++ ")"

newtype NeuralNetwork = NeuralNetwork
  { layers :: [Layer]
  }
  deriving (Show)

applyLayer :: Layer -> Layer -> Layer
applyLayer previousLayer layer =
  let (Parameter input _) = output previousLayer
      nextLayer = case layer of
        RecurrentLayer
          { weights,
            biases,
            output = previousOutput
          } ->
            let output = value biases + multiply (input <> value previousOutput) (value weights)
             in layer {input, output = Parameter output (zeroVector $ length output)}
        _ -> error "Not implemented"
   in nextLayer

propagateForward :: NeuralNetwork -> Vector -> NeuralNetwork
propagateForward (NeuralNetwork layers) input =
  let newLayers = foldl (\newLayers nextlayer -> newLayers ++ [applyLayer (last newLayers) nextlayer]) [InputLayer (Parameter input (zeroVector (length input)))] layers
   in NeuralNetwork $ tail newLayers

calculateLayerGrads :: Layer -> Vector -> (Layer, Vector)
calculateLayerGrads layer newOGrad = case layer of
  (RecurrentLayer {input, weights, biases, output}) ->
    let newWGrad = outerProduct newOGrad input
        newBGrad = newOGrad
        newLayer =
          layer
            { weights = weights {gradient = zipWith (+) (gradient weights) newWGrad},
              biases = biases {gradient = gradient biases + newBGrad},
              output = output {gradient = gradient output + newOGrad}
            }
        inputGrads = multiply newOGrad $ transpose $ value weights
     in (newLayer, inputGrads)
  _ -> error "Not implemented"

propagateBackward :: NeuralNetwork -> Vector -> NeuralNetwork
propagateBackward (NeuralNetwork layers) target =
  -- TODO: Generalize beyond cross entropy loss
  let outputGrad = value (output (last layers)) - target
      (updatedLayers, _) =
        foldr
          ( \layer (layersWithGrad, grad) ->
              let (layerWithGrad, newGrad) = calculateLayerGrads layer grad
               in (layerWithGrad : layersWithGrad, newGrad)
          )
          ([], outputGrad)
          layers
   in NeuralNetwork updatedLayers
