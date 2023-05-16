import Control.Monad (forM_)
import Data.Foldable (Foldable (foldl'))
import Math (Matrix, crossEntropyLossSequence)
import Ntm (ControllerOutput (networkOutput), Ntm (controller), forwardPass, forwardPassSequence, initNtm, ntms2String, parseControllerOutput)
import System.Random (mkStdGen)
import TrainData (trainData, trainExample2String)

-- train :: Ntm -> [(Matrix, Matrix)] -> Int -> Float -> IO ()
-- train ntm trainData numEpochs learningRate = do
--   forM_ [1 .. numEpochs] $ \epoch -> do
--     putStrLn $ "Epoch " ++ show epoch
--     let ntm' = foldl' trainOnSample ntm trainData
--     -- putStrLn $ "Loss: " ++ (show $ totalLoss ntm' trainData)
--     print ntm'
--     return ()
--   where
--     trainOnSample :: Ntm -> (Matrix, Matrix) -> Ntm
--     trainOnSample ntm (inputSequence, targetSequence) =
--       let ntms = forwardPassSequence ntm inputSequence
--           outputSequence = map (networkOutput . parseControllerOutput 3 8 . controller) ntms
--           loss = crossEntropyLossSequence targetSequence outputSequence
--           outputGrad = outputSequence - targetSequence
--           ntm'' = foldr backwardPass ntm' outputGrad
--        in updateParametersNtm ntm'' learningRate

-- totalLoss :: Ntm -> [(Matrix, Matrix)] -> Float
-- totalLoss ntm trainData = -- Compute

main :: IO ()
main = do
  let seed = 12345
  let gen = mkStdGen seed
  -- One-hot encoded binary values plus no-op
  let inputWidth = 3
  -- Maximum sequence length
  let memorySize = 8
  let ntm = initNtm gen inputWidth memorySize
  putStr ">>> NTM: "
  print ntm
  let example = head trainData
  putStr ">>> Training example: "
  print $ trainExample2String example
  let result = forwardPassSequence ntm (fst example)
  let outputs = map (networkOutput . parseControllerOutput inputWidth memorySize . controller) result
  putStr ">>> Output: "
  print $ ntms2String result
  let loss = crossEntropyLossSequence (snd example) outputs
  putStr ">>> Loss: "
  print loss
