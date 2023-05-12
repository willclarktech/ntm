import Math (crossEntropyLossSequence)
import Ntm (ControllerOutput (networkOutput), Ntm (controller), forwardPassSequence, initNtm, ntms2String, parseControllerOutput)
import System.Random (mkStdGen)
import TrainData (trainData, trainExample2String)

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
