import System.Random
import Data.List
import Control.Monad.State
import Control.Monad.Writer
import Control.Arrow
import Control.Applicative ((<$>))
import Debug.Trace
import Text.Printf

data NeuralNetwork = NeuralNetwork Nodes Nodes Nodes

data Node = Node { activation :: Activation
                 , weights    :: Weights
                 , momentums  :: Momentums
                 }

type Nodes = [Node]
type Activation = Double
type Weight = Double
type Weights = [Weight]
type Momentum = Double
type Momentums = [Momentum]

type PatternInput = [Double]
type PatternOutput = [Double]
type Pattern = (PatternInput,PatternOutput)
type Patterns = [Pattern]

type RandomState a = State StdGen a

main :: IO ()
main = do
  gen <- newStdGen
  let nn = initNetwork 2 2 1 gen
  let xor_patterns = [([0,0],[0])
                     ,([0,1],[1])
                     ,([1,0],[1])
                     ,([1,1],[0])]
  let nn' = trainFor nn xor_patterns 0.5 0.1 1000
  let results = test nn' xor_patterns
  mapM_ print results

----------------------------------
-- * Test
----------------------------------
test :: NeuralNetwork -> Patterns -> [String]
test _ [] = []
test nn pats = testPattern nn <$> pats

testPattern :: NeuralNetwork -> Pattern -> String
testPattern nn (i, _) =
  "Input : " ++ show i ++ " -> " ++ show (activation <$> o)
  where (NeuralNetwork _ _ o) = update nn i
----------------------------------
-- * Train
----------------------------------
trainFor :: NeuralNetwork -> Patterns -> Double -> Double -> Int -> NeuralNetwork
trainFor nn_i pats learnRate deltaP iterations =
  foldl (\nn _ -> train nn pats learnRate deltaP) nn_i [1..iterations]

train :: NeuralNetwork -> Patterns -> Double -> Double -> NeuralNetwork
train nn_i pats learnRate deltaP =
  foldl (\nn pat -> trainPattern nn pat learnRate deltaP) nn_i pats

trainPattern :: NeuralNetwork -> Pattern -> Double -> Double -> NeuralNetwork
trainPattern nn_i pat learnRate deltaP =
  backPropagate nn_u (snd pat) learnRate deltaP
  where nn_u = update nn_i $ fst pat
----------------------------------
-- * Update
----------------------------------
update :: NeuralNetwork -> PatternInput -> NeuralNetwork
update (NeuralNetwork i h o) inputs =
  let i' = applyInput i inputs
      h' = updateLayer h i'
      o' = updateLayer o h'
  in NeuralNetwork i' h' o'

applyInput :: Nodes -> PatternInput -> Nodes
applyInput nodes inputs =
  nodes' ++ [last nodes] -- the bias node
  where nodes' = zipNewActivation nodes inputs

updateLayer :: Nodes -> Nodes -> Nodes
updateLayer self prev =
  zipNewActivation self selfActivation
  where selfActivation = newActivation self prev

newActivation self prev =
  let indexes = take (length self) [0..]
      doNode i = fmap(\(Node a w _) -> a * (w!!i)) prev
      result = sum <$> fmap doNode indexes
  in sigmoid <$> result

zipNewActivation = zipWith (\node a -> node { activation = a })
----------------------------------
-- * BackPropagate
----------------------------------
backPropagate :: NeuralNetwork -> PatternOutput -> Double -> Double -> NeuralNetwork
backPropagate (NeuralNetwork i h o) outputs learnRate deltaP =
  --  deltaCalc takes a layer and the error of the layer
  let deltaCalc l e = zipWith(*) (dsigmoid <$> mapAc l) e
  --  ^ This could also be implemented with an arrow like v
  --  deltaCalc l e = (map dsigmoid >>> zipWith (*) $ mapAc l) e
  --  newLayer takes a layer and the delta of forward layer
      newLayer l d = backPropagateResults l d learnRate deltaP
  --
      outputErrors = zipWith (-) outputs (mapAc o)
      outputDeltas = deltaCalc o outputErrors
      hiddenErrors = sum . zipWith (*) outputDeltas . weights <$> h
      hiddenDeltas = deltaCalc h hiddenErrors
      h' = newLayer h outputDeltas
      i' = newLayer i hiddenDeltas
  in NeuralNetwork i' h' o

backPropagateResults :: Nodes -> [Double] -> Double -> Double -> Nodes
backPropagateResults nodes deltas learnRate deltaP =
  fmap (\node -> backPropagateResult node deltas learnRate deltaP) nodes

backPropagateResult :: Node -> [Double] -> Double -> Double -> Node
backPropagateResult (Node a w m) deltas learnRate deltaP =
  Node a w' m'
  where m' = (* a) <$> deltas
        w' = mzw deltaP m $ mzw learnRate m' w
        mzw x = map (* x) >>> zipWith (+)
----------------------------------
-- * Init
----------------------------------
initNetwork :: Int -> Int -> Int -> StdGen -> NeuralNetwork
initNetwork i' h o gen = do
  let i = i' + 1 -- input + 1 (bias node)
  let (iLayer, gen') = runState (initNodeLayer i h (-0.2, 0.2)) gen
  let (hLayer, _)    = runState (initNodeLayer h o (-2.0, 2.0)) gen'
  let (oLayer, _)    = runState (initNodeLayer o 0 (   0,   0)) gen'
  NeuralNetwork iLayer hLayer oLayer

initNodeLayer :: Int -> Int -> (Double, Double) -> RandomState Nodes
initNodeLayer ni no bnds = mapM (const $ initNode no bnds) [1 .. ni]

initNode :: Int -> (Double, Double) -> RandomState Node
initNode n bnds = randList bnds n >>= \w -> return $ Node 1.0 w $ replicate n 0.0
-----------------------------------
-- * Printers for debug
----------------------------------
doubleString :: Double -> String
doubleString d = printf "%.2f" d

doubleListString :: [Double] -> [String]
doubleListString line =
  fmap doubleString line

doubleListPrint :: [Double] -> String
doubleListPrint line =
  intercalate ", " lines
  where lines = doubleListString line

instance Show NeuralNetwork where
  show (NeuralNetwork i h o) = "NN" ++
    "\nAi : " ++ show (mapAc i) ++
    "\nAh : " ++ show (mapAc h) ++
    "\nAo : " ++ show (mapAc o) ++
    "\nWi : " ++ show (mapWe i) ++ "\nWo : " ++ show (mapWe h) ++
    "\nMi : " ++ show (mapMo i) ++ "\nMo : " ++ show (mapMo h)

instance Show Node where
  show (Node a w m) = "Node" ++
    "\nAc : " ++ (printf "%.2f" a) ++
    "\nWe : " ++ (doubleListPrint w) ++
    "\nMo : " ++ (doubleListPrint m)
----------------------------------
-- * Util
----------------------------------
mapAc = map activation
mapWe = map weights
mapMo = map momentums

getBoundedRandom :: Random a => (a,a) -> RandomState a
getBoundedRandom bnds = get >>= putRet . randomR bnds

runBoundedRandom :: Random a => (a,a) -> RandomState a
runBoundedRandom bnds = get >>= putRet . runState (getBoundedRandom bnds)

putRet :: MonadState s m => (b, s) -> m b
putRet (r, s) = put s >> return r

randList :: Random a => (a,a) -> Int -> RandomState [a]
randList bnds n = mapM (const $ runBoundedRandom bnds) [1..n]

-- This will actually use tanh to be consistent with bpnn.py
sigmoid :: Double -> Double
sigmoid x = tanh x

dsigmoid :: Double -> Double
dsigmoid x = 1.0 - x ** 2.0
