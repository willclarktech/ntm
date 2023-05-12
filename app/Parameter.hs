module Parameter where

data Parameter p = Parameter
  { value :: p,
    gradient :: p
  }
  deriving (Show)
