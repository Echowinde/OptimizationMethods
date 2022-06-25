# OptimizationMethods
Python based practical optimization methods (only use Numpy)  
徐翔老师优化算法课的作业代码，算法只使用Numpy库

## Available methods
### Unconstrained optimization methods
- Steepest Descent Method  
  1. Inexact line search (Goldstein condition)
  2. Inexact line search (Wolfe condition)
- Newton Method  
  1. Line Search Newton
  2. Modified Newton (Goldstein-Price)
  3. Modified Newton (Goldfeld)
- Quasi-Newton Method
  1. BFGS
  2. DFP
  3. SR1
- Conjuagate Gradient Method
  1. Fletcher-Reeves formula (FP)
  2. Polak-Ribiere-Polyak formula (PRP)
  3. Dai-Yuan formula (DY)
- Preconditioned Conjugate Gradient Method
  1. Jacobi preconditioning
  
### Constrained optimization methods
- Active Set Method
- Penalty Function Method
  1. Quadratic penalty
  2. Classical $l^1$ penalty
- Augmented Lagrangian Method