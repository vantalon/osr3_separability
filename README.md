# osr3_separability

This is the code supporting *Separable decompositions of 2 x n states with operator Schmidt rank three*. 

The figures of the paper can be reproduced thanks to `figure.py`. An example of end to end decomposition is written in `decomposition.py`.

We summarize the steps to construct the separable decomposition of a state $\rho$ with $osr(\rho) = 3$. 

*Step 1: reduction to the normal form in* `filtering.py`

(1) Compute
  
$$ 
\rho_1 = \frac{(V_1^\dagger \otimes I_n)\,\rho\,(V_1 \otimes I_n)}
                  {Tr\bigl[(V_1^\dagger \otimes I_n)\,\rho\,(V_1 \otimes I_n)\bigr]},
$$
  
  where $V_1$ is given by the Cariello SLOCC filtering (`cariello_filtering`). Then $\rho_1 = \rho_1^{T_A}$.

(2) If $\rho_{1,B}$ is not full rank, let $U_1$ be a unitary diagonalizing $\rho_{1,B}$, i.e.\ $\rho_{1,B} = U_1\bigl(\Lambda_m \oplus 0_{n-m}\bigr)U_1^\dagger$, and write

$$(I_2 \otimes U_1^\dagger)\,\rho_1\,(I_2\otimes U_1)
    = \begin{pmatrix}
        \rho_{2,(1,1)} & 0 & \rho_{2,(1,2)} & 0 \\
        0 & 0 & 0 & 0 \\
        \rho_{2,(2,1)} & 0 & \rho_{2,(2,2)} & 0 \\
        0 & 0 & 0 & 0
      \end{pmatrix}.$$
      
Then compute

$$\rho_2 = \begin{pmatrix}
               \rho_{2,(1,1)} & \rho_{2,(1,2)} \\
               \rho_{2,(2,1)} & \rho_{2,(2,2)}
             \end{pmatrix}  \in C^2\otimes C^m$$
  
(`full_rank_B`).
  
(3) Now that $\rho_{2,B}$ has full rank, let $\tau_B$ be the unique positive-definite Hermitian square root of $\rho_{2,B}$, i.e.\ $\tau_B^2 = \rho_{2,B}$. Compute

$$
    \rho_3 = \frac{(I_2 \otimes \tau_B^{-1})\,\rho_2\,(I_2 \otimes \tau_B^{-1})}
                  {Tr\bigl[(I_2 \otimes \tau_B^{-1})\,\rho_2\,(I_2 \otimes \tau_B^{-1})\bigr]},
$$

(`maximally_mixed_B_marginals`).Then $\rho_{3,B} = I_m/m$.
  

Define $C_1 = mTr_A\bigl[\rho_3\,(X \otimes I_m)\bigr]$ and $C_3 = m Tr_A\bigl[\rho_3\,(Z \otimes I_m)\bigr]$. We obtain the normal form

$$
  \rho_3 = \frac{1}{2m}\bigl(I_2 \otimes I_m + X\otimes C_1 + Z\otimes C_3\bigr).
$$

Let $T = C_1 + iC_3$ be the associated contraction, and $d = rank\,(I - T^\dagger T)^{1/2}$ its defect index.

*Step 2A: dilation for the pure-state decomposition in* `dilation.py`
Compute the dilation $U_2$ of $T$ on $\mathbb{C}^m \oplus \mathbb{C}^d$ given by Lem 6.5, and set $J_2 = \begin{psmallmatrix} \Id m \\ 0_d \end{psmallmatrix}$ for the isometry(` minimal_unitary_dilation`).

*Step 2B: dilation for the mixed-state decomposition in* `dilation.py`
Compute the dilation $U_3$ of $T$ given by Thm. 6.9 , and the isometry $J_3$ given in App. C (`unitary_dilation_with_shift`).

*Step 3: the decomposition with* `extract_decomposition` in `dilation.py`
Fix $j \in \{2,3\}$. Let $e^{i\theta_1},\dots,e^{i\theta_r}$ be the $r$ distinct eigenvalues of $U_j$, and $E_k$ the orthogonal projection onto the eigenspace of $e^{i\theta_k}$, $1 \le k \le r$. Define

$$
    L_k = J_j^\dagger E_k J_j .
$$

For each $k$ with $L_k \neq 0$, define

$$
  p_k = \frac{Tr(L_k)}{m}, \qquad
  \alpha_k = \tfrac12\bigl(\Id{2} + \cos\theta_k\,X + \sin\theta_k\,Z\bigr), \qquad
  \beta_k = \frac{L_k}{m\,p_k}.
$$
The decomposition of $\rho_3$ is then $\rho_3 = \sum_k p_k\,\alpha_k \otimes \beta_k$.

*Step 4: invert Step 1 with* `inverse_all_filterings` in `filtering.py`

(1) Compute  $\gamma_k = \frac{\tau_B \beta_k  \tau_B}{Tr \left(\tau_B \beta_k  \tau_B \right)}$ and $p'_k = p_k\frac{Tr \left[\tau_B \beta_k  \tau_B \right]}{Tr\bigl[(I_2 \otimes \tau_B)\,\rho_3\,(I_2 \otimes \tau_B)\bigr]}$. Then $\rho_2 =  \sum_k p'_k \alpha_k \otimes \gamma_k $.

 (2) If $\rho$ was not full rank, pad the $\gamma_k$ with $0$ matrices to obtain an operator on $C^n$, and compute $\zeta_k$

$$
    \zeta_k 
    = U_1 \begin{pmatrix}
        \gamma_k & 0 \\
        0 & 0_{n-m} &\\
      \end{pmatrix}  U_1^{\dagger}.
$$

Then $\rho_1 = \sum_k p'_k \alpha_k\otimes \zeta_k$.

(3)  Compute $\xi_k = \frac{V_1^{-\dagger } \alpha_k  V_1^{-\dagger}}{Tr [V_1^{-\dagger } \alpha_k  V_1^{-\dagger}]}$, and  $p''_k = p'_k\frac{Tr [V_1^{-\dagger } \alpha_k  V_1^{-\dagger}]}{Tr\bigl[(V_1^\dagger \otimes I_n)\,\rho_1\,(V_1 \otimes I_n)\bigr]]}$. Then the final decomposition is 

$$
  \rho =  \sum_k p''_k \xi_k \otimes \zeta_k .
$$
