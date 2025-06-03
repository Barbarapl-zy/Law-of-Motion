# Convert AR(2) to VAR(1)

$$
y_t = \rho_1 y_{t-1} + \rho_2 y_{t-2} + \varepsilon_t
$$

Forecaster correctly knows this structure and models the following state space model using a vector-form Kalman filter:

- State Equation:
	$$
	x_{t+1} = A x_t + w_t, \quad 
	
	x_t = \begin{bmatrix}
	y_t \\
	y_{t-1}
	\end{bmatrix}, \quad
	
	A = \begin{bmatrix}
	\rho_1 & \rho_2 \\
	1 & 0
	\end{bmatrix}
	$$

- Observation Equation:
	$$
	s_t^{(i)} = H x_t + v_t^{(i)}, \quad H = [1 \ 0]
	$$

------

# Construct the state space model

**linear Gaussian state space model**：

- **State transition (VAR(1))**:
	$$
	x_{t+1} = A x_t + w_t, \quad w_t \sim \mathcal{N}(0, Q)
	$$

- **Observation equation (signal)**:
	$$
	s_t^{(i)} = H x_t + v_t^{(i)}, \quad v_t^{(i)} \sim \mathcal{N}(0, R)
	$$
	Where, $H = [1 \quad 0]$，represents we observe $y_t$, rather than the entire state.

So each forecaster receives a noisy signal:
$$
s_t^{(i)} = y_t + \text{noise}
$$

------

# Forecaster uses the vector-form Kalman filter

- **Prediction**：
	$$
	\hat{x}_{t|t-1} = A \hat{x}_{t-1|t-1}
	$$

	$$
	P_{t|t-1} = A P_{t-1|t-1} A' + Q
	$$

- **Update**：
	$$
	K_t = P_{t|t-1} H' (H P_{t|t-1} H' + R)^{-1}
	$$

	$$
	\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (s_t - H \hat{x}_{t|t-1})
	$$

	$$
	P_{t|t} = (I - K_t H) P_{t|t-1}
	$$

> As for:
> $$
> \hat{y}_{t+h|t} = E[y_{t+h} \mid \text{data up to time } t]
> $$
> Since we already have $\hat{x}_{t|t}$, then:
> $$
> \hat{x}_{t+1|t} = A \hat{x}_{t|t}, \quad \hat{x}_{t+2|t} = A^2 \hat{x}_{t|t}, \quad \ldots, \quad \hat{x}_{t+h|t} = A^h \hat{x}_{t|t}
> $$
> Recall that:
> $$
> y_{t+h} = \text{the first dimension of } x_{t+h}
> $$
> which means:
> $$
> \hat{y}_{t+h|t} = H \hat{x}_{t+h|t} = H A^h \hat{x}_{t|t}, \quad \text{where } H = [1 \quad 0]
> $$

Therefore, the final forecast is:
$$
\hat{y}_{t+h|t} = [1 \quad 0] \cdot A^h \hat{x}_{t|t}
$$


------

# Analyst uese Eq(7) and Eq(8)

Now the analyst gets the forecast from forecaster：
$$
\hat{y}_{t+h|t} = \text{from Kalman + VAR(1)} = H A^h \hat{x}_{t|t}
$$

$$
\hat{y}_{t|t} = H \hat{x}_{t|t}
$$

$$
H = [1 \quad 0]
$$

Then the analyst uses Eq(7) and Eq(8) to test whether the forecast conforms to the structure of an AR(1):

Eq(7)：
$$
\left(\frac{\tilde{\rho}}{\rho}\right)^h = \frac{\text{Cov}(\hat{y}_{t+h|t}, y_t)}{\text{Cov}(\hat{y}_{t|t}, y_{t+h})}
$$
Eq(8)：
$$
\text{stat}_8 = \frac{\text{Cov}(\hat{y}_{t+h|t}, \hat{y}_{t|t})}{\text{Var}(\hat{y}_{t|t})} \cdot \frac{\text{Var}(y_t)}{\text{Cov}(y_{t+h}, y_t)}
$$