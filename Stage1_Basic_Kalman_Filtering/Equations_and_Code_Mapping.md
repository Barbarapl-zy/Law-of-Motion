This document describes each mathematical equation used in the simulation and its corresponding MATLAB code block.

---

Compute Steady-State Kalman Gain via Riccati Equation
===

**P: forecast error variance**
$$
P_t = \text{Var}(y_t - \hat{y}_{t|t-1})
$$
**Discrete Riccati equation:**
$$
P = \rho^2 \left( P - \frac{P^2}{P + R} \right) + Q
$$

- $\frac{P^2}{P + R}$ represents how much uncertainty is reduced by observing the noisy signal.
- The new $P$ incorporates:
	- the forecast error from the previous time step, and
	- the structure of the signal extraction problem.
- The loop stops when the change in $P$ is smaller than a tiny threshold `tol`.

**Kalman gain:**
$$
\boxed{K = \frac{P}{P + R}}
$$

```matlab
Q = 1;                   % Process noise variance
R = omega^2;             % Observation noise variance
rho = rho_hat;           % Subjective rho used by forecasters

P = 1;                   % Initial value
tol = 1e-10;             % Convergence threshold
max_iter = 1000;         % Maximum number of iterations

for iter = 1:max_iter
    P_new = rho^2 * (P - (P^2)/(P + R)) + Q;
    if abs(P_new - P) < tol
        break;
    end
    P = P_new;
end

K = P / (P + R);         % Steady-state Kalman gain
```



# Data Generation: AR(1) Process

**Mathematical Formula:**
$$
y_t = \rho_{\text{true}} \cdot y_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)
$$


**MATLAB Code:**
```matlab
N = 500;    
T = 80;          
M = 100;          
rho_true = 0.5;   
rho_hat = 0.8;    
omega = 0.5;      
h = 1;            

stat_eq7 = zeros(N,1);
stat_eq8 = zeros(N,1);

for rep = 1:N
    y = zeros(T,1);
    eps = randn(T,1);
    for t = 2:T
        y(t) = rho_true * y(t-1) + eps(t);
    end

```

---

# Signal Observation and Kalman Belief Updating

**Signal Observation:**
$$
s_{i,t-1} = y_{t-1} + \omega \cdot \varepsilon_{i,t}, \quad \varepsilon_{i,t} \sim \mathcal{N}(0,1)
$$
**Kalman Update (Steady-State):**
$$
\hat{y}_{i,t-1} = K \cdot s_{i,t-1}
$$
**MATLAB Code:**

```matlab
    signals = zeros(M,T-1);
    yhat = zeros(M,T-1);
    for i = 1:M
        for t = 2:T
            signals(i,t-1) = y(t-1) + omega * randn();
            K = 1 / (1 + omega^2);  %revise
            yhat(i,t-1) = K * signals(i,t-1);
        end
    end

```

---

# Individual Forecasting (h-step ahead)

**Formula:**
$$
f_{i,t} = \widetilde{\rho}^h \cdot \hat{y}_{i,t}
$$
**MATLAB Code:**

```matlab
    pred = zeros(M,T-h);
    for i = 1:M
        for t = 1:(T-h)
            pred(i,t) = rho_hat^h * yhat(i,t);
        end
    end

```

---

# Consensus Forecasting

**Consensus Forecast of Future:**
$$
\mathbb{F}_t y_{t+h} = \frac{1}{M} \sum_{i=1}^M f_{i,t}
$$


**Consensus Belief about Present:**
$$
\mathbb{F}_t y_t = \frac{1}{M} \sum_{i=1}^M \hat{y}_{i,t}
$$
**MATLAB Code:**

```matlab
Ft_ytph = mean(pred,1)';
Ftyt = mean(yhat,1)';
```

---

# Preparing True Variables

**True Past and Future States:**
$$
y_t = y(1:T-h), \quad y_{t+h} = y((1+h):T)
$$
**MATLAB Code:**

```matlab
yt = y(1:T-h);
ytph = y((1+h):T);
```

---

# Equation (7) Statistic (Self-Adjoint Test)

**Formula:**
$$
\text{stat}_7 = \frac{\text{Cov}(\mathbb{F}_t y_{t+h}, y_t)}{\text{Cov}(\mathbb{F}_t y_t, y_{t+h})}
$$
**MATLAB Code:**
```matlab
cov1 = cov(Ft_ytph, yt);    
cov2 = cov(Ftyt, ytph);    
stat_eq7(rep) = cov1(1,2) / cov2(1,2);
```

---

# Equation (8) Statistic (Structure Error Test)

**Formula:**
$$
\text{stat}_8 = 
\left( \frac{\text{Cov}(\mathbb{F}_t y_{t+h}, \mathbb{F}_t y_t)}{\text{Var}(\mathbb{F}_t y_t)} \right)
\cdot
\left( \frac{\text{Var}(y_t)}{\text{Cov}(y_{t+h}, y_t)} \right)
$$
**MATLAB Code:**

```matlab
cov3 = cov(Ft_ytph, Ft_ytph);  
cov4 = cov(ytph, Ft_ytph);    
cov5 = cov(yt, yt);           
cov6 = cov(ytph, yt);        
stat_eq8(rep) = (cov4(1,2)/cov3(1,1)) * (cov5(1,1)/cov6(1,2));
```
