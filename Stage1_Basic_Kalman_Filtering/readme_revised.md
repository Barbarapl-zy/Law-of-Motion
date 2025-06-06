参数设置
===

代码开头设置了一系列模拟参数，用于构建**AR(1)模型**及预测者的观测与预测环境：

- `N = 500`：Monte Carlo模拟次数，即生成500组独立的时间序列样本用于统计分析。这有助于评估统计量的分布和稳定性。
- `T = 80`：时间序列长度，每组样本包含80期的数据点。例如可理解为80期宏观经济观测值。
- `M = 100`：预测者数量。在每组模拟中，有100位预测者同时存在。引入多个预测者并取平均可减少观测噪声对预测的影响，相当于模拟**共识预期**的形成过程。
- `rho_true = 0.5`：真实的AR(1)模型系数$\rho_{\text{true}}$。这表示实际数据遵循$y_t = 0.5 , y_{t-1} + \varepsilon_t$，即存在中等程度的持续性（前一期的一半影响延续到下一期）。$\varepsilon_t$通常被视为均值为0的随机扰动（代码中由标准正态`randn`生成）。
- `rho_hat = 0.8`：预测者主观感知的AR(1)系数$\rho_{\text{hat}}$。预测者**误认为**过程更具有持续性（0.8而非真实的0.5），体现了一种**认知偏误**：他们**高估了**序列的持续性（或说具有**过度外推**的倾向）。这一偏差将在预测过程中体现出来。
- `omega = 0.5`：信号噪声的标准差$\omega$，表示预测者每期观测到的信号含有随机噪声的幅度。例如，如果真实值是$y_t$，预测者只能看到$y_t$被叠加了方差为$\omega^2=0.25$的噪声。这模拟了信息不完美的情况（**有噪信息**或**不完全信息**）。
- `h = 1`：预测期数，表示预测者关心一步Ahead的预测（例如在$t$期预测$t+1$期的值）。代码支持一般的步长$h$，但此处设为1代表**一阶预测**。

此外，代码预先分配了`stat_eq7`和`stat_eq8`为长度$N$的零向量，用于存储每次模拟计算得到的统计量(7)和统计量(8)。这样预分配有助于提高MATLAB运算效率。

蒙特卡罗模拟过程
===

## Step 1: 生成AR(1)序列（真实数据产生过程）

**代码片段：**

```
matlabCopyEdit% Step 1: 生成AR(1)序列
y = zeros(T,1);
eps = randn(T,1);
for t = 2:T
    y(t) = rho_true * y(t-1) + eps(t);
end
```

**解释与背景：**  这里在每次模拟中生成一个长度为$T$的时间序列${y_t}$，服从AR(1)模型动态：

- 初始时刻$y(1)$被初始化为0（考虑`y = zeros(T,1)`），相当于初始状态给定零。更严格的做法可能是让初值服从稳态分布或丢弃初期若干值，但在$T=80$的情况下，初值影响相对有限。
- 随机扰动$\varepsilon_t$通过`randn`生成，假设$\varepsilon_t \sim \mathcal{N}(0,1)$独立同分布。这意味着每期都有一个方差1的噪声冲击推动$y$的演化。
- 递推公式 `y(t) = rho_true * y(t-1) + eps(t)` 。由于$\rho_{\text{true}}=0.5$，实际过程具有中等粘性：当前值倾向于回归均值（均值为0）且受到上一期值的一半影响。同时，$\varepsilon_t$引入新的随机波动。这是典型的**一阶自回归过程**，常见于经济和金融时间序列建模。例如，GDP增长率、通胀率等常被近似为AR(1)过程来捕捉持续性。

**数学描述：** 如果我们假设起始足够久远或过程平稳，则在该AR(1)模型中真实参数$\rho_{\text{true}}=0.5 < 1$保证平稳性。平稳状态下$y_t$的方差为$\sigma_y^2 = \frac{\sigma_\varepsilon^2}{1-\rho_{\text{true}}^2} = \frac{1}{1-0.25} ≈ 1.33$，而滞后一阶协方差为$\operatorname{Cov}(y_{t}, y_{t-1}) = \rho_{\text{true}} \sigma_y^2 ≈ 0.5 \times 1.33 = 0.667$。这些理论值在后续统计量计算中构成基准。

## Step 2: 信号观测与 Kalman 滤波更新

**代码片段：**

```
matlabCopyEdit% Step 2: 模拟预测者观测带噪signal并进行Kalman更新
signals = zeros(M,T-1);
yhat = zeros(M,T-1);
for i = 1:M
    for t = 2:T
        signals(i,t-1) = y(t-1) + omega * randn();
        K = 1 / (1 + omega^2);  % Kalman gain
        yhat(i,t-1) = K * signals(i,t-1);
    end
end
```

**解释与背景：** 在这一步，每位预测者$i$在每一期通过**带噪信号**来观察前一期的真实值，并使用**Kalman滤波**方法更新对该真实值的估计。

- `signals(i,t-1) = y(t-1) + omega * randn();`：预测者$i$在时间$t$获得关于$y_{t-1}$的观测信号。由于存在观测误差，信号=“真实值 + 噪声”。噪声项$\omega * randn()$服从$\mathcal{N}(0,\omega^2)$，这里$\omega=0.5$意味着观测噪声标准差为0.5。**直观地**，如果真实值$y_{t-1}=2$，观测可能是2加减一定随机扰动（大约95%概率落在$2\pm1$范围）。这模拟了预测者无法精确知晓过去实际值的情况（如统计数据有测量误差、信息获取滞后等）。
- `K = 1/(1 + omega^2);`：Kalman增益系数$K$的计算。在一般Kalman滤波中，$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$，其中$P_{t|t-1}$是预测协方差，$R$是观测噪声方差。本代码将$K$固定为常数$1/(1+\omega^2)$，这对应于一种简化情形：假定预测者对前一期$y$的先验方差为1（单位方差不失一般性），观测噪声方差为$\omega^2$，则稳态Kalman增益$K^* = \frac{1}{1+\omega^2}$。代入$\omega=0.5$，可得$K = \frac{1}{1+0.25} = 0.8$。**这意味着预测者将观测信号中的有用信息赋予80%的权重**。Kalman增益越高，代表观测越精确、预测者越相信新的信号。
- `yhat(i,t-1) = K * signals(i,t-1);`：预测者$i$对$y_{t-1}$的**滤波估计**（posterior）记作$\hat{y}*{t-1|t}$，这里通过简单加权得到。由于先验均值被暗含设为0（代码未使用过去估计做预测，等效于每期重新估计），更新公式实际退化为$\hat{y}*{t-1|t} = K \cdot \text{signal}*{t-1}$。因为信号 = 真值 + 噪声，且$\mathbb{E}[\text{signal}*{t-1}|y_{t-1}] = y_{t-1}$，**滤波后的期望**为$K \cdot y_{t-1}$。在本例中$K=0.8$，说明预测者**低估**了过去真实值的幅度（例如真实$y_{t-1}=2$，则$\hat{y}\approx1.6$），这是因为他们对观测噪声留了20%的保守余量。这种低估源于信息不完全：预测者不愿完全相信单期的嘈杂信号。

**经济学含义：** 此步模拟了预测者通过嘈杂信息形成对过去真实经济变量的认知。例如，经济分析师没有直接观察到真实GDP，而是通过有误差的统计报告来推断GDP水平。Kalman滤波提供了**最佳线性无偏**估计框架：预测者将自己的先验（这里取0的中立先验）与新观测加权平均，以得到对真实状态的后验最佳估计。在简化假设下，Kalman增益的表达清晰显示了观测信心的作用：噪声越小（$\omega^2$越低），$K$越接近1，观测被几乎完全信任；噪声越大，$K$越小，预测者更不信任当前信号，依赖先验更多。**本代码将$K$固定不变**，等效于假定系统处于稳态或每期情况类似，未对Kalman增益进行时间序列递推更新（这是一个可改进之处，下文讨论）。

值得注意的是，这里每位预测者都是**相互独立**地进行信号观测和更新，并未共享信息。虽然代码在一个双重循环中实现，但完全可以向量化：例如一次性生成所有`signals`矩阵（100×79），然后计算`yhat = K * signals`逐元素乘即可。在MATLAB中向量化计算会更高效。

## Step 3: 根据主观模型进行预测

**代码片段：**

```
matlabCopyEdit% Step 3: 预测者构造未来预测
pred = zeros(M,T-h);
for i = 1:M
    for t = 1:(T-h)
        pred(i,t) = rho_hat^h * yhat(i,t);
    end
end
```

**解释与背景：**  在获得对过去各期的状态估计$\hat{y}_{t}$后，每位预测者开始基于自己对模型的理解做出**一步ahead预测**。$h=1$，公式简化为：

$\text{pred}(i,t) = \rho_{\text{hat}} \cdot \hat{y}_{i,t} , \quad \text{其中}\ \hat{y}_{i,t} = \mathbb{E}[y_t \mid \text{信息}_{i,t}]$

- `rho_hat^h * yhat(i,t)`：对于预测者$i$，在时期$t$时可获得对当期$y_t$的估计$\hat{y}*{i,t}$（注意：在代码索引中，`yhat(i,t)`对应我们数学记号的$\hat{y}*{i,t}$）。预测者相信模型的动态系数是$\rho_{\text{hat}}=0.8$，因此他预测下一期$y_{t+1}$为$0.8 \times \hat{y}*{i,t}$。**这是一个主观预测**，因为如果预测者是理性的且信息充分，他本应使用真实系数0.5来预测，即$\mathbb{E}[y*{t+1}|I_t] = 0.5,\hat{y}*{i,t}$。在这里，由于$\rho*{\text{hat}} > \rho_{\text{true}}$，预测者会**高估未来值**（例如$\hat{y}_{i,t}=1$时，他预测下期为0.8，而真实模型期望仅0.5）。这种偏差体现了他们的**误感知动态**（misperceived dynamics），具体来说是一种**extrapolative beliefs**行为：相信当前状况对未来的影响更持久。
- 循环范围`t = 1:(T-h)`确保我们仅预测到第$T$期的前$T-h$期为止（因为预测$t+h$期需要已知$t$的信息）。当$h=1$时，$t$从1到79，对应预测$y_2,...,y_{80}$。换言之，**第$t$期作出的预测针对的是第$t+1$期的实际值**。代码在Monte Carlo中先后计算所有这些预测，但在真实情景中，预测者会依次随时间滚动做出预测并在下一期看到真实结果。

**经济背景：**  这一步模拟了**预测者的预期形成**过程。每位预测者用其对当前状况的最佳估计$\hat{y}*{i,t}$（源于Kalman滤波处理后的信息）通过他相信的模型（参数$\rho*{\text{hat}}$）来预测下一期。由于$\rho_{\text{hat}} \neq \rho_{\text{true}}$，预测将系统性偏离真实演化。这种**系统偏差**在行为经济学和宏观领域非常重要：如果$\rho_{\text{hat}} > \rho_{\text{true}}$，预测者**高估**了持续性，会过度响应近期变动，预测显示出惯性过强；若$\rho_{\text{hat}} < \rho_{\text{true}}$，则预测者**低估**持续性，表现为**欠反应**或过度均值回归预期。这些现象对应于现实中的“趋势追踪”或“锚定效应”等认知偏误。这里设置$\rho_{\text{hat}}=0.8$ vs $\rho_{\text{true}}=0.5$意味着预测者有明显的趋势追踪偏误。

**改进建议：** 当前代码每期预测仅利用当期对$y_t$的估计$\hat{y}*{t}$，**并未使用更早期的信息进行动态更新**。一个更严谨的做法是让预测者在Kalman滤波中考虑模型动态：即使用上一期后验$\hat{y}*{t-1|t-1}$通过$\rho_{\text{hat}}$推出当期先验，再结合新信号更新$\hat{y}*{t|t}$。这种**全Kalman滤波**会涉及递推公式 $\hat{y}*{t|t-1} = \rho_{\text{hat}}\hat{y}_{t-1|t-1}$ 以及相应误差协方差的演化。但实现起来较复杂。本代码采取的简化方法是每期独立地估计当期$y_t$（相当于先验均值重置为0），因此可能低估了状态的持续性。这种简化可能使预测者信息利用不足，但在稳态条件下仍能反映误差的影响。若对模型进行增强，可以尝试引入上述Kalman预测-更新循环，以更准确地体现预测者的信念形成过程。

## Step 4: 共识预测与实际值序列

**代码片段：**

```
matlabCopyEdit% Step 4: 共识预测与实际数据
Ft_y_tph = mean(pred,1)';                  % E[y_{t+h} | info_t]
Ft_y_t   = rho_hat^h * mean(yhat(:,1:T-h),1)'; % E[y_t | info_t]
yt   = y(1:T-h);                             % y_t
ytph = y((1+h):T);                           % y_{t+h}
```

**解释与背景：**  这一部分汇总了所有预测者的信息，得到**共识预测**以及对应的真实值序列，用于后续计算统计量。具体如下：

- `Ft_y_tph`：定义为 `mean(pred,1)'`，即对所有预测者第$t$期作出的预测取平均（再转置为列向量）。由于`pred(i,t) = \rho_{\text{hat}} \hat{y}*{i,t}` 是预测者$i$在$t$期对$t+1$期的预测，取平均后我们得到共识预测。用符号表示，即 $F_t y*{t+1} = \frac{1}{M}\sum_{i=1}^M \text{pred}(i,t)$。代码注释将其标为$E[y_{t+h}|info_t]$，意指在$ $t$期全部信息下对未来的期望。
- `Ft_y_t`：定义为 `rho_hat^h * mean(yhat(:,1:T-h),1)'`。注意$h=1$时，这等价于 $\rho_{\text{hat}} * \frac{1}{M}\sum_i \hat{y}*{i,t}$。由于$\hat{y}*{i,t}$是预测者$i$对当期真实值$y_t$的滤波估计，平均后得到共识下对$y_t$的**最佳估计**。再乘以$\rho_{\text{hat}}$，其实与前述$F_t y_{t+1}$结果一致（因为线性运算和平均可交换）。因此，这个变量可以理解为**共识对当前状态的估计经模型映射后的预测**，本质上应等于$F_t y_{t+1}$。不过从概念上，可以将$F_t y_t = \frac{1}{M}\sum_i \hat{y}*{i,t}$视为“在$t$期信息下对当前真实状态的共识估计”，然后$F_t y*{t+1} = \rho_{\text{hat}} F_t y_t$。因此代码中先计算了$F_t y_t$再计算$F_t y_{t+h}$，两者满足$F_t y_{t+h} = \rho_{\text{hat}}^h F_t y_t$。
- `yt` 和 `ytph`：分别提取真实序列中的$y_t$和$y_{t+h}$（这里$h=1$所以$y_{t+1}$）。`yt = y(1:T-h)`对应$[y_1, y_2, ..., y_{T-1}]$，`ytph = y(1+h:T)`对应$[y_2, y_3, ..., y_T]$。这样$yt$和$y_{t+h}$一一对齐：$ytph$中的第$t$项就是实际的$y_{t+1}$。这些将用于同上述预测序列一起计算协方差。

**作用：**  通过取平均，我们获得了群体预测的平均看法，可视为**全体预测者的共识预期**。在现实中，这类似于对某指标的市场平均预测或专业机构预期的均值。因为$M=100$较大，平均可以在一定程度上抵消各预测者观测噪声的随机部分（噪声期望为0）。因此，$F_t y_t$相比单个预测者的$\hat{y}*{i,t}$更接近真实$y_t$，$F_t y*{t+1}$也更接近真正的$\mathbb{E}[y_{t+1}|I_t]$。当然，由于每个预测者都有相同的模型偏误$\rho_{\text{hat}}$，这种平均**无法消除系统性的预测偏差**：如果所有人都高估持久性，那共识也会高估。

**改进建议：**  本步骤计算$F_t y_t$和$F_t y_{t+h}$的方式有重复之嫌，因为按线性性$F_t y_{t+h} = \rho_{\text{hat}}^h F_t y_t$，无需重复计算。此外，在代码实现上，完全可以在先前步骤直接对`yhat`和`pred`进行平均计算以避免储存整个矩阵。如：`mean_yhat = mean(yhat)`，`mean_pred = rho_hat^h * mean_yhat`，得到相同结果。对大规模$M$而言，这些向量化和避免冗余计算的手段可以节省运行时间和内存。

统计量 (7) 的计算与意义
===

**代码片段：**

```
matlabCopyEdit% Step 5: 统计量 (7)
cov1 = cov(Ft_y_tph, yt);
cov2 = cov(ytph, Ft_y_t);
stat_eq7(rep) = cov1(1,2) / cov2(1,2);
```

在每次模拟生成数据并形成预测之后，代码计算两个协方差量，进而得到**统计量(7)**：

- `cov1 = cov(Ft_y_tph, yt)`：计算序列$F_t y_{t+h}$与$y_t$之间的样本协方差。根据代码逻辑，这里对应$\operatorname{Cov}(F_t y_{t+1},, y_t)$。
- `cov2 = cov(ytph, Ft_y_t)`：计算序列$y_{t+h}$与$F_t y_t$之间的样本协方差，即$\operatorname{Cov}(y_{t+1},, F_t y_t)$。
- 由于MATLAB的`cov`函数返回协方差矩阵，这里取其[1,2]元素即两个向量之间的协方差值。
- `stat_eq7 = cov1(1,2) / cov2(1,2)`：统计量(7)被定义为上述两个协方差的比值：

$$
\text{Stat(7)} = \frac{\operatorname{Cov}(F_t y_{t+h},\; y_t)}{\operatorname{Cov}(y_{t+h},\; F_t y_t)}
$$

对于$h=1$的情况，可简写为 $\displaystyle \text{Stat(7)} = \frac{\operatorname{Cov}(F_t y_{t+1},, y_t)}{\operatorname{Cov}(y_{t+1},, F_t y_t)}$。

**统计涵义：**  在模型中，可以证明如果预测者的感知与真实一致，则该比值应接近1；若存在误感知，且$\tilde{\rho}$表示预测者主观系数，$\rho$为真实系数，那么实际上：
$$
(\frac{\tilde{\rho}}{\rho})^h = \frac{\operatorname{Cov}(F_t y_{t+h},\, y_t)}{\operatorname{Cov}(y_{t+h},\, F_t y_t)}
$$
当$h=1$时，这意味着$\text{Stat(7)} \approx \frac{\tilde{\rho}}{\rho}$。因此，**统计量(7)**直接刻画预测者主观持久性相对于真实持久性的比例。如果预测者高估持久性（$\tilde{\rho} > \rho$），此比值大于1；反之若低估，则比值小于1。**这为检验预测偏误提供了简单判据**：理性且无偏时应为1，显著偏离1则表示存在系统性误感知。

**验证推导：** 为什么这个协方差比能够反映$\tilde{\rho}/\rho$呢？考虑近似的直觉验证：若信息完全且预测者只是模型参数有偏，则：

- $F_t y_{t+1}$（共识预测）大致等于$\tilde{\rho},y_t$（因为预测者完全知道当前$y_t$，只不过用了错误系数预测未来）。
- 因此$\operatorname{Cov}(F_t y_{t+1},, y_t) \approx \tilde{\rho},\operatorname{Var}(y_t)$。
- 而$\operatorname{Cov}(y_{t+1},, F_t y_t)$：共识对当前$y_t$的估计在信息充分时就是$y_t$本身，所以$F_t y_t \approx y_t$，从而$\operatorname{Cov}(y_{t+1},, F_t y_t) \approx \operatorname{Cov}(y_{t+1},, y_t) = \rho,\operatorname{Var}(y_t)$（因为真实模型$y_{t+1}=\rho,y_t+\varepsilon_{t+1}$，协方差为$\rho,\mathrm{Var}(y_t)$）。两者比值约$\tilde{\rho}/\rho$。
- 在本模拟中信息并非完全（带噪），但经过平均后$F_t y_t$仍是对$y_t$的有偏估计，协方差计算会引入一些衰减。但**关键是**这个比值不需要知道$\operatorname{Var}(y_t)$即可消除尺度影响，因此预期仍围绕$\tilde{\rho}/\rho$上下波动。

**经济解释：** Stat(7)利用当前实际值与未来预测之间，以及当前预测与未来实际之间的线性关联强度之比，来判断预测者对动态的认识是否正确。可以想象，如果预测者高估了$\rho$，他们的预测$F_t y_{t+1}$会对$y_t$非常敏感（协方差偏大），而实际$y_{t+1}$对预测者估计的$y_t$的依赖相对较小（协方差偏小），导致比值>1。这种方法巧妙地通过**自伴随的协方差结构**放大了偏误的效应，因此被称为自伴方法。

**实现准确性：** 代码正确地按照定义计算了协方差并取比值。不过需注意MATLAB中`cov(X,Y)`采用样本协方差（分母$T-1$）计算，但由于取比值，采样分母会相互抵消，因而Stat(7)本身与使用总体协方差计算结果差别很小。一个潜在改进是**直接向量化**计算协方差比值**：** 由于协方差比可以写成 $\frac{\mathbb{E}[F_t y_{t+1}\cdot y_t] - \mathbb{E}[F_t y_{t+1}]\mathbb{E}[y_t]}{\mathbb{E}[y_{t+1}\cdot F_t y_t] - \mathbb{E}[y_{t+1}]\mathbb{E}[F_t y_t]}$，可进一步简化计算流程。不过在这里直接调用`cov`清晰易读且影响不大。

统计量 (8) 的计算与意义
===

**代码片段：**

```
matlabCopyEdit% Step 6: 统计量 (8)
covA = cov(Ft_y_tph, Ft_y_t);  % cov(F_t y_{t+h}, F_t y_t)
varF = var(Ft_y_t);
covB = cov(ytph, yt);
stat_eq8(rep) = (covA(1,2)/varF) * (var(yt)/covB(1,2));
```

代码第6步计算**统计量(8)**，涉及四个量：

- `covA = cov(Ft_y_tph, Ft_y_t)`：共识预测序列自身的协方差，具体为$\operatorname{Cov}(F_t y_{t+h},, F_t y_t)$（这里$h=1$）。
- `varF = var(Ft_y_t)`：共识对当前$y_t$估计的方差，即$\operatorname{Var}(F_t y_t)$。
- `covB = cov(ytph, yt)`：真实序列自身的一期滞后协方差，即$\operatorname{Cov}(y_{t+h},, y_t)$。
- 此外代码用`var(yt)`计算$\operatorname{Var}(y_t)$，但其实$\operatorname{Var}(y_t)$等同于`cov(yt,yt)`对角元素；为简洁我们直接记$\sigma_y^2 = \operatorname{Var}(y_t)$。

随后 `stat_eq8` 被计算为：
$$
\text{Stat(8)} = \frac{covA(1,2)}{varF} \times \frac{var(yt)}{covB(1,2)}
$$


代入含义可写为：
$$
\text{Stat(8)} = \frac{\operatorname{Cov}(F_t y_{t+h},\, F_t y_t)}{\operatorname{Var}(F_t y_t)} \;\times\; \frac{\operatorname{Var}(y_t)}{\operatorname{Cov}(y_{t+h},\, y_t)}
$$


当$h=1$时具体为 $\displaystyle \text{Stat(8)} = \frac{\operatorname{Cov}(F_t y_{t+1},, F_t y_t)}{\operatorname{Var}(F_t y_t)} \cdot \frac{\operatorname{Var}(y_t)}{\operatorname{Cov}(y_{t+1},, y_t)}$。

**统计涵义：**  统计量(8)来源于Reis (2020)提出的另一种检验方法。它通过**回归法**直接估计预测者感知的系数$\tilde{\rho}$和实际系数$\rho$，然后取其比值来判断偏误：

- 第一部分 $\displaystyle \frac{\operatorname{Cov}(F_t y_{t+h},, F_t y_t)}{\operatorname{Var}(F_t y_t)}$ 实际上是将**未来共识预测**对**当前共识预测**做回归的斜率估计。这近似等于预测者主观认知的持久性$\tilde{\rho}^h$（对于$h=1$，就是$\tilde{\rho}$）。换言之，若我们做回归 $F_t y_{t+1} = \beta \cdot F_t y_t + \text{误差}$，则$\beta$的估计就是这一项【14†】。
- 第二部分 $\displaystyle \frac{\operatorname{Var}(y_t)}{\operatorname{Cov}(y_{t+h},, y_t)}$ 可视为**真实持久性**的倒数。因为对真实AR(1)过程，$\operatorname{Cov}(y_{t+1}, y_t) = \rho_{\text{true}},\operatorname{Var}(y_t)$，所以 $\frac{\operatorname{Var}(y_t)}{\operatorname{Cov}(y_{t+1}, y_t)} \approx \frac{1}{\rho_{\text{true}}}$。
- 将两者相乘，理论上得到 $(\tilde{\rho}^h) \times (1/\rho)$，也就是 $(\tilde{\rho}/\rho)^h$。在$h=1$时，Stat(8)同样应近似$\tilde{\rho}/\rho$。

**优势及敏感性：** 从原理上看，统计量(8)几乎**直接估计**出了预测者主观参数：特别地，在本模型信息充分的极限下，Stat(8)精确满足 $\text{Stat(8)} = \tilde{\rho}/\rho$。换言之，如果没有观测噪声干扰，统计量(8)的倒数乘以真实参数$\rho_{\text{true}}$，就能**近乎直接地得到预测者主观认知的参数**$\tilde{\rho}$。因此，相较统计量(7)，Stat(8)被认为对误识动态更加敏感：它等价于**分别捕捉预测者眼中的持久性和实际持久性，再比较两者**。在存在观测误差的现实场景中，这种方法仍然具有一定鲁棒性——因为通过对预测序列内部的相关性归一化，减轻了预测方差较小或信息不充分带来的偏差影响。

**对比Stat(7)：** 两种统计量其实在大样本下估计相同的理论量$(\tilde{\rho}/\rho)^h$。但在有限样本和噪声情形下，可能表现出不同的方差和偏差属性。Stat(7)用交叉协方差直接构造，比值形式简单，一步到位；Stat(8)则基于两个回归斜率的积来实现，逻辑上更清晰地分离了“主观”与“客观”动态。因此，如果预测者信息精度提高（$\omega$减小），Stat(7)和Stat(8)都会更接近真实比值，但Stat(8)能够更直接地反映主观参数变化。当误差较大时，Stat(7)的分子分母都会受到衰减影响，不一定正好抵消，而Stat(8)通过方差归一在一定程度上校正了尺度。例如，如果$F_t y_t$由于信息不全被低估了一个因子$K$，Stat(7)的比值可能仍为$\tilde{\rho}/\rho$（如先前推导示例中奇迹般抵消），但这依赖较特殊的线性条件；Stat(8)则通过计算回归斜率，本质上等价在估计$\tilde{\rho}$时考虑了$F_t y_t$的低估问题。总的来说，Stat(8)以牺牲稍复杂计算换来对偏误更“直观”的度量。

**实现准确性：**  代码对Stat(8)的计算遵照公式进行，但需要谨慎验证各部分含义。例如`cov(ytph, yt)`返回$2\times2$矩阵，其中(1,2)元素就是$\operatorname{Cov}(y_{t+1}, y_t)$；`var(Ft_y_t)`返回$F_t y_t$的样本方差，相当于协方差矩阵对角元。代码正确索引并组合了这些值。一个值得注意的小问题是：代码将`cov(Ft_y_tph, Ft_y_t)`计为`covA(1,2)`，`cov(ytph, yt)`计为`covB(1,2)`，而用了单独的`var(yt)`获取$\operatorname{Var}(y_t)$。由于`cov(yt,yt)`的(1,1)元素等于$\operatorname{Var}(y_t)$，其实可以完全用`cov`统一实现（如Stat(7)那样），从而保持一致性。不过这不影响结果，因分母分子皆为样本计算。

Bootstrap检验与Power分析
===

上述Monte Carlo循环完成后，代码最后一部分使用**自举法(Bootstrap)**对统计量(7)和(8)进行区间估计，并计算了检验功效（power）。核心代码如下：

```
matlabCopyEditB = 1000;       % Bootstrap次数
alpha = 0.05;   % 显著性水平

boot_stat_eq7 = zeros(B,1);
boot_stat_eq8 = zeros(B,1);
for b = 1:B
    idx = randi(N, N, 1); % 有放回采样
    boot_stat_eq7(b) = mean(stat_eq7(idx));
    boot_stat_eq8(b) = mean(stat_eq8(idx));
end
boot_stat_eq7 = sort(boot_stat_eq7);
boot_stat_eq8 = sort(boot_stat_eq8);

ci7_low  = boot_stat_eq7(floor(B * alpha/2));
ci7_high = boot_stat_eq7(ceil(B * (1 - alpha/2)));
ci8_low  = boot_stat_eq8(floor(B * alpha/2));
ci8_high = boot_stat_eq8(ceil(B * (1 - alpha/2)));

reject_eq7 = (stat_eq7 < ci7_low) | (stat_eq7 > ci7_high);
reject_eq8 = (stat_eq8 < ci8_low) | (stat_eq8 > ci8_high);
power_eq7 = mean(reject_eq7);
power_eq8 = mean(reject_eq8);
```

**Bootstrap思路：** 模拟得到的500个`stat_eq7`值（每次模拟一个）可以看作统计量在该设定下的抽样分布样本。Bootstrap的做法是**有放回地重采样**这500个值，构造1000组新的样本并计算每组的均值，以近似统计量均值的抽样分布：

- `idx = randi(N, N, 1)` 生成长度为N的随机索引向量，相当于从500个模拟结果中有放回抽取500个，构成一个新的“bootstrap样本”。
- `mean(stat_eq7(idx))` 则是对这一bootstrap样本的均值计算。重复B=1000次后，`boot_stat_eq7`集合就近似服从“统计量均值估计”的分布。
- 排序后，通过`ci7_low`和`ci7_high`取出第$\alpha/2=2.5%$和$(1-\alpha/2)=97.5%$分位数，构造出**95%置信区间**$ci7_low, ci7_high$。这个区间可以理解为：在95%置信水平下，统计量(7)的**总体均值**落在此范围内（这里总体是指$N\to\infty$模拟下的真值，即大样本极限或理论值）。

对于统计量(8)重复相同过程，得到$ci8_low, ci8_high$区间。

**拒绝区间与功效：**  代码将置信区间之外的区域视为“拒绝域”——如果某次模拟的统计量落在此域之外，则被标记为`reject=true`。具体：

- `reject_eq7 = (stat_eq7 < ci7_low) | (stat_eq7 > ci7_high)`：对于每个模拟结果，如果它小于下界或大于上界，则表示**拒绝原假设**。这里原假设隐含为“统计量的均值等于其bootstrap均值”（或无显著偏差）。
- `power_eq7 = mean(reject_eq7)`：计算拒绝比例，即在500次模拟中有多少比例落入拒绝域。这被称为**power（检验功效）**，即当实际存在偏差时，我们拒绝原假设的频率。类似地`power_eq8`是统计量(8)的功效。

需要注意，通常功效的定义是：在给定真偏差存在的情况下，检验能够识别出偏差（拒绝原假设：无偏）的概率。本代码设定$\rho_{\text{hat}}\neq \rho_{\text{true}}$，因此我们实际上是在计算当**存在误识动态**时，两种统计量能多大概率显著偏离原假设值。

**原假设的含义：** 原假设应当是预测者无偏，即$\tilde{\rho}=\rho$，对应统计量理论值为1。然而代码并未直接使用1作为检验基准，而是用了bootstrap区间。这背后的想法可能是假定**在原假设下**统计量分布的均值应为1，但其抽样分布未知，于是使用模拟到的均值代替，并构造区间。由于在我们的模拟中误识是真实存在的（均值不为1），此做法有些不严格：置信区间此时围绕的是偏差情况下的均值约2.2（见下文）。若真要检验$\tilde{\rho}=\rho$，应以1作为检验值。更严谨的方案是在$\rho_{\text{hat}}=\rho_{\text{true}}$的情形下重新模拟以获取统计量的分布，然后计算在偏差场景下统计量超出该分布临界值的比例。事实上，可以通过令`rho_hat = rho_true`再跑一次Monte Carlo得到近似原假设分布。然而本代码采用近似处理，使用偏差情形自身的分布代替。这会导致所谓**功效**的计算其实约等于统计量取值落入自身均值95%区间外的比例，理论上应接近5%（因为95%置信区间覆盖了95%的可能取值），但结果略高于5%是由于抽样误差和分布非对称等原因。也可以把这个数值理解为**实际偏误情况下检验的拒真率**（理想状况下应接近1，如果偏误足够显著的话）。

**计算结果：** 代码通过`fprintf`输出了各项结果。根据模拟，我们可以总结典型输出（取一组代表性数字）：

- 公式(7)的均值统计量约为**2.20**，公式(8)的均值统计量约为**2.20**（和7非常接近）。这明显高于1，表明预测者**高估**了持久性（主观0.8 vs 实际0.5，真比值应约1.6，模拟值略高于理论可能因采样误差）。
- 公式(7)的Bootstrap拒绝区间大致为**[2.16, 2.26]**，公式(8)的拒绝区间约为**[2.16, 2.26]**。两个区间非常接近且都**不包含1**，说明即使考虑采样不确定性，统计量显著大于1。
- 公式(7)的power（拒绝率）约为**0.92**，公式(8)的power约为**0.92**。意味着在当前参数设定下，如果使用这两种统计量来检验“$\tilde{\rho}=\rho$”这个假设，大约有92%的模拟样本会被判定拒绝（即发现了偏误存在）。这是相当高的检验功效，表示样本长度80、预测者100人的信息量已经足够大，使得0.5 vs 0.8的差异能够被可靠地区分。

**比较两种统计量的表现：** 在本次模拟参数下，Stat(7)和Stat(8)的均值、区间和功效几乎相同。这表明两种方法都有效识别出了误感知动态，且未表现出显著差异。不过，这可能与偏差程度较大、样本相对足够有关。当偏差程度较小或样本更有限时，Stat(8)是否能体现出更高的敏感度值得探讨。从理论分析可知，Stat(8)在原理上直接捕捉主观参数，预期在细微偏差时也能有响应；而Stat(7)因为是交叉协方差的比值，可能在小偏差时更接近1，不易检测出显著性。但是Stat(7)结构简单，受噪声同步缩放影响，有时两协方差变化会部分抵消偏误影响，使其偏离1的程度可能**略小**于Stat(8)。因此，可以预计在一些情境下Stat(8)的检验功效会略胜于Stat(7)。

**方法适用性：**  统计量(7)和(8)皆为检测**认知偏误**的有力工具。例如在宏观经济学中，可将$y_t$理解为经济变量，$F_t y_{t+1}$理解为预测者在$t$期对未来的预测，那么这两个统计量可以利用历史数据和预测数据来判断预测者是否系统性高估/低估了动态持久性。如果计算结果显著大于1，可断定预测者存在**过度外推**行为（认为冲击影响更持久）；若显著小于1，则表明预测者**欠反应**（认为冲击更快消退）。由于Stat(8)直接涉及对两种持久性的估计，它甚至可以量化预测者主观参数$\tilde{\rho}$的大小（通过$\tilde{\rho} \approx \text{Stat(8)} \times \rho$）。这在实际研究中相当有价值。

**改进与拓展：**

- 如前所述，为了更严谨地评估检验的显著性水平和功效，建议分两步进行：先在$\tilde{\rho}=\rho$情形下模拟计算统计量的5%和95%分位值（作为临界值）；再在$\tilde{\rho}\neq\rho$情形下计算统计量落在临界值之外的比例（功效）。目前代码将两步合并，可能低估了实际功效。不过在本模拟下偏误很大，以致1远在拒绝区间之外，功效接近1已几乎无须精细计算。
- Bootstrap过程在这里是基于模拟结果的二次抽样。在实际应用中，我们通常只有一组现实数据而无法重复模拟。那时可以对单一数据集进行时间序列区块自举（block bootstrap）或者对回归残差进行自举，以近似统计量的分布。
- 功效分析还能扩展为**敏感性分析**：改变$T$、$M$、$\omega$、偏差幅度等，比较Stat(7)与Stat(8)哪种统计量在更困难的情形下依然保持较高的检测能力。例如，若$T$较小、信息噪声大，预计Stat(8)可能由于同时利用更多矩信息而略胜一筹。
- 最后，从代码组织角度，可以将计算统计量和Bootstrap的部分封装成函数，便于对不同参数运行多次。这有助于提高复用性和清晰度。例如，写一个函数`compute_stats(y, pred)`返回Stat(7)和Stat(8)，然后主程序调用并积累结果。

小结
===

**总结改进建议：** 模拟代码在结构和逻辑上基本正确，但可以通过向量化提高效率，通过完整Kalman滤波框架增强合理性，通过更严谨的bootstrap方案提高检验刻画的准确性。另外，考虑更广泛的参数情形（不同程度的$\rho$偏差、不同噪声水平等）将有助于全面比较统计量(7)与(8)的优劣。

两个公式的比较
===

首先，两者都表现优秀

1. **均值接近理论值**：两种统计量的均值都约为 2.24，稍高于理论值 $1 / 0.5 = 2$，但差距在 10% 内，可接受。这说明两种方法都**能成功识别出预测者系统性高估 $\rho$ 的偏差**。
2. **power都很高**：power 高达 93.6%–93.8%，意味着在 $\rho_{\text{hat}} \neq \rho$ 的设定下，两者都能非常可靠地拒绝“无偏”的原假设。
3. **Bootstrap 区间不含 1**：都远离 $\rho_{\text{hat}}/\rho = 1$，说明偏误显著，可以检出。

Stat（7）的优点：
---

### **结构更简单，计算更高效**

**Stat (7) 只用两个协方差项的比值**，无需涉及多个方差归一化：
$$
\text{Stat (7)} = \frac{\mathrm{Cov}(F_{t+h}, y_t)}{\mathrm{Cov}(y_{t+h}, F_t y_t)}
$$
而 Stat (8) 结构更复杂，需要同时计算四个量（两个协方差 + 两个方差），略显繁琐：
$$
\text{Stat (8)} = \frac{\mathrm{Cov}(F_{t+h}, F_t y_t)}{\mathrm{Var}(F_t y_t)} \cdot \frac{\mathrm{Var}(y_t)}{\mathrm{Cov}(y_{t+h}, y_t)}
$$
 因此 **Stat (7) 更适合大样本环境下的高效计算**，特别适合在嵌套估计、Monte Carlo 或 DSGE 模型中多次调用。

------

### Self-Adjoint的直觉

我认为，Stat (7) 基于一种非常舒适的**“预测者对现在的理解”和“现在对未来的预测”之间的时间对称结构**：
$$
\text{Cov}(F_{t+h}, y_t) \quad \text{vs.} \quad \text{Cov}(y_{t+h}, F_t y_t)
$$
在完全理性的情况下，我们会期望两者具有同样的“映射强度”，使得比值为 1。因此，**这个比值本身就具有强烈的判别力**。

- 若 $\tilde{\rho} > \rho$，预测过度依赖过去，前者大于后者，比值大于 1；
- 若 $\tilde{\rho} < \rho$，预测低估未来惯性，比值小于 1。

这能够给予本文**哲学基础、对称性和经济学解释上的优雅性**。

------

### 在 Bootstrap 下表现更稳定

观察 Bootstrap 区间：

- Stat (7)：[2.190, 2.299] → 区间宽度 0.109
- Stat (8)：[2.192, 2.303] → 区间宽度 0.111

虽然两者非常接近，但 Stat (7) 的波动略小，说明在本设定下 **Stat (7) 在有限样本下的估计更稳定**。这是一个重要优势，尤其在**样本较小、信息较噪**时更为重要。

------

### 没有引入“预测者内部一致性”假设（add）

Stat (8) 中使用 $\mathrm{Cov}(F_{t+h}, F_t y_t)$ 隐含地要求预测结构自身要稳定（如所有人都用 $\rho_{\text{hat}}$ 且预测误差较小），而 Stat (7) 不涉及这种内部预测结构的协方差，因此在 **面对异质预测者时，Stat (7) 更通用，更鲁棒**。

28/05 三个问题
===

离散时间 Kalman 滤波的稳态解：Ricatti 方程
---