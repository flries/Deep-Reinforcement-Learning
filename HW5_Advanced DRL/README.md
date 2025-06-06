## HW5_Advanced DRL: Gradient Surgery for Multi-Task Learning
> **來源網址：** https://paperswithcode.com/paper/gradient-surgery-for-multi-task-learning-1  
> **論文網址：** https://arxiv.org/pdf/2001.06782v4.pdf  
> **程式網址：** https://github.com/WeiChengTseng/Pytorch-PCGrad

這篇論文《Gradient Surgery for Multi-Task Learning》提出了一種名為「PCGrad」（Projected Conflicting Gradients）的技術，旨在解決多任務學習（Multi-Task Learning, MTL）中常見的梯度衝突問題。以下是對該論文的簡要解釋：

* * *

### 🔍 問題背景：多任務學習中的梯度衝突

在多任務學習中，模型同時學習多個任務，這些任務可能會產生方向不一致的梯度，導致模型在某些任務上性能下降，甚至無法有效收斂。這種梯度之間的干擾被稱為「梯度衝突」。

***

### 🎯 核心問題：多任務學習的梯度衝突

在多任務學習中，不同任務的梯度可能方向不一致（甚至相反），這種**梯度衝突**會妨礙模型的學習。作者指出這種衝突在以下三種情況同時發生時特別致命（稱為「悲慘三聯性」）：

1.  **衝突的梯度方向**（負的餘弦相似度）
2.  **梯度大小差異大**（某些任務主導）
3.  **高曲率的優化地形**（容易誤估效果）
    

這會導致某些任務的性能提升被高估、其他任務的損失被低估，進而阻礙整體學習進展。

* * *

### 🛠 解決方案：PCGrad 技術

PCGrad 的核心思想是通過「梯度手術」來減少任務之間的干擾。具體做法如下：

1.  **檢測衝突**：計算任務之間梯度的餘弦相似度，判斷是否存在衝突（即相似度為負值）。
2.  **投影操作**：對於存在衝突的梯度，將其中一個梯度投影到另一個梯度的法向平面上，從而去除相互干擾的部分。
3.  **更新參數**：使用經過投影處理的梯度來更新模型參數，避免梯度之間的負面影響。
    

這種方法不依賴於特定的模型架構，易於實現，並且可以與其他多任務學習方法結合使用。

* * *

### 📊 實驗結果與應用

作者在多個數據集和任務上驗證了 PCGrad 的有效性，包括：

*   **多任務分類**：在 CIFAR-100 和 MultiMNIST 數據集上，PCGrad 提升了分類準確率。
*   **場景理解**：在 NYUv2 數據集上，結合 PCGrad 的模型在語義分割、深度估計和表面法線預測等任務上表現更佳。
*   **強化學習**：在 Meta-World 基準測試中的多任務強化學習和目標導向強化學習任務中，PCGrad 提高了數據效率和最終性能。
    

這些實驗表明，PCGrad 能夠有效減少梯度衝突，提高多任務學習的整體性能。

🛠 解法：PCGrad（Projected Conflicting Gradients）
---------------------------------------------

PCGrad 的主要思路是——**如果兩個任務的梯度方向衝突，就把其中一個梯度投影到另一個梯度的法向平面上**，從而去除彼此之間的干擾分量。

### 實作步驟簡述：

1.  對每對任務  $i, j$ ，檢查梯度  $g_i, g_j$  是否衝突（內積 < 0）
2.  若衝突，將  $g_i$  投影到與  $g_j$  垂直的平面（即刪去  $g_i$  在  $g_j$  方向上的分量）
3.  對每個任務都重複這個過程，再用修正後的梯度進行更新
    

此方法是**模型無關的（model-agnostic）**，可與現有多任務學習架構結合使用。

* * *

### 📐 理論分析
在凸優化問題中，PCGrad 至少會收斂到局部最小值或梯度完全對立的點
在滿足「悲慘三聯性」的情況下，PCGrad 更新可以比傳統梯度下降得到更低的損失
    
### 🧪 實驗結果

#### **1\. 多任務監督式學習**

*   **數據集**：MultiMNIST、CelebA、CIFAR-100、NYUv2（語意分割、深度估計、表面法線）
*   **結果**：PCGrad 提升了準確率與穩定性，尤其與其他架構（如 MTAN、Cross-Stitch）結合時效果更佳。
    

#### **2\. 多任務強化學習（Multi-task RL）**

*   **平台**：Meta-World（MT10, MT50）
*   **結果**：PCGrad 在樣本效率與成功率上明顯優於 baseline（如 SAC、GradNorm）
    

#### **3\. 消融實驗**

*   只調整梯度方向或大小單獨效果都差，證明**兩者皆需改變**才有效。
    

✅ 總結與貢獻
-------

*   **問題釐清**：定義了多任務學習中的三大優化困境（悲慘三聯性）
*   **方法簡單有效**：PCGrad 是一種簡潔、通用的方法，能有效減少梯度干擾
*   **實證結果佳**：在多種場景中顯著提升表現
*   **可擴展性強**：可應用於其他場景，如元學習、持續學習、語言任務等

---

### 🧠 模型架構與訓練設定

*   **共享特徵萃取器**：MultiLeNetR
*   **左右任務專屬輸出頭**：各自一個 MultiLeNetO
*   **優化器**：Adam（learning rate = 0.0005）
*   **梯度處理方式**：PCGrad 封裝後使用 `.pc_backward()` 處理多任務損失
*   **訓練週期**：100 個 epoch
*   **批次大小**：256

---

### 📉 訓練損失分析

![loss_left](https://github.com/user-attachments/assets/f9579755-bd00-4647-b776-19471bbf1d1c)
#### 左任務（Training Loss - Left Digit）
*   初始 loss 約為 1.75
*   前 20 epoch 急速下降，顯示模型迅速學會特徵
*   最終穩定於約 **0.60**
*   曲線平滑無過擬合現象

![loss_right](https://github.com/user-attachments/assets/cb0f3144-7b74-49fa-8c34-eebf382dd28a)
#### 右任務（Training Loss - Right Digit）
*   趨勢與左任務類似，但中後段震盪稍大
*   收斂速度稍慢，最終 loss 穩定在約 **0.65–0.7**

🔎 **觀察結果**：兩任務皆成功學習，顯示 PCGrad 能有效支持平衡訓練。

---

### 📈 驗證準確率分析

![acc_left](https://github.com/user-attachments/assets/52a81da0-0a03-42fe-b0ac-ace635e464a4)
#### 左任務（Validation Accuracy - Left Digit）
*   準確率穩定上升，最終收斂於 **約 91%**
*   上升趨勢平滑，無大波動

![acc_right](https://github.com/user-attachments/assets/89ae1597-e4bd-4a12-aa70-849d40755316)
#### 右任務（Validation Accuracy - Right Digit）
*   同樣快速提升，最終收斂於 **約 89%**
*   稍低於左任務，推測任務難度或樣本特徵略有差異

---

### 📊 混淆矩陣分析

![confusion_left](https://github.com/user-attachments/assets/1d3f86f7-bd2d-48eb-8b4a-c3dd6e45dedd)
#### 左任務（Confusion Matrix - Left Digit）
*   對角線分布明顯，準確性良好
*   較常誤判：
    *   2 → 9（2 次）  
    *   8 → 1（2 次） 
*   主要錯誤來自相似字型的混淆（如 2 vs 9）

![confusion_right](https://github.com/user-attachments/assets/a5aa5d05-e372-4b27-b592-b316d7e9e380)
#### 右任務（Confusion Matrix - Right Digit）
*   分布同樣集中於對角線
*   較分散誤判情形出現在類別 9
*   整體表現與左任務相當，精度略低
    
---

![frame_100](https://github.com/user-attachments/assets/9651dae1-3136-468b-b799-03f71bfe4dc1)
### 📊 綜合曲線圖（Loss + Accuracy）
*   **Loss 曲線**：左任務（藍）與右任務（橘）皆收斂良好
*   **Accuracy 曲線**：左任務（綠）略高，右任務（紅）稍低，但穩定
*   可清楚觀察 PCGrad 幫助兩任務共同收斂，且無任務明顯被壓制或犧牲
*   
---
### ✅ 總結分析

| 評估面向 | 左數字任務 | 右數字任務 | 結論 |
| --- | --- | --- | --- |
| 訓練損失 | 收斂於 ~0.60 | 收斂於 ~0.70 | 雙方穩定收斂 |
| 驗證準確率 | 收斂於 ~91% | 收斂於 ~89% | 表現良好 |
| 混淆矩陣集中性 | 高（誤差可預期） | 高（誤判較分散） | 無顯著偏差 |
| 收斂速度 | 快速 | 稍慢但穩定 | 表現一致 |
| 任務均衡性 | 無明顯偏袒 | 無明顯偏袒 | PCGrad 有效緩解梯度衝突 |
