# 🧠 Brain Tumour Detection Using Deep Learning

This project implements a deep convolutional neural network (DCNN) with transfer learning to detect brain tumours from MRI images. With limited data and computational resources, the system achieved 95.2% accuracy, along with perfect AUC and AUPRC scores — comparable to diagnostic performance reported in clinical literature.

---

## 🧬 Motivation

Brain tumours are among the most lethal forms of cancer, with survival rates as low as 5% in some cases. Early and accurate diagnosis is critical. This project explores how AI can assist in medical imaging tasks, potentially reducing diagnostic error and workload for radiologists.

---

## 🔍 Key Features

- ✅ 95.2% accuracy on test data
- 🎯 ROC AUC and AUPRC scores of 1.0
- 🧠 Transfer learning using ResNet-50V2
- ⚙️ Bayesian hyperparameter optimization
- 🧪 K-fold cross-validation with stratified sampling
- 🧩 Data augmentation to address class imbalance

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Frameworks/Libraries:** TensorFlow, Keras, NumPy, scikit-learn  
- **Tools:** Matplotlib, Pandas, OpenCV, Google Colab / Jupyter  
- **Models Used:** ResNet-50V2, VGG16, EfficientNet, InceptionV3 (evaluated)

---

## 🧪 Methodology

### 🖼️ Preprocessing
- Pixel normalization (0–1)
- Brightness equalization across scans
- Cropping to remove background
- Image resizing (224×224)
- 10% hold-out test set

### 📊 Hyperparameter Optimization
- **Search method:** Bayesian search (150 iterations)
- **Evaluated:** Learning rate, batch size, optimiser, dense layer size, dropout
- **Best params:**  
  - Learning rate: `9.56e-5`  
  - Batch size: `32`  
  - Optimiser: `Adam`  
  - Dense neurons: `1024`  
  - Dropout: `0.6775`  
  - Model: `ResNet-50V2`

### 📈 Training Process
- Pretrained models used with frozen base layers
- Features extracted once for efficiency
- Task-specific head trained with early stopping
- Finetuning with lower learning rate to improve performance

---

## 📊 Results

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | 95.23%    |
| Precision    | 1.0       |
| Recall       | 0.929     |
| ROC AUC      | 1.0       |
| AUPRC        | 1.0       |

> Only 1 sample was misclassified in the final test set.  
> Model outperformed classical methods like Random Forests (86% accuracy).

---

## 🧠 Evaluation Insights

- Model generalised well on small dataset (n = 222)
- Effective class balancing via augmentation
- Strong validation through k-fold cross-validation
- Stratified folds reduced class imbalance impact

---

## ⚠️ Limitations

- Small dataset (222 samples, 21 test images)
- Limited diversity in imaging sources
- No fine-tuning of ensemble methods or GAN-based augmentation

---

## 🚀 Future Work

- Validate on larger, multi-center datasets
- Evaluate performance of radiologists vs. AI side-by-side
- Use GANs or more advanced data augmentation techniques
- Investigate ensemble learning approaches

---

## 📄 Citation

If you use or reference this work in your research, please cite the full report or credit the original authors of the referenced papers included in the bibliography.

---

## 📚 References

This project builds upon research from over 30 academic sources. For a full bibliography, refer to the original [report](./report.pdf).

---

## 📝 License

MIT License. See `LICENSE` for details.
