# EcoVision AI ♻️
### Waste Classification Using Deep Learning and Computer Vision

**EcoVision AI** is a computer vision project focused on the automatic identification of solid waste from images. A multiclass image classification system was developed and evaluated to distinguish six types of waste materials. The project compares a convolutional neural network (CNN) trained from scratch with a pretrained architecture adapted through transfer learning.

The project covers the full Deep Learning development workflow: image exploration and preprocessing, model training, architecture comparison, quantitative evaluation, visual interpretation of predictions, and deployment of an interactive application.

**Live application:** [ecovision-david.streamlit.app](https://ecovision-david.streamlit.app)  
**Developed by:** David Santiago Cifuentes Grimaldo and Juan David Jiménez  
**Academic context:** Final project — Deep Learning for Computer Vision

---

## Problem Statement and Dataset

Visual waste classification is challenging due to differences in shape, texture, color, and lighting, as well as similarities between certain materials. EcoVision AI addresses this problem through an image classifier covering six categories:

**Cardboard · Glass · Metal · Paper · Plastic · General waste**

The dataset contains **2,524 images**, divided into three independent, stratified subsets:

| Dataset split | Images | Proportion |
|:--|--:|--:|
| Training | 1,766 | 70% |
| Validation | 379 | 15% |
| Testing | 379 | 15% |
| **Total** | **2,524** | **100%** |

Data preparation included reviewing images and labels, detecting exact duplicates, resizing, and normalization. Data augmentation was applied during training to increase image variability and support model generalization.

## Model Development

Two image classification approaches were implemented using **PyTorch and Torchvision**:

**Custom CNN.** A convolutional neural network was developed with feature extraction blocks, batch normalization, activation functions, and regularization mechanisms. This architecture served as a baseline trained from scratch for the classification task.

**ResNet18 with transfer learning.** A ResNet18 architecture initialized with pretrained ImageNet weights was adapted to the project. Its learned feature extraction capabilities were reused, and the classification head was modified to predict the six target classes.

Both architectures were trained and evaluated under a common experimental methodology. Validation metrics were monitored throughout training, and the best-performing model was retained based on **macro F1-score**, enabling a comparison of performance across classes beyond overall accuracy.

## Results and Evaluation

Final evaluation was conducted on **379 test images** that were kept separate from the data used for training and model selection.

| Test metric | Custom CNN | **ResNet18** |
|:--|--:|--:|
| Accuracy | 56.20% | **78.63%** |
| Macro precision | 57.51% | **76.91%** |
| Macro recall | 58.59% | **75.78%** |
| Macro F1-score | 53.63% | **76.16%** |

**ResNet18 achieved the strongest overall performance**, improving macro F1-score by **22.53 percentage points** compared with the custom CNN. It correctly classified **298 out of 379 test images**.

The evaluation included learning curves, confusion matrices, class-level metrics, and an analysis of incorrect predictions. **Cardboard and paper** exhibited comparatively strong performance, while **general waste** proved more challenging. Confusion was also observed between visually similar materials, particularly **glass and plastic**.

To support model interpretability, **Grad-CAM** was used to visualize image regions that influenced selected predictions. This qualitative analysis complemented the quantitative results and provided additional insight into the model's behavior.

## Interactive Application

The selected ResNet18 model was integrated into an application developed with **Streamlit**. The interface allows users to upload waste images, view the predicted category, and examine the model's score distribution across all six classes.

The application connects the experimental training process with a practical use case, making model predictions accessible through a web interface.

**[Explore the EcoVision AI live demo →](https://ecovision-david.streamlit.app)**

## Technology Stack

**Programming and analysis:** Python, NumPy, Pandas  
**Deep Learning:** PyTorch, Torchvision, convolutional neural networks, ResNet18, transfer learning  
**Evaluation and visualization:** Scikit-learn, Matplotlib, Seaborn, confusion matrix, Grad-CAM  
**Application and deployment:** Streamlit, GitHub

## Scope and Limitations

EcoVision AI is an **academic prototype** for visual waste classification. The reported metrics were measured on the project's held-out test set and do not guarantee equivalent performance in all real-world settings.

Predictions may be affected by images containing multiple objects, mixed waste, complex backgrounds, different lighting conditions, or materials that are underrepresented in the dataset. In addition, the confidence scores shown in the application are **not calibrated probabilities**.

The project illustrates the potential of transfer learning for image classification using a moderately sized dataset and highlights the importance of complementing accuracy with class-level metrics, error analysis, and interpretability.

---

**Technical resources:** [Training and evaluation notebook](EcoVision_Proyecto_Final.ipynb) · [Project presentation](EcoVision_Presentacion.pdf) · [Live application](https://ecovision-david.streamlit.app)

**Language:** [Versión en español](README.md)
