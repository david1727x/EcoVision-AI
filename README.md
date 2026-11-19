# EcoVision AI ♻️
### Clasificación de residuos mediante Deep Learning y visión por computador

**EcoVision AI** es un proyecto de visión por computador orientado a la identificación automática de residuos sólidos a partir de imágenes. Se desarrolló y evaluó un sistema de clasificación multiclase que distingue seis tipos de materiales, comparando una red neuronal convolucional construida desde cero con una arquitectura preentrenada mediante transferencia de aprendizaje.

El proyecto integra el ciclo completo de desarrollo de un modelo de Deep Learning: exploración y preparación de imágenes, entrenamiento, comparación de arquitecturas, evaluación cuantitativa, interpretación visual de predicciones y despliegue de una aplicación interactiva.

**Aplicación web:** [ecovision-david.streamlit.app](https://ecovision-david.streamlit.app)  
**Desarrollado por:** David Santiago Cifuentes Grimaldo y Juan David Jiménez  
**Contexto:** Proyecto final de Deep Learning para Visión

---

## Problema y conjunto de datos

La clasificación visual de residuos presenta dificultades relacionadas con las diferencias de forma, textura, color e iluminación, así como con la similitud entre algunos materiales. EcoVision AI aborda este problema mediante un clasificador de imágenes de seis categorías:

**Cartón · Vidrio · Metal · Papel · Plástico · Basura general**

Se trabajó con un conjunto de **2.524 imágenes**, distribuidas de forma estratificada en tres subconjuntos independientes:

| Partición | Imágenes | Proporción |
|:--|--:|--:|
| Entrenamiento | 1.766 | 70 % |
| Validación | 379 | 15 % |
| Prueba | 379 | 15 % |
| **Total** | **2.524** | **100 %** |

La preparación de los datos incluyó la revisión de las imágenes y sus etiquetas, la detección de duplicados exactos, el redimensionamiento y la normalización. Durante el entrenamiento se aplicaron transformaciones para ampliar la variabilidad de las imágenes y favorecer la generalización.

## Desarrollo de los modelos

Se implementaron dos enfoques de clasificación mediante **PyTorch y Torchvision**:

**CNN propia.** Se construyó una red convolucional con bloques de extracción de características, normalización por lotes, funciones de activación y mecanismos de regularización. Esta arquitectura permitió establecer una línea base entrenada desde cero para el problema.

**ResNet18 con transferencia de aprendizaje.** Se adaptó una red ResNet18 inicializada con pesos preentrenados en ImageNet. Se reutilizó su capacidad de extracción de características y se modificó la capa de clasificación para producir las seis categorías del proyecto.

Ambas arquitecturas se entrenaron y evaluaron con una metodología común. Durante el entrenamiento se supervisaron las métricas de validación y se conservó el mejor modelo según el **F1-score macro**, con el propósito de comparar el rendimiento entre categorías, más allá de la exactitud global.

## Resultados y evaluación

La evaluación final se realizó sobre **379 imágenes de prueba**, separadas de los datos empleados para el entrenamiento y la selección de modelos.

| Métrica en prueba | CNN propia | **ResNet18** |
|:--|--:|--:|
| Exactitud (accuracy) | 56,20 % | **78,63 %** |
| Precisión macro | 57,51 % | **76,91 %** |
| Recall macro | 58,59 % | **75,78 %** |
| F1-score macro | 53,63 % | **76,16 %** |

**ResNet18 presentó el mejor desempeño**, con una mejora de **22,53 puntos porcentuales en F1-score macro** frente a la CNN propia. El modelo clasificó correctamente **298 de las 379 imágenes de prueba**.

El análisis incluyó curvas de aprendizaje, matriz de confusión, métricas por categoría y revisión de predicciones incorrectas. Las clases **cartón y papel** mostraron un comportamiento comparativamente favorable, mientras que **basura general** presentó mayores dificultades. También se observaron confusiones entre materiales visualmente similares, en particular entre **vidrio y plástico**.

Como componente de interpretabilidad, se empleó **Grad-CAM** para visualizar las regiones de las imágenes que influyeron en determinadas predicciones. Este análisis complementó las métricas cuantitativas y permitió examinar el comportamiento del modelo desde una perspectiva visual.

## Aplicación interactiva

El modelo ResNet18 seleccionado se integró en una aplicación desarrollada con **Streamlit**. La interfaz permite cargar fotografías de residuos, obtener la categoría estimada y visualizar la distribución de puntajes para las seis clases.

La aplicación conecta el entrenamiento experimental con un escenario de uso práctico, facilitando la exploración de resultados mediante una interfaz web.

**[Ver demostración interactiva de EcoVision AI →](https://ecovision-david.streamlit.app)**

## Tecnologías

**Lenguaje y análisis:** Python, NumPy, Pandas  
**Deep Learning:** PyTorch, Torchvision, redes convolucionales, ResNet18, transferencia de aprendizaje  
**Evaluación y visualización:** Scikit-learn, Matplotlib, Seaborn, matriz de confusión, Grad-CAM  
**Aplicación y despliegue:** Streamlit, GitHub

## Alcance y limitaciones

EcoVision AI constituye un **prototipo académico** de clasificación visual de residuos. Los resultados reportados corresponden al conjunto de prueba utilizado en el proyecto y no implican un rendimiento equivalente en cualquier entorno.

Las predicciones pueden verse afectadas por fotografías con varios objetos, residuos mezclados, fondos complejos, condiciones de iluminación diferentes o materiales poco representados en los datos. Asimismo, los puntajes de confianza mostrados por la aplicación no representan probabilidades calibradas.

El proyecto demuestra el potencial de la transferencia de aprendizaje para abordar tareas de clasificación de imágenes con conjuntos de datos de tamaño moderado, así como la importancia de complementar la exactitud con métricas por clase, análisis de errores e interpretabilidad.

---

**Documentación técnica:** [Notebook de entrenamiento y evaluación](EcoVision_Proyecto_Final.ipynb) · [Presentación del proyecto](EcoVision_Presentacion.pdf) · [Aplicación web](https://ecovision-david.streamlit.app)
