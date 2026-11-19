# EcoVision AI — Clasificación inteligente de residuos ♻️

**Proyecto final · Deep Learning para Visión**

**Autores:** David Santiago Cifuentes Grimaldo y Juan David Jiménez.

### 🌐 [Probar la aplicación EcoVision AI en Streamlit](https://ecovision-david.streamlit.app)

## Objetivo
Desarrollar un prototipo de visión por computador que clasifique una fotografía de un único residuo en seis categorías: cartón, vidrio, metal, papel, plástico y basura general. Se comparan una **CNN entrenada desde cero** y **ResNet18 mediante transferencia de aprendizaje**.

## Resultados (prueba independiente, 379 imágenes)

| Métrica | CNN propia | ResNet18 |
| --- | ---: | ---: |
| Accuracy | 56,20 % | **78,63 %** |
| Precisión macro | 57,51 % | **76,91 %** |
| Recall macro | 58,59 % | **75,78 %** |
| F1-score macro | 53,63 % | **76,16 %** |

ResNet18 clasificó correctamente **298 de 379 imágenes de prueba**. Estas métricas corresponden al conjunto de prueba; no garantizan el mismo rendimiento con fotografías externas.

## Archivos

```text
app.py                          Aplicación web Streamlit
requirements.txt                Dependencias
best_model.pth                  Checkpoint entrenado (ya publicado)
EcoVision_Proyecto_Final.ipynb  Experimentos ejecutados (ya publicado)
EcoVision_Presentacion.pdf     Diapositivas de 5 páginas (ya publicado)
```

> **Archivos confirmados en GitHub:** modelo `best_model.pth`, notebook `EcoVision_Proyecto_Final.ipynb`, presentación `EcoVision_Presentacion.pdf`. La aplicación está publicada en [Streamlit Community Cloud](https://ecovision-david.streamlit.app), según la prueba del equipo.

## Uso en Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud seleccionar el repositorio `david1727x/EcoVision-AI`, rama `main`, archivo `app.py`.

La interfaz acepta una fotografía, muestra la clase predicha y el puntaje del modelo para las seis categorías. Las puntuaciones **no están calibradas**, por lo que no deben interpretarse como probabilidades verificadas.

## Limitaciones

Prototipo académico. La identificación puede fallar cuando hay varios objetos, materiales mezclados, cambios fuertes de iluminación o fotografías diferentes a las del dataset. No sustituye sistemas reales de clasificación industrial.

## Entrega académica

La entrega recomendada comprende el notebook ejecutado, la presentación de cinco diapositivas, el repositorio y el enlace a la [aplicación Streamlit](https://ecovision-david.streamlit.app).
