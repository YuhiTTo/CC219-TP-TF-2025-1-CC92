# 🏨 Analizador de Comentarios Hoteleros con BERT

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

Una aplicación web moderna y optimizada para análisis de sentimientos en comentarios hoteleros utilizando BERT (BETO) en español, con soporte completo para GPU NVIDIA y procesamiento por lotes eficiente.

## ✨ Características Principales

- 🤖 **Modelo BERT en Español**: Utiliza BETO (BERT para español) para análisis de sentimientos
- 🚀 **Optimización GPU**: Aprovecha completamente las GPUs NVIDIA con CUDA
- 📊 **Procesamiento por Lotes**: Análisis eficiente de múltiples comentarios
- 🎯 **Interpretabilidad**: Explicaciones con SHAP y análisis de palabras clave
- 📈 **Dashboard Avanzado**: Visualizaciones interactivas y métricas detalladas
- 💾 **Exportación de Resultados**: Descarga de análisis en formato CSV
- 🔧 **Monitoreo de GPU**: Seguimiento del rendimiento y uso de memoria

## 🚀 Instalación Rápida

### Prerrequisitos

- Python 3.8 o superior
- GPU NVIDIA compatible con CUDA 12.1+ (opcional pero recomendado)
- 8GB RAM mínimo (16GB recomendado)

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/analizador-comentarios-hoteleros.git
cd analizador-comentarios-hoteleros
```

### 2. Instalar Dependencias

**Para GPU (Recomendado):**
```bash
pip install -r requirements_gpu.txt
```

**Para CPU:**
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación

```bash
streamlit run app_simple.py
```

La aplicación estará disponible en: `http://localhost:8501`

## 🎮 Uso de la Aplicación

### 📝 Análisis Individual
- Ingresa un comentario de hotel en el área de texto
- Obtén la predicción de estrellas (1-5) instantáneamente
- Visualiza las palabras clave positivas y negativas
- Explora la interpretabilidad con SHAP (si está disponible)

### 📁 Análisis por Lotes
- Sube un archivo CSV con comentarios
- Configura el tamaño de lote para optimizar rendimiento
- Monitorea el progreso en tiempo real
- Descarga los resultados procesados

### 📊 Dashboard de Resultados
- Visualiza distribución de sentimientos
- Analiza métricas estadísticas
- Explora nubes de palabras por categoría
- Recibe recomendaciones automáticas

### ⚡ Monitoreo de GPU
- Verifica el estado de CUDA y GPU
- Monitorea uso de memoria en tiempo real
- Ejecuta benchmarks de velocidad
- Optimiza configuraciones automáticamente

## 📁 Estructura del Proyecto

```
analizador-comentarios-hoteleros/
├── app_simple.py                 # Aplicación principal optimizada
├── modelo_beto_estrellas/        # Modelo BERT entrenado
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.txt
├── tripadvisor_hotel_reviews.csv # Dataset de ejemplo
├── comentarios_ejemplo.csv      # Ejemplos para pruebas
├── requirements.txt             # Dependencias básicas
├── requirements_gpu.txt         # Dependencias con soporte GPU
├── run_app_gpu_optimized.bat   # Script de ejecución optimizado
└── README.md                   # Este archivo
```

## 🛠️ Configuración Avanzada

### Optimización de GPU

La aplicación detecta automáticamente tu GPU y optimiza las configuraciones. Para configuración manual:

```python
# En app_simple.py, línea ~90
self.batch_size = 32  # Ajustar según memoria GPU
self.max_len = 160    # Longitud máxima de secuencia
```

### Procesamiento por Lotes

Para datasets grandes, ajusta el tamaño de lote:

- **GPU alta gama (RTX 4080+)**: batch_size = 64-128
- **GPU media (RTX 3060-4070)**: batch_size = 32-64  
- **GPU básica (GTX 1660+)**: batch_size = 16-32
- **CPU**: batch_size = 8-16

## 📊 Formato de Datos

### Archivo CSV de Entrada

El archivo debe contener una columna con comentarios:

```csv
comentario
"El hotel es excelente, muy recomendable"
"Servicio regular, podría mejorar"
"Fantástica experiencia, volveré pronto"
```

### Archivo CSV de Salida

```csv
comentario,prediccion,estrellas,confianza,palabras_positivas,palabras_negativas
"El hotel es excelente...",4.2,4,0.85,"excelente,recomendable","ninguna"
```

## 🔧 Solución de Problemas

### Error: CUDA no disponible
```bash
# Verificar instalación de CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstalar PyTorch con CUDA
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Error: Modelo no encontrado
```bash
# Descomprimir modelo si es necesario
# El modelo debe estar en la carpeta modelo_beto_estrellas/
```

### Error: Memoria GPU insuficiente
- Reduce el batch_size en la configuración
- Usa mixed precision (FP16)
- Limpia el cache de GPU desde la interfaz

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👥 Autores

- **Tu Nombre** - Desarrollo principal - [@tu-usuario](https://github.com/tu-usuario)

## 🙏 Agradecimientos

- Modelo BETO por el equipo de [dccuchile](https://github.com/dccuchile/beto)
- Dataset de TripAdvisor para entrenamiento
- Comunidad de Streamlit por la excelente documentación

## 📈 Roadmap

- [ ] Soporte para modelos multilingües
- [ ] API REST para integración
- [ ] Análisis de aspectos específicos
- [ ] Clasificación de temas automática
- [ ] Soporte para análisis en tiempo real