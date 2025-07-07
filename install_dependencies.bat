@echo off
echo ========================================
echo Instalador de Dependencias Avanzadas
echo Analizador de Comentarios Hoteleros v2.0
echo ========================================

echo.
echo Instalando dependencias esenciales...
pip install streamlit torch transformers pandas numpy plotly

echo.
echo Instalando dependencias para análisis avanzado...
pip install shap wordcloud matplotlib seaborn scikit-learn scipy

echo.
echo Verificando instalación...
python -c "import streamlit, torch, transformers, pandas, numpy, plotly; print('✅ Dependencias esenciales instaladas correctamente')"

echo.
echo Verificando dependencias opcionales...
python -c "
try:
    import shap, wordcloud, matplotlib, seaborn, sklearn, scipy
    print('✅ Todas las dependencias avanzadas instaladas correctamente')
except ImportError as e:
    print(f'⚠️ Algunas dependencias opcionales no están disponibles: {e}')
    print('La aplicación funcionará con funcionalidades básicas')
"

echo.
echo ========================================
echo Instalación completada!
echo Ejecuta: streamlit run app_simple.py
echo ========================================
pause
