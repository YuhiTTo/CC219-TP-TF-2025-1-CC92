@echo off
echo ========================================
echo Analizador de Comentarios Hoteleros v2.0
echo Sistema Avanzado con IA y Dashboard
echo ========================================

echo.
echo 🔍 Verificando sistema...
python test_app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error en verificación del sistema
    echo 💡 Ejecuta install_dependencies.bat para instalar dependencias
    pause
    exit /b 1
)

echo.
echo 🚀 Iniciando aplicación avanzada...
echo.
echo 📋 Características disponibles:
echo   📝 Análisis individual con SHAP
echo   📊 Procesamiento por lotes
echo   📈 Dashboard interactivo  
echo   🔍 Comparación de comentarios
echo   ☁️ Nubes de palabras
echo   📥 Exportación de resultados
echo.
echo 🌐 La aplicación se abrirá en tu navegador
echo ⏹️ Para detener: Ctrl+C en esta ventana
echo.

streamlit run app_simple.py

echo.
echo 👋 Aplicación cerrada
pause
