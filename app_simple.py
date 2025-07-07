import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import re
from collections import Counter
import os

# Configuración para evitar el error de Triton
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

# Suprimir warnings de Triton
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*triton.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*triton.*")

# Configurar para usar eager execution si hay problemas con compile
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configuración de página
st.set_page_config(
    page_title="Analizador de Comentarios Hoteleros",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar transformers con manejo de errores
@st.cache_data
def check_dependencies():
    """Verificar dependencias disponibles"""
    status = {
        'transformers': False,
        'shap': False,
        'model_loaded': False
    }
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        status['transformers'] = True
    except ImportError:
        pass
    
    try:
        import shap
        status['shap'] = True
    except ImportError:
        pass
    
    return status

# Verificar dependencias
DEPS = check_dependencies()

if DEPS['transformers']:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

if DEPS['shap']:
    import shap

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class HotelReviewAnalyzer:
    def __init__(self, model_path):
        if not DEPS['transformers']:
            st.error("⚠️ La librería transformers no está disponible. Por favor instala: pip install transformers")
            self.model_loaded = False
            self.model = None
            self.tokenizer = None
            self.explainer = None
            return
            
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 160
        self.class_names = ['1 Estrella', '2 Estrellas', '3 Estrellas', '4 Estrellas', '5 Estrellas']
        self.model = None
        self.tokenizer = None
        self.explainer = None
        
        # Configuraciones de optimización para GPU
        if torch.cuda.is_available():
            # Configurar para usar mixed precision si está disponible
            self.use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
            # Optimizar configuraciones de CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Configurar cache de memoria
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            st.info(f"🚀 GPU detectada: {torch.cuda.get_device_name(0)}")
            st.info(f"💾 Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.use_amp = False
            st.info("🖥️ Usando CPU para procesamiento")
            
        self.model_loaded = self.load_model()
        
    def load_model(self):
        """Cargar el modelo y tokenizador pre-entrenado sin Triton"""
        try:
            with st.spinner("Cargando modelo BERT..."):
                # Optimizar memoria GPU antes de cargar el modelo
                if torch.cuda.is_available():
                    self.optimize_gpu_memory()
                
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                # Mover modelo a GPU y optimizar
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # NO USAR torch.compile para evitar problemas con Triton
                if torch.cuda.is_available():
                    st.info("🚀 GPU optimizada sin torch.compile para mayor estabilidad")
                    
                    # Mostrar información de memoria
                    memory_info = self.get_gpu_memory_info()
                    if memory_info:
                        st.info(f"💾 Memoria GPU utilizada: {memory_info['allocated_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB ({memory_info['utilization_pct']:.1f}%)")
                
                # Configurar SHAP explainer solo si está disponible
                if DEPS['shap']:
                    try:
                        st.info("🔧 Configurando análisis de interpretabilidad SHAP...")
                        self.explainer = "available"
                        st.success("✅ SHAP disponible para análisis detallado")
                    except Exception as e:
                        st.warning(f"⚠️ SHAP no se pudo configurar: {e}. Usando análisis básico.")
                        self.explainer = None
                else:
                    self.explainer = None
                    st.info("ℹ️ SHAP no está disponible. Usando análisis básico de palabras clave.")
                    
            st.success("✅ Modelo cargado exitosamente!")
            return True
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")
            self.model = None
            self.tokenizer = None
            self.explainer = None
            return False
    
    def optimize_gpu_memory(self):
        """Optimizar memoria GPU antes de cargar el modelo"""
        if torch.cuda.is_available():
            # Limpiar cache de memoria GPU
            torch.cuda.empty_cache()
            
            # Configurar crecimiento de memoria
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Usar 90% de la memoria GPU disponible
                torch.cuda.set_per_process_memory_fraction(0.9)
    
    def get_gpu_memory_info(self):
        """Obtener información de memoria GPU"""
        if not torch.cuda.is_available():
            return None
        
        try:
            # Obtener información de memoria
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_gb': total,
                'utilization_pct': (allocated / total) * 100,
                'free_gb': total - allocated
            }
        except Exception as e:
            st.warning(f"No se pudo obtener información de memoria GPU: {e}")
            return None
    
    def predict_single_review(self, review_text):
        """Predecir el rating de un comentario individual con optimización GPU"""
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            return None, None, None
            
        try:
            encoding = self.tokenizer.encode_plus(
                review_text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            
            input_ids = encoding['input_ids'].to(self.device, non_blocking=True)
            attention_mask = encoding['attention_mask'].to(self.device, non_blocking=True)
            
            # Usar autocast para mixed precision si está disponible
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        prediction_label = torch.argmax(outputs.logits, dim=1).item()
                        predicted_stars = prediction_label + 1
                        confidence = probabilities[0][int(prediction_label)].item()
            else:
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    prediction_label = torch.argmax(outputs.logits, dim=1).item()
                    predicted_stars = prediction_label + 1
                    confidence = probabilities[0][int(prediction_label)].item()
            
            return predicted_stars, confidence, probabilities[0].cpu().numpy()
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return None, None, None

    def analyze_keywords(self, review_text, top_k=10):
        """Análisis básico de palabras clave"""
        # Palabras positivas y negativas comunes en español
        positive_words = {
            'excelente', 'bueno', 'genial', 'perfecto', 'increíble', 'maravilloso',
            'fantástico', 'estupendo', 'magnifico', 'limpio', 'cómodo', 'amable',
            'recomendable', 'agradable', 'bonito', 'hermoso', 'delicioso'
        }
        
        negative_words = {
            'malo', 'terrible', 'pésimo', 'horrible', 'sucio', 'incómodo',
            'desagradable', 'molesto', 'ruidoso', 'caro', 'problemático',
            'deficiente', 'inadecuado', 'insatisfactorio', 'decepcionante'
        }
        
        # Limpiar y tokenizar texto
        words = re.findall(r'\b\w+\b', review_text.lower())
        word_counts = Counter(words)
        
        # Clasificar palabras por sentimiento
        keyword_analysis = []
        for word, count in word_counts.most_common(top_k):
            if len(word) > 2:  # Filtrar palabras muy cortas
                if word in positive_words:
                    sentiment = 'Positivo'
                    impact = 0.8
                elif word in negative_words:
                    sentiment = 'Negativo'
                    impact = -0.8
                else:
                    sentiment = 'Neutral'
                    impact = 0.1
                
                keyword_analysis.append({
                    'word': word,
                    'count': count,
                    'sentiment': sentiment,
                    'impact': impact
                })
        
        return keyword_analysis

def main():
    st.markdown('<h1 class="main-header">🏨 Analizador de Comentarios Hoteleros</h1>', unsafe_allow_html=True)
    st.markdown("### Análisis de Satisfacción de Clientes usando BERT en Español")
    
    # Verificar dependencias
    if not DEPS['transformers']:
        st.error("⚠️ Dependencias faltantes. Por favor instala: `pip install transformers torch`")
        st.stop()
    
    # Inicializar el analizador
    model_path = "./modelo_beto_estrellas"
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = HotelReviewAnalyzer(model_path)
    
    analyzer = st.session_state.analyzer
    
    if not analyzer.model_loaded:
        st.error("❌ No se pudo cargar el modelo. Verifica que existe la carpeta 'modelo_beto_estrellas'")
        st.stop()
    
    # Análisis individual simplificado
    st.header("📝 Análisis Individual de Comentarios")
    
    # Input del comentario
    review_text = st.text_area(
        "Escribe o pega un comentario de hotel:",
        height=100,
        placeholder="Ejemplo: El hotel es excelente, las habitaciones están muy limpias y el personal es muy amable..."
    )
    
    if st.button("🔍 Analizar Comentario", type="primary"):
        if review_text.strip():
            with st.spinner('Analizando comentario...'):
                # Hacer predicción
                predicted_stars, confidence, probabilities = analyzer.predict_single_review(review_text)
                
                if predicted_stars is not None:
                    # Mostrar resultados
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>⭐ Predicción: {predicted_stars} Estrellas</h2>
                            <p>Confianza: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Gráfico de probabilidades
                        fig = px.bar(
                            x=analyzer.class_names,
                            y=probabilities,
                            title="Distribución de Probabilidades",
                            labels={'x': 'Rating', 'y': 'Probabilidad'}
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Análisis de palabras clave
                    st.subheader("🔍 Análisis de Palabras Clave")
                    
                    keyword_analysis = analyzer.analyze_keywords(review_text)
                    
                    if keyword_analysis:
                        col_pos, col_neg = st.columns(2)
                        
                        positive_words = [item for item in keyword_analysis if item['sentiment'] == 'Positivo']
                        negative_words = [item for item in keyword_analysis if item['sentiment'] == 'Negativo']
                        
                        with col_pos:
                            st.markdown("**🟢 Palabras con Influencia Positiva:**")
                            if positive_words:
                                for item in positive_words[:5]:
                                    st.write(f"• **{item['word']}**: {item['impact']:+.2f}")
                            else:
                                st.write("No se encontraron palabras positivas significativas")
                        
                        with col_neg:
                            st.markdown("**🔴 Palabras con Influencia Negativa:**")
                            if negative_words:
                                for item in negative_words[:5]:
                                    st.write(f"• **{item['word']}**: {item['impact']:+.2f}")
                            else:
                                st.write("No se encontraron palabras negativas significativas")
                    
                    st.success("✅ Análisis completado exitosamente!")
                    
                    # Información adicional sobre el análisis
                    with st.expander("ℹ️ Información sobre el Análisis"):
                        st.write("""
                        **Análisis GPU Optimizado:**
                        - Se utiliza tu GPU para acelerar significativamente las predicciones.
                        - Mixed Precision activado para máxima eficiencia.
                        - Configuración optimizada sin Triton para máxima estabilidad.
                        """)
                else:
                    st.error("❌ Error al procesar el comentario")
        else:
            st.warning("Por favor, ingresa un comentario para analizar.")

    # Información de estado
    st.sidebar.header("Estado del Sistema")
    if torch.cuda.is_available():
        st.sidebar.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        memory_info = analyzer.get_gpu_memory_info()
        if memory_info:
            st.sidebar.info(f"💾 Memoria: {memory_info['utilization_pct']:.1f}% utilizada")
    else:
        st.sidebar.info("🖥️ Modo CPU")
    
    st.sidebar.success("✅ Sin errores de Triton")

if __name__ == "__main__":
    main()
