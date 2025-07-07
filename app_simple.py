import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import os
import io
import time
from datetime import datetime

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
        'wordcloud': False,
        'matplotlib': False,
        'seaborn': False,
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
    
    try:
        from wordcloud import WordCloud
        status['wordcloud'] = True
    except ImportError:
        pass
    
    try:
        import matplotlib.pyplot as plt
        status['matplotlib'] = True
    except ImportError:
        pass
    
    try:
        import seaborn as sns
        status['seaborn'] = True
    except ImportError:
        pass
    
    return status

# Verificar dependencias
DEPS = check_dependencies()

if DEPS['transformers']:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

if DEPS['shap']:
    import shap

if DEPS['wordcloud']:
    from wordcloud import WordCloud

if DEPS['matplotlib']:
    import matplotlib.pyplot as plt

if DEPS['seaborn']:
    import seaborn as sns

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
        
        # Import transformers aquí para evitar problemas de scope
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 160
        self.class_names = ['1 Estrella', '2 Estrellas', '3 Estrellas', '4 Estrellas', '5 Estrellas']
        self.model = None
        self.tokenizer = None
        self.explainer = None
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForSequenceClassification = AutoModelForSequenceClassification
        
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
                
                self.model = self.AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_path)
                
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

    def predict_batch(self, reviews_list, batch_size=8):
        """Predecir ratings para múltiples comentarios en lotes"""
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            return [], [], []
        
        predictions = []
        confidences = []
        probabilities_list = []
        
        try:
            for i in range(0, len(reviews_list), batch_size):
                batch = reviews_list[i:i + batch_size]
                batch_predictions = []
                batch_confidences = []
                batch_probabilities = []
                
                for review in batch:
                    pred, conf, probs = self.predict_single_review(review)
                    if pred is not None:
                        batch_predictions.append(pred)
                        batch_confidences.append(conf)
                        batch_probabilities.append(probs)
                    else:
                        batch_predictions.append(None)
                        batch_confidences.append(None)
                        batch_probabilities.append(None)
                
                predictions.extend(batch_predictions)
                confidences.extend(batch_confidences)
                probabilities_list.extend(batch_probabilities)
                
                # Pequeña pausa para evitar sobrecarga de GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    time.sleep(0.1)
        
        except Exception as e:
            st.error(f"Error en predicción por lotes: {e}")
            return [], [], []
        
        return predictions, confidences, probabilities_list

    def analyze_shap_explanation(self, review_text, max_evals=200):
        """Análisis SHAP completo usando PartitionExplainer para BERT"""
        if not DEPS['shap'] or not self.model_loaded:
            return self.analyze_keywords(review_text)
        
        try:
            import shap
            
            # Crear función wrapper optimizada para BERT
            def bert_predict_wrapper(texts):
                """Wrapper que maneja correctamente la tokenización y predicción de BERT"""
                predictions = []
                
                with torch.no_grad():
                    for text in texts:
                        if isinstance(text, str) and text.strip():
                            # Tokenizar con el tokenizer de BERT
                            encoding = self.tokenizer.encode_plus(
                                text,
                                add_special_tokens=True,
                                max_length=self.max_len,
                                return_token_type_ids=False,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt',
                                truncation=True
                            )
                            
                            input_ids = encoding['input_ids'].to(self.device)
                            attention_mask = encoding['attention_mask'].to(self.device)
                            
                            # Predicción con BERT
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                            else:
                                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                            
                            # Obtener probabilidades
                            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            predictions.append(probabilities[0].cpu().numpy())
                        else:
                            # Para textos vacíos o inválidos
                            predictions.append(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
                
                return np.array(predictions)
            
            # Intentar análisis SHAP robusto
            try:
                st.info("🔬 Iniciando análisis SHAP avanzado con PartitionExplainer...")
                
                # Crear explainer con partición optimizada para texto
                explainer = shap.PartitionExplainer(bert_predict_wrapper, shap.maskers.Text(tokenizer=r'\W+'))
                
                # Calcular valores SHAP
                with st.spinner("Calculando importancia de palabras con SHAP..."):
                    shap_values = explainer([review_text], max_evals=max_evals)
                
                # Procesar resultados
                if hasattr(shap_values, 'data') and hasattr(shap_values, 'values'):
                    # Extraer tokens y valores
                    tokens = shap_values.data[0] if hasattr(shap_values.data[0], '__iter__') else [review_text]
                    values = shap_values.values[0]
                    
                    # Si values es multidimensional, tomar la clase predicha
                    if len(values.shape) > 1:
                        # Obtener la clase predicha
                        pred_class = np.argmax(bert_predict_wrapper([review_text])[0])
                        values = values[:, pred_class]
                    
                    # Crear análisis estructurado
                    shap_analysis = []
                    for i, (token, value) in enumerate(zip(tokens, values)):
                        if isinstance(token, str) and token.strip() and len(token.strip()) > 1:
                            # Filtrar tokens especiales de BERT
                            if token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                                sentiment = 'Positivo' if value > 0 else 'Negativo'
                                shap_analysis.append({
                                    'word': token.strip(),
                                    'importance': abs(value),
                                    'sentiment': sentiment,
                                    'impact': value,
                                    'shap_value': value
                                })
                    
                    # Ordenar por importancia
                    shap_analysis.sort(key=lambda x: x['importance'], reverse=True)
                    
                    st.success("✅ Análisis SHAP completado exitosamente")
                    return shap_analysis[:12]  # Top 12 palabras más importantes
                
            except Exception as shap_error:
                st.warning(f"Error con PartitionExplainer: {shap_error}")
                
                # Fallback a análisis de perturbación más sofisticado
                try:
                    st.info("🔄 Usando análisis de perturbación como fallback...")
                    
                    # Tokenización mejorada
                    tokens = self.tokenizer.tokenize(review_text)
                    
                    # Obtener predicción base
                    base_pred = bert_predict_wrapper([review_text])[0]
                    predicted_class = np.argmax(base_pred)
                    base_confidence = base_pred[predicted_class]
                    
                    word_impacts = {}
                    
                    # Análisis de perturbación por token
                    for i, token in enumerate(tokens):
                        if token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                            # Crear versión sin este token
                            modified_tokens = tokens[:i] + ['[MASK]'] + tokens[i+1:]
                            modified_text = self.tokenizer.convert_tokens_to_string(modified_tokens)
                            
                            # Predicción modificada
                            modified_pred = bert_predict_wrapper([modified_text])[0]
                            modified_confidence = modified_pred[predicted_class]
                            
                            # Calcular impacto
                            impact = base_confidence - modified_confidence
                            word_impacts[token] = impact
                    
                    # Crear análisis estructurado
                    shap_analysis = []
                    for token, impact in word_impacts.items():
                        if abs(impact) > 0.001:  # Filtrar impactos muy pequeños
                            sentiment = 'Positivo' if impact > 0 else 'Negativo'
                            shap_analysis.append({
                                'word': token.replace('##', ''),  # Limpiar subwords de BERT
                                'importance': abs(impact),
                                'sentiment': sentiment,
                                'impact': impact,
                                'shap_value': impact
                            })
                    
                    # Ordenar por importancia
                    shap_analysis.sort(key=lambda x: x['importance'], reverse=True)
                    
                    st.success("✅ Análisis de perturbación completado")
                    return shap_analysis[:10]
                    
                except Exception as fallback_error:
                    st.warning(f"Error en análisis de perturbación: {fallback_error}")
                    return self.analyze_keywords(review_text)
            
        except Exception as e:
            st.warning(f"Error general en análisis SHAP: {e}. Usando análisis básico.")
            return self.analyze_keywords(review_text)

    def generate_wordcloud(self, reviews_text, sentiment='all'):
        """Generar nube de palabras"""
        if not DEPS['wordcloud']:
            return None
        
        try:
            from wordcloud import WordCloud
            
            # Combinar todo el texto
            if isinstance(reviews_text, list):
                combined_text = ' '.join(reviews_text)
            else:
                combined_text = reviews_text
            
            # Palabras a filtrar
            stopwords = {
                'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 
                'da', 'su', 'por', 'son', 'con', 'para', 'al', 'está', 'una', 'del', 'los', 'las',
                'hotel', 'hoteles', 'habitación', 'habitaciones', 'muy', 'más', 'pero', 'este',
                'esta', 'están', 'tiene', 'todo', 'bien', 'día', 'días', 'fue', 'ser', 'hay'
            }
            
            # Configurar WordCloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=stopwords,
                max_words=100,
                colormap='viridis',
                collocations=False
            ).generate(combined_text)
            
            return wordcloud
            
        except Exception as e:
            st.error(f"Error generando nube de palabras: {e}")
            return None

    def analyze_dataset_stats(self, df, text_column, prediction_column):
        """Análisis estadístico completo del dataset"""
        try:
            stats = {}
            
            # Estadísticas básicas
            stats['total_reviews'] = len(df)
            stats['avg_stars'] = df[prediction_column].mean()
            stats['std_stars'] = df[prediction_column].std()
            
            # Distribución por estrellas
            distribution = df[prediction_column].value_counts().sort_index()
            stats['distribution'] = distribution.to_dict()
            
            # Análisis de longitud de texto
            df['text_length'] = df[text_column].str.len()
            df['word_count'] = df[text_column].str.split().str.len()
            
            stats['avg_text_length'] = df['text_length'].mean()
            stats['avg_word_count'] = df['word_count'].mean()
            
            # Correlación longitud vs rating
            stats['length_rating_corr'] = df['word_count'].corr(df[prediction_column])
            
            # Análisis por rangos de estrellas
            stats['low_ratings'] = len(df[df[prediction_column] <= 2])
            stats['medium_ratings'] = len(df[df[prediction_column] == 3])
            stats['high_ratings'] = len(df[df[prediction_column] >= 4])
            
            return stats
            
        except Exception as e:
            st.error(f"Error en análisis estadístico: {e}")
            return None

    def visualize_shap_results(self, review_text, shap_analysis):
        """Crear visualización avanzada de resultados SHAP"""
        if not shap_analysis:
            return None
        
        try:
            # Crear visualización de palabras con colores
            words = review_text.split()
            word_scores = {item['word']: item['impact'] for item in shap_analysis}
            
            # Crear HTML con palabras coloreadas
            html_parts = []
            html_parts.append('<div style="font-size: 18px; line-height: 2.0; padding: 20px; border-radius: 10px; background-color: #f8f9fa;">')
            
            for word in words:
                # Buscar coincidencia exacta o parcial
                score = 0
                for analyzed_word, impact in word_scores.items():
                    if analyzed_word.lower() in word.lower() or word.lower() in analyzed_word.lower():
                        score = impact
                        break
                
                # Determinar color basado en el impacto
                if abs(score) > 0.01:  # Solo colorear palabras con impacto significativo
                    if score > 0:
                        # Palabras positivas en verde
                        intensity = min(int(abs(score) * 1000), 100)
                        color = f"background-color: rgba(0, 255, 0, {intensity/100}); padding: 2px 4px; border-radius: 3px;"
                    else:
                        # Palabras negativas en rojo
                        intensity = min(int(abs(score) * 1000), 100)
                        color = f"background-color: rgba(255, 0, 0, {intensity/100}); padding: 2px 4px; border-radius: 3px;"
                    
                    html_parts.append(f'<span style="{color}" title="Impacto: {score:.4f}">{word}</span> ')
                else:
                    html_parts.append(f'{word} ')
            
            html_parts.append('</div>')
            html_parts.append('<p style="font-size: 12px; color: #666; margin-top: 10px;">')
            html_parts.append('💡 <strong>Verde:</strong> palabras que aumentan el rating | <strong>Rojo:</strong> palabras que disminuyen el rating')
            html_parts.append('</p>')
            
            return ''.join(html_parts)
            
        except Exception as e:
            st.warning(f"Error creando visualización SHAP: {e}")
            return None

    def create_shap_bar_chart(self, shap_analysis, top_n=8):
        """Crear gráfico de barras para los resultados SHAP"""
        if not shap_analysis:
            return None
        
        try:
            # Tomar los top N más importantes
            top_words = shap_analysis[:top_n]
            
            words = [item['word'] for item in top_words]
            impacts = [item['impact'] for item in top_words]
            colors = ['green' if impact > 0 else 'red' for impact in impacts]
            
            # Crear gráfico con Plotly
            fig = go.Figure(data=[
                go.Bar(
                    x=impacts,
                    y=words,
                    orientation='h',
                    marker=dict(
                        color=colors,
                        opacity=0.7
                    ),
                    text=[f"{impact:.4f}" for impact in impacts],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Importancia de Palabras (Análisis SHAP)",
                xaxis_title="Impacto en la Predicción",
                yaxis_title="Palabras",
                height=400,
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Error creando gráfico SHAP: {e}")
            return None
def main():
    st.markdown('<h1 class="main-header">🏨 Analizador de Comentarios Hoteleros Avanzado</h1>', unsafe_allow_html=True)
    st.markdown("### Análisis de Satisfacción de Clientes usando BERT en Español con IA Avanzada")
    
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
    
    # Sidebar con opciones
    st.sidebar.header("🔧 Configuraciones")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Análisis Individual", "📊 Análisis por Lotes", "📈 Dashboard", "🔍 Análisis Avanzado"])
    
    with tab1:
        st.header("📝 Análisis Individual de Comentarios")
        
        # Opciones de análisis
        col_options1, col_options2 = st.columns(2)
        with col_options1:
            use_shap = st.checkbox("🧠 Usar análisis SHAP avanzado", value=DEPS['shap'])
        with col_options2:
            show_wordcloud = st.checkbox("☁️ Generar nube de palabras", value=DEPS['wordcloud'])
        
        # Botones de ejemplo
        col_example1, col_example2, col_example3 = st.columns(3)
        
        with col_example1:
            if st.button("📝 Ejemplo Positivo", help="Cargar un comentario positivo de ejemplo"):
                st.session_state.example_text = "El hotel es absolutamente increíble! Las habitaciones son espaciosas y muy limpias, el personal es extremadamente amable y servicial. La ubicación es perfecta, cerca de todos los sitios de interés. El desayuno buffet es delicioso con mucha variedad. Definitivamente recomendaría este hotel a cualquiera que visite la ciudad."
        
        with col_example2:
            if st.button("📝 Ejemplo Negativo", help="Cargar un comentario negativo de ejemplo"):
                st.session_state.example_text = "Terrible experiencia en este hotel. Las habitaciones están sucias y huelen mal, el aire acondicionado no funciona correctamente. El personal es grosero y poco profesional. La ubicación es ruidosa y el wifi es extremadamente lento. El desayuno es básico y de mala calidad. No lo recomiendo para nada."
        
        with col_example3:
            if st.button("📝 Ejemplo Mixto", help="Cargar un comentario con aspectos positivos y negativos"):
                st.session_state.example_text = "El hotel tiene una ubicación excelente y las vistas son hermosas. Sin embargo, las habitaciones son pequeñas y un poco anticuadas. El personal es amable pero el servicio es lento. El precio está bien para lo que ofreces, aunque esperaba algo mejor en el desayuno."
        
        # Usar texto de ejemplo si está disponible
        if 'example_text' in st.session_state:
            review_text = st.text_area(
                "Escribe o pega un comentario de hotel:",
                value=st.session_state.example_text,
                height=120,
                placeholder="Ejemplo: El hotel es excelente, las habitaciones están muy limpias y el personal es muy amable..."
            )
            # Limpiar el texto de ejemplo después de usarlo
            del st.session_state.example_text
        else:
            review_text = st.text_area(
                "Escribe o pega un comentario de hotel:",
                height=120,
                placeholder="Ejemplo: El hotel es excelente, las habitaciones están muy limpias y el personal es muy amable..."
            )
        if st.button("🔍 Analizar Comentario", type="primary"):
            if review_text.strip():
                with st.spinner('Analizando comentario con IA avanzada...'):
                    # Hacer predicción
                    predicted_stars, confidence, probabilities = analyzer.predict_single_review(review_text)
                    
                    if predicted_stars is not None:
                        # Mostrar resultados principales
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2>⭐ Predicción: {predicted_stars} Estrellas</h2>
                                <p>Confianza: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Gráfico de probabilidades mejorado
                            fig = px.bar(
                                x=analyzer.class_names,
                                y=probabilities,
                                title="Distribución de Probabilidades",
                                labels={'x': 'Rating', 'y': 'Probabilidad'},
                                color=probabilities,
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Análisis de interpretabilidad
                        st.subheader("🔍 Análisis de Interpretabilidad")
                        
                        if use_shap and DEPS['shap']:
                            with st.spinner("Calculando importancia de palabras con SHAP..."):
                                analysis = analyzer.analyze_shap_explanation(review_text)
                                
                                if analysis:
                                    # Visualización SHAP avanzada
                                    st.subheader("🎨 Visualización SHAP Interactiva")
                                    
                                    # Mostrar texto con palabras coloreadas
                                    shap_html = analyzer.visualize_shap_results(review_text, analysis)
                                    if shap_html:
                                        st.markdown(shap_html, unsafe_allow_html=True)
                                    
                                    # Gráfico de barras SHAP
                                    col_shap1, col_shap2 = st.columns([2, 1])
                                    
                                    with col_shap1:
                                        shap_chart = analyzer.create_shap_bar_chart(analysis)
                                        if shap_chart:
                                            st.plotly_chart(shap_chart, use_container_width=True)
                                    
                                    with col_shap2:
                                        st.markdown("**📊 Métricas SHAP:**")
                                        if analysis:
                                            total_positive = sum(1 for item in analysis if item['impact'] > 0)
                                            total_negative = sum(1 for item in analysis if item['impact'] < 0)
                                            max_impact = max(abs(item['impact']) for item in analysis)
                                            
                                            st.metric("Palabras Positivas", total_positive)
                                            st.metric("Palabras Negativas", total_negative)
                                            st.metric("Impacto Máximo", f"{max_impact:.4f}")
                        else:
                            analysis = analyzer.analyze_keywords(review_text)
                        
                        # Análisis detallado de palabras clave
                        if analysis:
                            st.subheader("📝 Análisis Detallado de Palabras")
                            
                            col_analysis1, col_analysis2 = st.columns(2)
                            
                            positive_words = [item for item in analysis if item.get('sentiment') == 'Positivo' or item.get('impact', 0) > 0]
                            negative_words = [item for item in analysis if item.get('sentiment') == 'Negativo' or item.get('impact', 0) < 0]
                            
                            with col_analysis1:
                                st.markdown("**🟢 Palabras con Influencia Positiva:**")
                                if positive_words:
                                    for item in positive_words[:5]:
                                        impact_val = item.get('impact', item.get('importance', 0))
                                        st.write(f"• **{item['word']}**: {impact_val:+.3f}")
                                else:
                                    st.write("No se encontraron palabras positivas significativas")
                            
                            with col_analysis2:
                                st.markdown("**🔴 Palabras con Influencia Negativa:**")
                                if negative_words:
                                    for item in negative_words[:5]:
                                        impact_val = item.get('impact', item.get('importance', 0))
                                        st.write(f"• **{item['word']}**: {impact_val:+.3f}")
                                else:
                                    st.write("No se encontraron palabras negativas significativas")
                        
                        # Nube de palabras individual
                        if show_wordcloud and DEPS['wordcloud']:
                            st.subheader("☁️ Nube de Palabras del Comentario")
                            wordcloud = analyzer.generate_wordcloud(review_text)
                            if wordcloud:
                                try:
                                    import matplotlib.pyplot as plt
                                    fig_wc, ax = plt.subplots(figsize=(10, 5))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig_wc)
                                    plt.close()
                                except ImportError:
                                    st.warning("Matplotlib no está disponible para mostrar la nube de palabras")
                        
                        st.success("✅ Análisis completado exitosamente!")
                    else:
                        st.error("❌ Error al procesar el comentario")
            else:
                st.warning("Por favor, ingresa un comentario para analizar.")
    
    with tab2:
        st.header("📊 Análisis por Lotes de Comentarios")
        
        # Opciones de entrada
        input_method = st.radio(
            "Selecciona el método de entrada:",
            ["📝 Texto manual (separado por líneas)", "📁 Cargar archivo CSV", "🗂️ Usar dataset de ejemplo"]
        )
        
        reviews_to_analyze = []
        
        if input_method == "📝 Texto manual (separado por líneas)":
            batch_text = st.text_area(
                "Ingresa múltiples comentarios (uno por línea):",
                height=200,
                placeholder="Comentario 1\nComentario 2\nComentario 3..."
            )
            if batch_text.strip():
                reviews_to_analyze = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        elif input_method == "📁 Cargar archivo CSV":
            uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Vista previa del archivo:")
                    st.dataframe(df.head())
                    
                    columns = df.columns.tolist()
                    text_column = st.selectbox("Selecciona la columna con los comentarios:", columns)
                    
                    if text_column:
                        reviews_to_analyze = df[text_column].dropna().tolist()[:100]  # Limitar a 100 para evitar sobrecarga
                        st.info(f"📊 Se analizarán {len(reviews_to_analyze)} comentarios")
                
                except Exception as e:
                    st.error(f"Error al leer el archivo: {e}")
        
        elif input_method == "🗂️ Usar dataset de ejemplo":
            # Verificar si existe el dataset de ejemplo
            example_files = ["tripadvisor_hotel_reviews.csv", "comentarios_ejemplo.csv"]
            available_file = None
            
            for file in example_files:
                if os.path.exists(file):
                    available_file = file
                    break
            
            if available_file:
                try:
                    df_example = pd.read_csv(available_file)
                    st.write(f"Dataset de ejemplo cargado: {available_file}")
                    st.dataframe(df_example.head())
                    
                    # Detectar columna de texto automáticamente
                    text_cols = [col for col in df_example.columns if 'review' in col.lower() or 'comment' in col.lower() or 'text' in col.lower()]
                    if text_cols:
                        text_column = text_cols[0]
                    else:
                        text_column = st.selectbox("Selecciona la columna con los comentarios:", df_example.columns.tolist())
                    
                    num_samples = st.slider("Número de muestras a analizar:", 5, min(100, len(df_example)), 20)
                    reviews_to_analyze = df_example[text_column].dropna().head(num_samples).tolist()
                    
                except Exception as e:
                    st.error(f"Error al cargar dataset de ejemplo: {e}")
            else:
                st.warning("No se encontró dataset de ejemplo. Sube tu propio archivo o usa texto manual.")
        
        # Análisis por lotes
        if reviews_to_analyze and st.button("🚀 Analizar Lote de Comentarios", type="primary"):
            with st.spinner(f'Analizando {len(reviews_to_analyze)} comentarios...'):
                
                # Configurar barra de progreso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Realizar predicciones por lotes
                predictions, confidences, probabilities_list = analyzer.predict_batch(reviews_to_analyze, batch_size=4)
                
                if predictions:
                    # Crear DataFrame con resultados
                    results_df = pd.DataFrame({
                        'Comentario': reviews_to_analyze,
                        'Estrellas_Predichas': predictions,
                        'Confianza': confidences,
                        'Longitud': [len(review) if review else 0 for review in reviews_to_analyze]
                    })
                    
                    # Mostrar resultados
                    st.success(f"✅ Análisis completado para {len(predictions)} comentarios!")
                    
                    # Métricas generales
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    
                    with col_metric1:
                        avg_rating = np.mean([p for p in predictions if p is not None])
                        st.metric("📊 Rating Promedio", f"{avg_rating:.2f}")
                    
                    with col_metric2:
                        avg_confidence = np.mean([c for c in confidences if c is not None])
                        st.metric("🎯 Confianza Promedio", f"{avg_confidence:.2%}")
                    
                    with col_metric3:
                        positive_reviews = len([p for p in predictions if p and p >= 4])
                        st.metric("😊 Reseñas Positivas", f"{positive_reviews}")
                    
                    with col_metric4:
                        negative_reviews = len([p for p in predictions if p and p <= 2])
                        st.metric("😞 Reseñas Negativas", f"{negative_reviews}")
                    
                    # Visualizaciones
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Distribución de ratings
                        rating_counts = pd.Series([p for p in predictions if p is not None]).value_counts().sort_index()
                        fig_dist = px.bar(
                            x=rating_counts.index,
                            y=rating_counts.values,
                            title="Distribución de Ratings Predichos",
                            labels={'x': 'Estrellas', 'y': 'Cantidad'},
                            color=rating_counts.values,
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col_viz2:
                        # Relación longitud vs rating
                        valid_results = results_df.dropna()
                        if len(valid_results) > 0:
                            fig_scatter = px.scatter(
                                valid_results,
                                x='Longitud',
                                y='Estrellas_Predichas',
                                size='Confianza',
                                title="Longitud vs Rating",
                                labels={'Longitud': 'Caracteres', 'Estrellas_Predichas': 'Estrellas'},
                                color='Confianza',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Tabla de resultados detallados
                    st.subheader("📋 Resultados Detallados")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Exportar resultados
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Descargar Resultados (CSV)",
                        data=csv_data,
                        file_name=f"analisis_comentarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("❌ Error al procesar los comentarios en lote")
    
    with tab3:
        st.header("📈 Dashboard de Análisis")
        
        # Aquí podríamos cargar datos históricos o usar el dataset de ejemplo
        dashboard_source = st.radio(
            "Fuente de datos para el dashboard:",
            ["🗂️ Dataset de ejemplo", "📁 Cargar datos históricos"]
        )
        
        if dashboard_source == "🗂️ Dataset de ejemplo":
            example_files = ["tripadvisor_hotel_reviews.csv", "comentarios_ejemplo.csv"]
            available_file = None
            
            for file in example_files:
                if os.path.exists(file):
                    available_file = file
                    break
            
            if available_file:
                df_dashboard = pd.read_csv(available_file)
                
                # Detectar columnas automáticamente
                text_cols = [col for col in df_dashboard.columns if 'review' in col.lower() or 'comment' in col.lower() or 'text' in col.lower()]
                rating_cols = [col for col in df_dashboard.columns if 'rating' in col.lower() or 'star' in col.lower() or 'score' in col.lower()]
                
                if text_cols and rating_cols:
                    text_col = text_cols[0]
                    rating_col = rating_cols[0]
                    
                    # Análisis estadístico
                    stats = analyzer.analyze_dataset_stats(df_dashboard, text_col, rating_col)
                    
                    if stats:
                        # Métricas del dashboard
                        col_dash1, col_dash2, col_dash3, col_dash4 = st.columns(4)
                        
                        with col_dash1:
                            st.metric("📊 Total Reseñas", f"{stats['total_reviews']:,}")
                        
                        with col_dash2:
                            st.metric("⭐ Rating Promedio", f"{stats['avg_stars']:.2f}")
                        
                        with col_dash3:
                            st.metric("📝 Palabras Promedio", f"{stats['avg_word_count']:.0f}")
                        
                        with col_dash4:
                            correlation = stats.get('length_rating_corr', 0)
                            st.metric("🔗 Correlación Longitud-Rating", f"{correlation:.3f}")
                        
                        # Visualizaciones del dashboard
                        col_dash_viz1, col_dash_viz2 = st.columns(2)
                        
                        with col_dash_viz1:
                            # Distribución de ratings
                            distribution_data = list(stats['distribution'].items())
                            if distribution_data:
                                ratings, counts = zip(*distribution_data)
                                fig_dist_dash = px.pie(
                                    values=counts,
                                    names=[f"{r} Estrellas" for r in ratings],
                                    title="Distribución de Ratings en Dataset"
                                )
                                st.plotly_chart(fig_dist_dash, use_container_width=True)
                        
                        with col_dash_viz2:
                            # Categorización por sentimiento
                            sentiment_data = {
                                'Negativo (1-2★)': stats['low_ratings'],
                                'Neutral (3★)': stats['medium_ratings'],
                                'Positivo (4-5★)': stats['high_ratings']
                            }
                            
                            fig_sentiment = px.bar(
                                x=list(sentiment_data.keys()),
                                y=list(sentiment_data.values()),
                                title="Categorización por Sentimiento",
                                color=list(sentiment_data.values()),
                                color_continuous_scale='RdYlGn'
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        # Análisis de nube de palabras del dataset
                        if DEPS['wordcloud']:
                            st.subheader("☁️ Nube de Palabras del Dataset")
                            sample_reviews = df_dashboard[text_col].dropna().head(200).tolist()
                            wordcloud_dataset = analyzer.generate_wordcloud(sample_reviews)
                            
                            if wordcloud_dataset:
                                try:
                                    import matplotlib.pyplot as plt
                                    fig_wc_dataset, ax = plt.subplots(figsize=(12, 6))
                                    ax.imshow(wordcloud_dataset, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig_wc_dataset)
                                    plt.close()
                                except ImportError:
                                    st.warning("Matplotlib no está disponible para mostrar la nube de palabras")
                else:
                    st.warning("No se pudieron detectar automáticamente las columnas de texto y rating.")
            else:
                st.warning("No se encontró dataset de ejemplo para el dashboard.")
    
    with tab4:
        st.header("🔍 Análisis Avanzado y Comparaciones")
        
        st.subheader("🆚 Comparación de Comentarios")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            review1 = st.text_area("Comentario 1:", height=100, placeholder="Ingresa el primer comentario...")
        
        with col_comp2:
            review2 = st.text_area("Comentario 2:", height=100, placeholder="Ingresa el segundo comentario...")
        
        if st.button("🔍 Comparar Comentarios"):
            if review1.strip() and review2.strip():
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.markdown("**📝 Análisis Comentario 1:**")
                    pred1, conf1, prob1 = analyzer.predict_single_review(review1)
                    if pred1:
                        st.write(f"⭐ Predicción: {pred1} estrellas")
                        st.write(f"🎯 Confianza: {conf1:.2%}")
                        
                        # Análisis de palabras clave
                        keywords1 = analyzer.analyze_keywords(review1)
                        if keywords1:
                            positive1 = [k for k in keywords1 if k['sentiment'] == 'Positivo']
                            if positive1:
                                st.write("🟢 Palabras positivas:")
                                for k in positive1[:3]:
                                    st.write(f"  • {k['word']}")
                
                with col_result2:
                    st.markdown("**📝 Análisis Comentario 2:**")
                    pred2, conf2, prob2 = analyzer.predict_single_review(review2)
                    if pred2:
                        st.write(f"⭐ Predicción: {pred2} estrellas")
                        st.write(f"🎯 Confianza: {conf2:.2%}")
                        
                        # Análisis de palabras clave
                        keywords2 = analyzer.analyze_keywords(review2)
                        if keywords2:
                            positive2 = [k for k in keywords2 if k['sentiment'] == 'Positivo']
                            if positive2:
                                st.write("🟢 Palabras positivas:")
                                for k in positive2[:3]:
                                    st.write(f"  • {k['word']}")
                
                # Comparación visual
                if pred1 and pred2:
                    st.subheader("📊 Comparación Visual")
                    
                    comparison_data = {
                        'Comentario': ['Comentario 1', 'Comentario 2'],
                        'Estrellas': [pred1, pred2],
                        'Confianza': [conf1, conf2],
                        'Longitud': [len(review1), len(review2)]
                    }
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    fig_comparison = px.bar(
                        df_comparison,
                        x='Comentario',
                        y='Estrellas',
                        color='Confianza',
                        title="Comparación de Predicciones",
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Análisis de tendencias (si hay datos históricos)
        st.subheader("📈 Análisis de Tendencias")
        
        trend_info = st.info("""
        💡 **Funcionalidad de Tendencias:**
        - Análisis temporal de satisfacción
        - Identificación de patrones estacionales
        - Comparación entre períodos
        - Predicción de tendencias futuras
        
        *Disponible cuando se cargan datos con marcas temporales*
        """)

    # Información de estado en sidebar
    st.sidebar.header("🔧 Estado del Sistema")
    
    # Estado de GPU/CPU
    if torch.cuda.is_available():
        st.sidebar.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        memory_info = analyzer.get_gpu_memory_info()
        if memory_info:
            st.sidebar.info(f"💾 Memoria: {memory_info['utilization_pct']:.1f}% utilizada")
            st.sidebar.progress(memory_info['utilization_pct'] / 100)
    else:
        st.sidebar.info("🖥️ Modo CPU")
    
    # Estado de dependencias
    st.sidebar.subheader("📦 Dependencias")
    deps_status = [
        ("Transformers", DEPS['transformers'], "🤖"),
        ("SHAP", DEPS['shap'], "🧠"),
        ("WordCloud", DEPS['wordcloud'], "☁️"),
        ("Matplotlib", DEPS['matplotlib'], "📊"),
        ("Seaborn", DEPS['seaborn'], "🎨")
    ]
    
    for name, status, icon in deps_status:
        if status:
            st.sidebar.success(f"{icon} {name}")
        else:
            st.sidebar.warning(f"⚠️ {name} no disponible")
    
    st.sidebar.success("✅ Sistema optimizado")
    
    # Información adicional
    with st.sidebar.expander("ℹ️ Información del Sistema"):
        st.write(f"""
        **Versión:** 2.0.0 Avanzada
        **Modelo:** BERT Español Optimizado
        **Última actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
        **Características:**
        - Análisis individual y por lotes
        - Dashboard interactivo
        - Análisis SHAP avanzado
        - Nubes de palabras
        - Comparación de comentarios
        - Exportación de resultados
        """)
    
    # Información sobre SHAP
    if DEPS['shap']:
        with st.sidebar.expander("🧠 Sobre el Análisis SHAP"):
            st.write("""
            **SHAP (SHapley Additive exPlanations)** explica las predicciones del modelo:
            
            **🎯 Cómo funciona:**
            - Calcula la contribución de cada palabra
            - Usa teoría de juegos para asignar importancia
            - Colorea palabras según su impacto
            
            **🎨 Visualización:**
            - 🟢 Verde: aumenta el rating
            - 🔴 Rojo: disminuye el rating
            - Intensidad = importancia
            
            **⚡ Rendimiento:**
            - Análisis completo con PartitionExplainer
            - Fallback a análisis de perturbación
            - Optimizado para BERT
            """)
    
    # Tips de uso
    with st.sidebar.expander("💡 Tips de Uso"):
        st.write("""
        **Para mejores resultados:**
        
        📝 **Texto:**
        - Comentarios de 20-300 caracteres
        - Texto claro y coherente
        - Evitar abreviaciones excesivas
        
        🧠 **Análisis SHAP:**
        - Más preciso con GPU
        - Toma 10-30 segundos
        - Muestra palabras más influyentes
        
        📊 **Análisis por Lotes:**
        - Máximo 100 comentarios
        - Usa CSV para grandes datasets
        - Exporta resultados para análisis
        """)

if __name__ == "__main__":
    main()
