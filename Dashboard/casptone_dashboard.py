import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="🗂️ Garbage Classification Dashboard",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .class-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the trained model. Replace with your model path."""
    # For demo purposes, create a dummy model
    # In actual implementation, replace with: tf.keras.models.load_model('your_model_path.h5')
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(6, activation="softmax")
    ])
    return model

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    # Resize image to model input size
    image = image.resize((256, 256))
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(model, image_array):
    """Make prediction on the preprocessed image."""
    # For demo purposes, return random probabilities
    # In actual implementation: predictions = model.predict(image_array)
    predictions = np.random.rand(1, 6)
    predictions = predictions / np.sum(predictions)  # Normalize to sum to 1
    return predictions[0]

# Class information
CLASS_INFO = {
    'cardboard': {'count': 393, 'color': '#8B4513', 'description': 'Recyclable cardboard materials'},
    'glass': {'count': 491, 'color': '#00CED1', 'description': 'Glass bottles and containers'},
    'metal': {'count': 400, 'color': '#C0C0C0', 'description': 'Metal cans and objects'},
    'paper': {'count': 584, 'color': '#F5DEB3', 'description': 'Paper documents and materials'},
    'plastic': {'count': 472, 'color': '#FF6347', 'description': 'Plastic bottles and containers'},
    'trash': {'count': 127, 'color': '#696969', 'description': 'General waste and trash'}
}

CLASS_NAMES = list(CLASS_INFO.keys())

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🗂️ Garbage Classification Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dataset Overview")
        st.markdown('<div class="class-info">', unsafe_allow_html=True)
        st.markdown("**Garbage Classification Dataset**")
        st.markdown("Contains 6 classifications:")
        
        total_images = sum(info['count'] for info in CLASS_INFO.values())
        
        for class_name, info in CLASS_INFO.items():
            percentage = (info['count'] / total_images) * 100
            st.markdown(f"• **{class_name.title()}**: {info['count']} images ({percentage:.1f}%)")
        
        st.markdown(f"**Total Images**: {total_images}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Model Info")
        st.info("CNN Model with 85-15% train-validation split")
        
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="subheader">📤 Upload Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of garbage to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Load model
            with st.spinner("Loading model..."):
                model = load_model()
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(image)
                predictions = predict_image(model, processed_image)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = predictions[predicted_class_idx] * 100
            
            # Display main prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Prediction</h2>
                <h1>{predicted_class.upper()}</h1>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<div class="subheader">📈 Prediction Probabilities</div>', unsafe_allow_html=True)
            
            # Create probability dataframe
            prob_df = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Probability': predictions * 100,
                'Color': [CLASS_INFO[cls]['color'] for cls in CLASS_NAMES]
            }).sort_values('Probability', ascending=True)
            
            # Bar chart
            fig_bar = px.bar(
                prob_df, 
                x='Probability', 
                y='Class',
                color='Color',
                color_discrete_map={color: color for color in prob_df['Color']},
                title="Class Probabilities (%)",
                labels={'Probability': 'Probability (%)', 'Class': 'Garbage Class'},
                orientation='h'
            )
            fig_bar.update_layout(
                showlegend=False,
                height=400,
                xaxis_range=[0, 100]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Pie chart
            fig_pie = px.pie(
                prob_df,
                values='Probability',
                names='Class',
                title="Probability Distribution",
                color='Class',
                color_discrete_map={cls: CLASS_INFO[cls]['color'] for cls in CLASS_NAMES}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Detailed metrics
            st.markdown('<div class="subheader">📋 Detailed Results</div>', unsafe_allow_html=True)
            
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                with st.expander(f"{class_name.title()} - {prob*100:.2f}%"):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.metric("Probability", f"{prob*100:.4f}%")
                        st.metric("Rank", f"#{len(CLASS_NAMES) - i}")
                    with col_b:
                        st.write(f"**Description**: {CLASS_INFO[class_name]['description']}")
                        st.write(f"**Training samples**: {CLASS_INFO[class_name]['count']}")
                        
                        # Progress bar
                        st.progress(prob)
        else:
            st.markdown('<div class="subheader">👆 Upload an image to see predictions</div>', unsafe_allow_html=True)
            st.info("Please upload an image file (PNG, JPG, or JPEG) to get started with garbage classification.")
            
            # Dataset visualization
            st.markdown("### 📊 Dataset Distribution")
            
            # Create dataset overview chart
            dataset_df = pd.DataFrame([
                {'Class': cls, 'Count': info['count'], 'Color': info['color']} 
                for cls, info in CLASS_INFO.items()
            ])
            
            fig_dataset = px.bar(
                dataset_df,
                x='Class',
                y='Count',
                color='Color',
                color_discrete_map={color: color for color in dataset_df['Color']},
                title="Training Dataset Distribution"
            )
            fig_dataset.update_layout(showlegend=False)
            st.plotly_chart(fig_dataset, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🌱 Garbage Classification Dashboard | Built with Streamlit & TensorFlow</p>
        <p>Help protect the environment by properly classifying waste! ♻️</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()