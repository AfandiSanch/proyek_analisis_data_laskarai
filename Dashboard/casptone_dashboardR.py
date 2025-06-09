import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import cv2

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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the trained model. Replace with your actual model path."""
    try:
        # Try to load your trained model
        # model = tf.keras.models.load_model('path_to_your_model.h5')
        
        # For demo purposes, create a model with same architecture as notebook
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction - matching notebook preprocessing."""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV (if needed)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (224x224 as per notebook)
        img_array = cv2.resize(img_array, (224, 224))
        
        # Normalize pixel values to [0, 1] as per notebook
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image_array):
    """Make prediction on the preprocessed image."""
    try:
        if model is None or image_array is None:
            # Return random probabilities for demo
            predictions = np.random.rand(6)
            predictions = predictions / np.sum(predictions)
            return predictions
        
        # Make actual prediction
        predictions = model.predict(image_array, verbose=0)
        return predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        # Return random probabilities as fallback
        predictions = np.random.rand(6)
        predictions = predictions / np.sum(predictions)
        return predictions

# Class information based on your notebook data
CLASS_INFO = {
    'cardboard': {
        'original_count': 393, 
        'augmented_count': 393,  # No augmentation for cardboard
        'color': '#8B4513', 
        'description': 'Recyclable cardboard materials including boxes and packaging'
    },
    'glass': {
        'original_count': 491, 
        'augmented_count': 491,  # No augmentation for glass
        'color': '#00CED1', 
        'description': 'Glass bottles, jars, and containers'
    },
    'metal': {
        'original_count': 400, 
        'augmented_count': 700,  # Augmented by 300 images
        'color': '#C0C0C0', 
        'description': 'Metal cans, aluminum containers, and metallic objects'
    },
    'paper': {
        'original_count': 584, 
        'augmented_count': 884,  # Augmented by 300 images
        'color': '#F5DEB3', 
        'description': 'Paper documents, newspapers, and paper materials'
    },
    'plastic': {
        'original_count': 472, 
        'augmented_count': 472,  # No augmentation for plastic
        'color': '#FF6347', 
        'description': 'Plastic bottles, containers, and plastic waste'
    },
    'trash': {
        'original_count': 127, 
        'augmented_count': 427,  # Augmented by 300 images
        'color': '#696969', 
        'description': 'General waste and non-recyclable trash'
    }
}

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🗂️ Garbage Classification Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dataset Overview")
        st.markdown('<div class="class-info">', unsafe_allow_html=True)
        st.markdown("**Garbage Classification Dataset**")
        st.markdown("Contains 6 classifications with data augmentation:")
        
        total_original = sum(info['original_count'] for info in CLASS_INFO.values())
        total_augmented = sum(info['augmented_count'] for info in CLASS_INFO.values())
        
        for class_name, info in CLASS_INFO.items():
            original_pct = (info['original_count'] / total_original) * 100
            augmented_pct = (info['augmented_count'] / total_augmented) * 100
            
            if info['augmented_count'] > info['original_count']:
                st.markdown(f"• **{class_name.title()}**: {info['original_count']} → {info['augmented_count']} images ({augmented_pct:.1f}%) *augmented*")
            else:
                st.markdown(f"• **{class_name.title()}**: {info['original_count']} images ({augmented_pct:.1f}%)")
        
        st.markdown(f"**Original Total**: {total_original}")
        st.markdown(f"**After Augmentation**: {total_augmented}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Model Architecture")
        st.info("""
        **CNN Model Details:**
        - Input: 224×224×3 RGB images
        - Conv2D layers: 32, 64 filters
        - MaxPooling2D layers
        - Dense layers: 128, 6 neurons
        - Optimizer: Adam (lr=0.0005)
        - Loss: Sparse Categorical Crossentropy
        - Data Split: 80% train, 20% validation
        """)
        
        st.markdown("### 🔧 Data Augmentation")
        st.info("""
        **Applied to metal, paper, trash:**
        - Rotation (clockwise/anticlockwise)
        - Brightness adjustment
        - Gaussian blur
        - Image shearing
        - Vertical flipping
        - Warp shifting
        """)
    
    # Warning about demo model
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Demo Notice:</strong> This is a demonstration using the model architecture from your notebook. 
        To use your trained model, replace the model loading code with your actual saved model path.
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="subheader">📤 Upload Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of garbage to classify (will be resized to 224×224)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Show image info
            st.markdown(f"**Image Info:** {image.size[0]}×{image.size[1]} pixels, Mode: {image.mode}")
            
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
                <p>Model processed 224×224 normalized image</p>
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
            
            sorted_indices = np.argsort(predictions)[::-1]  # Sort by probability descending
            
            for rank, idx in enumerate(sorted_indices):
                class_name = CLASS_NAMES[idx]
                prob = predictions[idx]
                
                with st.expander(f"#{rank+1} {class_name.title()} - {prob*100:.2f}%"):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.metric("Probability", f"{prob*100:.4f}%")
                        st.metric("Rank", f"#{rank+1}")
                        st.metric("Logit Score", f"{prob:.6f}")
                    with col_b:
                        st.write(f"**Description**: {CLASS_INFO[class_name]['description']}")
                        st.write(f"**Original samples**: {CLASS_INFO[class_name]['original_count']}")
                        st.write(f"**Training samples**: {CLASS_INFO[class_name]['augmented_count']}")
                        
                        # Progress bar
                        st.progress(float(prob))
        else:
            st.markdown('<div class="subheader">👆 Upload an image to see predictions</div>', unsafe_allow_html=True)
            st.info("Please upload an image file (PNG, JPG, or JPEG) to get started with garbage classification.")
            
            # Dataset visualization
            st.markdown("### 📊 Training Dataset Distribution")
            
            # Create dataset overview chart
            dataset_df = pd.DataFrame([
                {
                    'Class': cls, 
                    'Original': info['original_count'], 
                    'After Augmentation': info['augmented_count'],
                    'Color': info['color']
                } 
                for cls, info in CLASS_INFO.items()
            ])
            
            # Melt the dataframe for grouped bar chart
            dataset_melted = pd.melt(dataset_df, 
                                   id_vars=['Class', 'Color'], 
                                   value_vars=['Original', 'After Augmentation'],
                                   var_name='Dataset', value_name='Count')
            
            fig_dataset = px.bar(
                dataset_melted,
                x='Class',
                y='Count',
                color='Dataset',
                title="Training Dataset Distribution (Before & After Augmentation)",
                barmode='group'
            )
            fig_dataset.update_layout(showlegend=True)
            st.plotly_chart(fig_dataset, use_container_width=True)
            
            # Show augmentation details
            st.markdown("### 🔧 Data Augmentation Applied")
            augmented_classes = [cls for cls, info in CLASS_INFO.items() 
                               if info['augmented_count'] > info['original_count']]
            
            if augmented_classes:
                st.success(f"**Augmented classes**: {', '.join(augmented_classes)}")
                st.info("""
                **Augmentation techniques used:**
                - Anticlockwise & clockwise rotation (0-180°)
                - Brightness adjustment (gamma correction)
                - Gaussian blur (9×9 kernel)
                - Image shearing (AffineTransform)
                - Vertical flipping
                - Warp shifting (translation)
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🌱 Garbage Classification Dashboard | Built with Streamlit & TensorFlow</p>
        <p>Model Architecture: CNN with 224×224 input | Data Augmentation Applied ♻️</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()