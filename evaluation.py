import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def run_full_evaluation(model, beats, labels):
    """Run evaluation on full dataset and return metrics"""
    X = beats[..., np.newaxis]
    preds = np.argmax(model.predict(X, verbose=0), axis=1)
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    report = classification_report(
        labels, preds,
        target_names=le.classes_,
        output_dict=True
    )
    
    cm = confusion_matrix(labels, preds)
    
    return report, cm, le.classes_

def plot_comparison():
    """Plot comparison between our model and existing papers"""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    papers = [
        'Alamatsaz\net al. 2024',
        'Huillcen\nBaca 2025',
        'Sachenko\net al. 2025',
        'Talukder\net al. 2025',
        'Our CNN-LSTM\n(Random Split)',
        'Our CNN-LSTM\n(Inter-Patient)'
    ]
    
    accuracies = [98.24, 99.57, 99.13, 99.74, 99.0, 76.0]
    colors = ['#444', '#444', '#444', '#444', '#2ecc71', '#f39c12']
    
    bars = ax.barh(papers, accuracies, color=colors)
    ax.set_xlabel('Accuracy (%)', color='white')
    ax.set_xlim(0, 105)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.set_title('Model Comparison with Existing Literature', color='white')
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc}%', va='center', color='white', fontsize=9)
    
    plt.tight_layout()
    return fig