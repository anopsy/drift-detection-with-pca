import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def generate_data(corr_factor):
    data = pd.DataFrame({
        'X': np.random.normal(loc=0, scale=1, size=1000),
        'Z': np.random.normal(loc=0, scale=1, size=1000),
        'Y': np.random.normal(loc=0, scale=1, size=1000)
    })
    data['Z'] = data['X'] * corr_factor + data['Z'] * (1 - abs(corr_factor))
    data['Y'] = data['Z'] * corr_factor + data['Y'] * (1 - abs(corr_factor))
    return data


np.random.seed(42)
initial_corr_factor = -1.0
data_initial = generate_data(initial_corr_factor)


pca = PCA(n_components=2)
pca.fit(data_initial)

# df to store reconstruction errors
if 'reconstruction_errors_df' not in st.session_state:
    st.session_state['reconstruction_errors_df'] = pd.DataFrame(columns=['Correlation Factor', 'Reconstruction Error'])

# update data, calculate reconstruction error and plot
def update_plot(corr_factor):
    # Generate data with the current correlation factor
    data_current = generate_data(corr_factor)

    # Transform and inverse transform using the pretrained PCA
    data_pca = pca.transform(data_current)
    data_reconstructed = pca.inverse_transform(data_pca)

    # Calculate reconstruction error
    reconstruction_error = np.mean((data_current - data_reconstructed) ** 2, axis=1).mean()

    # I've chosen to add every correlation point set by the user but I can also introduce a check if the current correlation factor is already present
    
    new_row = pd.DataFrame({'Correlation Factor': [corr_factor], 'Reconstruction Error': [reconstruction_error]})
    st.session_state['reconstruction_errors_df'] = pd.concat([st.session_state['reconstruction_errors_df'], new_row], ignore_index=True)

    
    norm = Normalize(vmin=-1, vmax=1)
    colors = plt.cm.viridis(norm(corr_factor))


    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 4, 1], height_ratios=[1, 4, 1], wspace=0.2, hspace=0.2)
    ax_scatter = fig.add_subplot(gs[1, 1], projection='3d')
    ax_histx = fig.add_subplot(gs[0, 1])
    ax_histy = fig.add_subplot(gs[1, 2])
    ax_line = fig.add_subplot(gs[2, 1])

    # 3D Scatter plot
    scatter = ax_scatter.scatter(data_current['X'], data_current['Y'], data_current['Z'], c=colors, cmap='plasma')
    ax_scatter.set_xlabel('X')
    ax_scatter.set_ylabel('Y')
    ax_scatter.set_zlabel('Z')
    ax_scatter.set_xlim(-3, 3)
    ax_scatter.set_ylim(-3, 3)
    ax_scatter.set_zlim(-3, 3)

    # Y distribution with fixed axes
    sns.histplot(data_current['Y'], ax=ax_histx, kde=True, color="blueviolet")
    ax_histx.set_xlim(-3, 3)
    ax_histx.set_xlabel('Y')
    ax_histx.set_ylabel('Frequency')

    # Z distribution with fixed axes
    sns.histplot(y=data_current['Z'], ax=ax_histy, kde=True, color="fuchsia")
    ax_histy.set_ylim(-3, 3)
    ax_histy.set_xlabel('Frequency')
    ax_histy.set_ylabel('Z')

    # Line plot for PCA reconstruction error
    ax_line.plot(st.session_state['reconstruction_errors_df']['Correlation Factor'], st.session_state['reconstruction_errors_df']['Reconstruction Error'], color='darkgreen', marker='o')
    ax_line.set_xlabel('Correlation Factor')
    ax_line.set_ylabel('PCA Reconstruction Error')
    ax_line.set_title('PCA Reconstruction Error vs Correlation Factor')
    #ax_line.set_xlim(-1, 1)
    ax_line.set_ylim(0, 0.5)


    # Add a color bar for reference
    cbar = fig.colorbar(scatter, ax=[ax_scatter, ax_histx, ax_histy], orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Correlation Factor')

    st.pyplot(fig)

# Streamlit slider for dynamic correlation factor
corr_factor = st.slider('Correlation factor', min_value=-1.0, max_value=1.0, step=0.1, value=-1.0)

# Update plot based on the current correlation factor
update_plot(corr_factor)
