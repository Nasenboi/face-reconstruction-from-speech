import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from scipy.io import loadmat


class BFMGenerator:
    def __init__(self, bfm_folder='./BFM', default_name='BFM_model_front.mat'):
        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            raise FileNotFoundError(f"Model not found at {os.path.join(bfm_folder, default_name)}")
        
        model = loadmat(os.path.join(bfm_folder, default_name))
        
        # Load BFM components
        self.mean_shape = model['meanshape'].astype(np.float32)
        self.id_base = torch.from_numpy(model['idBase'].astype(np.float32))
        self.face_buf = torch.from_numpy(model['tri'].astype(np.int64) - 1)
        self.keypoints = torch.from_numpy(np.squeeze(model['keypoints']).astype(np.int64) - 1)
        self.mean_tex = torch.from_numpy(model['meantex'].astype(np.float32) / 255.0)  # Average texture
        
        # Center mean shape
        mean_shape_reshaped = self.mean_shape.reshape([-1, 3])
        mean_shape_reshaped = mean_shape_reshaped - np.mean(mean_shape_reshaped, axis=0, keepdims=True)
        self.mean_shape = mean_shape_reshaped.reshape([-1, 1])
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_vertices = self.mean_shape.shape[0] // 3
        
    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if hasattr(value, 'device') or (type(value).__module__ == np.__name__ and key in ['mean_shape', 'id_base', 'face_buf', 'keypoints', 'mean_tex']):
                if type(value).__module__ == np.__name__:
                    setattr(self, key, torch.tensor(value).to(device))
                else:
                    setattr(self, key, value.to(device))
        return self
    
    def compute_face_shape(self, id_coeff):
        if isinstance(id_coeff, np.ndarray):
            id_coeff = torch.from_numpy(id_coeff).float()
        
        if id_coeff.dim() == 1:
            id_coeff = id_coeff.unsqueeze(0)
        
        if id_coeff.device != self.device:
            id_coeff = id_coeff.to(self.device)
        
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        face_shape = id_part + self.mean_shape.reshape([1, -1])
        vertices = face_shape.reshape([-1, 3])
        return vertices, self.face_buf
    
    def get_mesh_data(self, id_coeff):
        vertices, faces = self.compute_face_shape(id_coeff)
        landmarks = vertices[self.keypoints]
        
        # Get average face colors for each vertex
        colors = self.mean_tex.reshape([-1, 3]).cpu().numpy()
        
        return {
            'vertices': vertices.detach().cpu().numpy(),
            'faces': faces.detach().cpu().numpy(),
            'landmarks': landmarks.detach().cpu().numpy(),
            'colors': colors,
            'landmark_indices': self.keypoints.cpu().numpy()
        }
    
    def plot_comparison(self, id_coeffs_list, titles=None, plot_type='matplotlib', 
                   figsize=(20, 8), plotly_width=1000, plotly_height=500, elev=-90, azim=90):
        """
        Plot multiple faces for comparison
        
        Parameters:
            id_coeffs_list: list of ID coefficient arrays
            titles: list of titles for each subplot
            plot_type: 'matplotlib' or 'plotly'
            figsize: figure size for matplotlib (width, height)
            plotly_width: width for plotly plot
            plotly_height: height for plotly plot
        """
        n_faces = len(id_coeffs_list)
        
        if titles is None:
            titles = [f'Face {i+1}' for i in range(n_faces)]
        
        if plot_type == 'matplotlib':
            fig = plt.figure(figsize=figsize)
            
            for i, (id_coeff, title) in enumerate(zip(id_coeffs_list, titles)):
                ax = fig.add_subplot(1, n_faces, i+1, projection='3d')
                mesh_data = self.get_mesh_data(id_coeff)
                self.plot_mesh_matplotlib(mesh_data=mesh_data, ax=ax, 
                                        show_landmarks=False,
                                        view_elev=elev, view_azim=azim)
                ax.set_title(title, fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            return fig
        
        elif plot_type == 'plotly':
            fig = make_subplots(
                rows=1, cols=n_faces,
                specs=[[{'type': 'scene'} for _ in range(n_faces)]],
                subplot_titles=titles
            )
            
            for i, id_coeff in enumerate(id_coeffs_list):
                mesh_data = self.get_mesh_data(id_coeff)
                vertices = mesh_data['vertices']
                faces = mesh_data['faces']
                
                mesh_plot = go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1], 
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    vertexcolor=mesh_data['colors'],
                    opacity=1.0,
                    showscale=False,
                    lighting=dict(ambient=0.8, diffuse=0.9),
                    flatshading=True
                )
                
                fig.add_trace(mesh_plot, row=1, col=i+1)
                
                # Update scene properties
                fig.update_scenes(
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=0, y=0, z=1.8),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=1, z=0)
                    ),
                    xaxis_title='X',
                    yaxis_title='Y', 
                    zaxis_title='Z',
                    row=1, col=i+1
                )

            fig.update_layout(
                height=plotly_height, 
                width=plotly_width, 
                title_text="Face Comparison",
                title_x=0.5,
                font=dict(size=14)
            )
            return fig