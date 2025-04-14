import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def csv_to_tensor(cxv_path) -> torch.Tensor:
    device = input("Please enter the device (cpu or cuda): ")
    df= pd.read_csv(cxv_path,header=None)
    df = df.values[1:].astype(float)
    df = torch.tensor(df,device=device)
    return df

def kmeans_find_k(data:torch.Tensor,k:int):
    n_samples = data.size(0)
    n_features = data.shape[1]
    first_index = torch.randint(0,n_samples,(1,)).item()
    centers = [data[first_index]]

    for _ in range(1, k):
        centers_tensor = torch.stack(centers, dim=0)
        distances = torch.cdist(data, centers_tensor) ** 2
        min_distances, _ = distances.min(dim=1)

        epsilon = 1e-8
        min_distances = torch.clamp(min_distances, min=epsilon)
        probs = min_distances / min_distances.sum()

        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if probs.sum() > 0:
            probs = probs / probs.sum()  
        else:
            probs = torch.ones_like(probs) / probs.size(0)
        new_index = torch.multinomial(probs, 1).item()
        centers.append(data[new_index])
    
    return torch.stack(centers, dim=0)


    
if __name__ == "__main__":
    
    csv_path = 'data.csv'
    csv_tensor = csv_to_tensor(csv_path)
    start_time = time.time()
    data_nums=[200,1000,10000]
    k = 6
    for data_num in data_nums:
        print("working on sample number: ", data_num) 
        data = csv_tensor[:data_num]
        centers = kmeans_find_k(data, k)
        new_centers = torch.zeros_like(centers)
        kmeans_result = torch.zeros(data.size(0), dtype=torch.int)

        while not torch.equal(new_centers, centers):
            
            distance = torch.cdist(data, centers) ** 2
            min_distance, min_index = distance.min(dim=1)
            kmeans_result = torch.add(min_index, 1)
            
            new_centers = torch.zeros_like(centers)
            cluster_sizes = torch.zeros(k, device=centers.device)
            
            for i in range(data.size(0)):
                cluster_idx = min_index[i]
                new_centers[cluster_idx] += data[i]
                cluster_sizes[cluster_idx] += 1
                

            for i in range(k):
                new_centers[i] = new_centers[i] / cluster_sizes[i]

            centers = new_centers
                
            



        
        data_cpu = data.cpu().numpy()
        centers_cpu = centers.cpu().numpy()
        
        combined_data = np.vstack([data_cpu, centers_cpu])
        tsne = TSNE(n_components=2, random_state=42)
        combined_tsne = tsne.fit_transform(combined_data)
        
        data_tsne = combined_tsne[:data_cpu.shape[0]]
        centers_tsne = combined_tsne[data_cpu.shape[0]:]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=min_index.cpu().numpy(), cmap='viridis', alpha=0.6)
        plt.scatter(centers_tsne[:, 0], centers_tsne[:, 1], c='red', s=100, marker='X')
        plt.title(f'K-means Clustering with t-SNE (n={data_num})')
        plt.savefig(f'kmeans_tsne_{data_num}.png')
        plt.close()
