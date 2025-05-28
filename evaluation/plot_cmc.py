import os
import numpy as np
import matplotlib.pyplot as plt
from evaluation.evaluate_reid import compute_cmc_map, load_features


def plot_cmc_curve(cmc, descriptor):
    ranks = list(range(1, len(cmc) + 1))
    plt.plot(ranks, cmc, marker='o', label=f'{descriptor.upper()}')
    plt.title(f'CMC Curve for {descriptor.upper()} Descriptor')
    plt.xlabel('Rank')
    plt.ylabel('Recognition Rate')
    plt.xticks(ranks)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    descriptor = 'hog'  # Cambiar a 'gabor' para probar otro descriptor
    q_feats, q_labels, g_feats, g_labels = load_features(descriptor)

    cmc, mAP = compute_cmc_map(q_feats, q_labels, g_feats, g_labels, top_k=10)

    print(f"\n--- Resultados para {descriptor.upper()} ---")
    print("CMC top-1:", cmc[0])
    print("CMC top-5:", cmc[4])
    print("CMC top-10:", cmc[9])
    print("mAP:", mAP)

    plot_cmc_curve(cmc[:10], descriptor)
