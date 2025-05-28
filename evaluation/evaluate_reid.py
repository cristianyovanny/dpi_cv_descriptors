import os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def compute_cmc_map(query_features, query_labels, gallery_features, gallery_labels, top_k=10):
    query_features = np.array(query_features)
    gallery_features = np.array(gallery_features)

    cmc = np.zeros(top_k)
    average_precisions = []

    for i in range(len(query_features)):
        q_feat = query_features[i].reshape(1, -1)
        q_label = query_labels[i]

        dists = cosine_distances(q_feat, gallery_features)[0]
        sorted_indices = np.argsort(dists)
        sorted_labels = np.array(gallery_labels)[sorted_indices]
        matches = (sorted_labels == q_label)

        if not np.any(matches):
            continue

        # CMC
        rank = np.where(matches)[0][0]
        if rank < top_k:
            cmc[rank:] += 1

        # AP
        num_correct = 0
        precisions = []
        for rank_idx, is_match in enumerate(matches):
            if is_match:
                num_correct += 1
                precisions.append(num_correct / (rank_idx + 1))
        average_precisions.append(np.mean(precisions))

    cmc = cmc / len(query_features)
    mAP = np.mean(average_precisions)
    return cmc, mAP


def load_features(descriptor):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base_dir, 'features', descriptor)

    q_feats = np.load(os.path.join(features_path, 'query_feats.npy'), allow_pickle=True)
    q_labels = np.load(os.path.join(features_path, 'query_labels.npy'))
    g_feats = np.load(os.path.join(features_path, 'gallery_feats.npy'), allow_pickle=True)
    g_labels = np.load(os.path.join(features_path, 'gallery_labels.npy'))

    return np.vstack(q_feats), q_labels, np.vstack(g_feats), g_labels


if __name__ == '__main__':
    descriptor = 'hog'  # o 'gabor'
    q_feats, q_labels, g_feats, g_labels = load_features(descriptor)

    cmc, mAP = compute_cmc_map(q_feats, q_labels, g_feats, g_labels)

    print(f"\n--- Resultados para {descriptor.upper()} ---")
    print("CMC top-1:", cmc[0])
    print("CMC top-5:", cmc[4])
    print("CMC top-10:", cmc[9])
    print("mAP:", mAP)
